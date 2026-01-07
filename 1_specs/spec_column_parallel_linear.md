# Spec: Column Parallel Linear Layer

## Overview
Implement `ColumnParallelLinear` layer that splits the weight matrix along the column dimension across multiple devices. This enables training models with output dimensions larger than a single device's memory capacity.

## Context
In column parallelism, the output dimension is split across devices. Each device computes a portion of the output, which can then be kept sharded (for subsequent row-parallel layers) or gathered (for final outputs).

## Implementation Details

### 1. ColumnParallelLinear Class

```csharp
namespace MLFramework.Layers.TensorParallel;

public class ColumnParallelLinear : LinearLayer
{
    private readonly int _inputSize;
    private readonly int _outputSize;
    private readonly int _worldSize;
    private readonly int _rank;
    private readonly bool _gatherOutput;
    private readonly TensorParallelGroup? _processGroup;

    // The sharded weight matrix: [output_size / world_size, input_size]
    // Each rank holds a different column slice of the full weight matrix
    private Tensor _weight;

    private Tensor? _bias; // Optional bias: [output_size / world_size]

    public ColumnParallelLinear(
        int inputSize,
        int outputSize,
        bool bias = true,
        bool gatherOutput = false,
        TensorParallelGroup? processGroup = null)
        : base(inputSize, outputSize, bias)
    {
        _inputSize = inputSize;
        _outputSize = outputSize;
        _worldSize = TensorParallel.GetWorldSize();
        _rank = TensorParallel.GetRank();
        _gatherOutput = gatherOutput;
        _processGroup = processGroup;

        // Calculate sharded dimensions
        int shardOutputSize = outputSize / _worldSize;
        if (outputSize % _worldSize != 0)
        {
            throw new ArgumentException(
                $"outputSize ({outputSize}) must be divisible by worldSize ({_worldSize})");
        }

        // Initialize sharded weight
        // Shape: [output_size / world_size, input_size]
        _weight = InitializeWeight(shardOutputSize, inputSize);

        if (bias)
        {
            // Sharded bias: [output_size / world_size]
            _bias = Tensor.Zeros(shardOutputSize);
        }
    }

    /// <summary>
    /// Forward pass for column parallel linear layer
    /// Input:  [batch_size, ..., input_size]
    /// Weight: [output_size / world_size, input_size]
    /// Output: [batch_size, ..., output_size / world_size] (if !gatherOutput)
    ///         [batch_size, ..., output_size] (if gatherOutput)
    /// </summary>
    public override Tensor Forward(Tensor input)
    {
        // Perform local matrix multiplication
        // input: [batch_size, ..., input_size]
        // weight: [output_shard, input_size]
        // output_local: [batch_size, ..., output_shard]
        var outputLocal = Tensor.MatMul(input, _weight, transposeB: true);

        // Add bias if present
        if (_bias != null)
        {
            outputLocal += _bias;
        }

        // Optionally gather output across ranks
        if (_gatherOutput)
        {
            // Gather along the output dimension (last dimension)
            // Result: [batch_size, ..., output_size]
            var comm = _processGroup?.Communicator ?? TensorParallel.GetCommunicator();
            return comm.AllGatherAsync(outputLocal, dim: -1).Result;
        }

        return outputLocal;
    }

    /// <summary>
    /// Get the local weight shard (for inspection, not modification)
    /// </summary>
    public Tensor GetLocalWeight() => _weight;

    /// <summary>
    /// Get the local bias shard (for inspection, not modification)
    /// </summary>
    public Tensor? GetLocalBias() => _bias;

    /// <summary>
    /// Get the shape of the local weight shard
    /// </summary>
    public (int rows, int cols) GetLocalWeightShape()
    {
        return (_outputSize / _worldSize, _inputSize);
    }

    /// <summary>
    /// Initialize weight with appropriate scaling
    /// </summary>
    private Tensor InitializeWeight(int outFeatures, int inFeatures)
    {
        // Use Xavier/Glorot initialization
        double std = Math.Sqrt(2.0 / (inFeatures + outFeatures));
        return Tensor.RandomNormal(outFeatures, inFeatures, mean: 0.0, std: std);
    }
}
```

### 2. Column Parallel Linear with Input Gathering (for non-sharded inputs)

```csharp
public class ColumnParallelLinearWithInputGather : ColumnParallelLinear
{
    private readonly bool _inputIsSharded;

    public ColumnParallelLinearWithInputGather(
        int inputSize,
        int outputSize,
        bool bias = true,
        bool gatherOutput = false,
        bool inputIsSharded = false,
        TensorParallelGroup? processGroup = null)
        : base(inputSize, outputSize, bias, gatherOutput, processGroup)
    {
        _inputIsSharded = inputIsSharded;
    }

    /// <summary>
    /// Forward pass that optionally gathers input first
    /// </summary>
    public override Tensor Forward(Tensor input)
    {
        Tensor readyInput = input;

        // If input is sharded across previous row-parallel layer, gather it
        if (_inputIsSharded)
        {
            var comm = _processGroup?.Communicator ?? TensorParallel.GetCommunicator();
            readyInput = comm.AllGatherAsync(input, dim: -1).Result;
        }

        return base.Forward(readyInput);
    }
}
```

### 3. Helper for Creating TP-Ready Layers

```csharp
public static class ColumnParallelLinearFactory
{
    /// <summary>
    /// Create a column parallel linear layer with sensible defaults
    /// </summary>
    public static ColumnParallelLinear Create(
        int inputSize,
        int outputSize,
        bool bias = true,
        bool gatherOutput = false)
    {
        return new ColumnParallelLinear(inputSize, outputSize, bias, gatherOutput);
    }

    /// <summary>
    /// Create for transformer attention projection (no bias, gather output)
    /// </summary>
    public static ColumnParallelLinear CreateForAttention(
        int inputSize,
        int outputSize)
    {
        return new ColumnParallelLinear(
            inputSize,
            outputSize,
            bias: false,
            gatherOutput: true);
    }

    /// <summary>
    /// Create for MLP hidden layer (with bias, don't gather - feed to row parallel)
    /// </summary>
    public static ColumnParallelLinear CreateForMLPHidden(
        int inputSize,
        int hiddenSize)
    {
        return new ColumnParallelLinear(
            inputSize,
            hiddenSize,
            bias: true,
            gatherOutput: false);
    }
}
```

### 4. Integration with Layer Registration

```csharp
// Ensure ColumnParallelLinear is registered in the layer factory
public static class LayerRegistryExtensions
{
    public static void RegisterTensorParallelLayers(this LayerRegistry registry)
    {
        registry.Register<ColumnParallelLinear>("column_parallel_linear");
        registry.Register<ColumnParallelLinearWithInputGather>("column_parallel_linear_with_input_gather");
    }
}
```

## Files to Create

### Source Files
- `src/MLFramework/Layers/TensorParallel/ColumnParallelLinear.cs`
- `src/MLFramework/Layers/TensorParallel/ColumnParallelLinearWithInputGather.cs`
- `src/MLFramework/Layers/TensorParallel/ColumnParallelLinearFactory.cs`
- `src/MLFramework/Layers/TensorParallel/LayerRegistryExtensions.cs` (if registry exists)

### Test Files
- `tests/MLFramework.Tests/Layers/TensorParallel/ColumnParallelLinearTests.cs`

## Test Requirements

1. **Basic Functionality Tests**
   - Test forward pass returns correct shape (sharded vs gathered)
   - Test output dimensions are correct: [batch, output/WorldSize] or [batch, output]
   - Test bias is correctly added

2. **Weight Sharding Tests**
   - Verify weight shard shape is [output/world_size, input_size]
   - Verify each rank has different weight values
   - Test that concatenated weights across ranks equal full weight (in unit test)

3. **Output Gathering Tests**
   - Test that gatherOutput=true performs all-gather
   - Test that gatherOutput=false keeps output sharded
   - Verify gathered output matches expected full output

4. **Process Group Tests**
   - Test that custom process group is used for communication
   - Verify operations only affect group members

5. **Edge Cases**
   - Test outputSize not divisible by worldSize throws exception
   - Test with zero bias
   - Test with different batch sizes

6. **Integration Tests**
   - Test in conjunction with RowParallelLinear
   - Test gradient flow (when gradient sync is implemented)

## Dependencies
- `LinearLayer` base class from existing framework
- `TensorParallel` context manager
- `ICommunicator` and `TensorParallelGroup` from communication primitives
- Tensor operations (MatMul, RandomNormal, Zeros, etc.)

## Success Criteria
- [ ] ColumnParallelLinear correctly shards weights across ranks
- [ ] Forward pass produces correctly sharded or gathered output
- [ ] Bias is correctly added to local output
- [ ] Output dimensions are correct
- [ ] Works with default TP context and custom process groups
- [ ] Factory methods create appropriately configured layers
- [ ] Unit tests pass for all scenarios
- [ ] Exception thrown when outputSize not divisible by worldSize

## Estimated Time
45-60 minutes
