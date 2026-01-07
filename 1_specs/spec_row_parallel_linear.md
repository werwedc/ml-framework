# Spec: Row Parallel Linear Layer

## Overview
Implement `RowParallelLinear` layer that splits the weight matrix along the row dimension across multiple devices. Each device computes with a portion of the input, and results are aggregated via all-reduce. This is typically paired with column-parallel layers in MLP/attention blocks.

## Context
In row parallelism, the input dimension is split across devices. Each device performs computation with a portion of the input features, and the results are summed across all devices to produce the final output.

## Implementation Details

### 1. RowParallelLinear Class

```csharp
namespace MLFramework.Layers.TensorParallel;

public class RowParallelLinear : LinearLayer
{
    private readonly int _inputSize;
    private readonly int _outputSize;
    private readonly int _worldSize;
    private readonly int _rank;
    private readonly bool _inputIsSharded;
    private readonly TensorParallelGroup? _processGroup;

    // The sharded weight matrix: [output_size, input_size / world_size]
    // Each rank holds a different row slice of the full weight matrix
    private Tensor _weight;

    private Tensor? _bias; // Optional bias: [output_size] (not sharded, shared)

    public RowParallelLinear(
        int inputSize,
        int outputSize,
        bool bias = true,
        bool inputIsSharded = true,
        TensorParallelGroup? processGroup = null)
        : base(inputSize, outputSize, bias)
    {
        _inputSize = inputSize;
        _outputSize = outputSize;
        _worldSize = TensorParallel.GetWorldSize();
        _rank = TensorParallel.GetRank();
        _inputIsSharded = inputIsSharded;
        _processGroup = processGroup;

        // Calculate sharded dimensions
        int shardInputSize = inputSize / _worldSize;
        if (inputSize % _worldSize != 0)
        {
            throw new ArgumentException(
                $"inputSize ({inputSize}) must be divisible by worldSize ({_worldSize})");
        }

        // Initialize sharded weight
        // Shape: [output_size, input_size / world_size]
        _weight = InitializeWeight(outputSize, shardInputSize);

        if (bias)
        {
            // Bias is NOT sharded: [output_size]
            // All ranks use the same bias values
            _bias = Tensor.Zeros(outputSize);
        }
    }

    /// <summary>
    /// Forward pass for row parallel linear layer
    /// Input:  [batch_size, ..., input_size] (full)
    ///         [batch_size, ..., input_size / world_size] (if sharded)
    /// Weight: [output_size, input_size / world_size]
    /// Output: [batch_size, ..., output_size] (after all-reduce)
    /// </summary>
    public override Tensor Forward(Tensor input)
    {
        Tensor readyInput = input;

        // If input is supposed to be sharded but isn't, we can still work
        // But typically, input comes from previous column-parallel layer
        if (_inputIsSharded)
        {
            // Input should already be sharded from previous column-parallel layer
            // Verify shape matches expected shard size
            int expectedInputSize = _inputSize / _worldSize;
            var inputShape = input.Shape;
            int lastDim = inputShape[^1];

            if (lastDim != expectedInputSize)
            {
                throw new InvalidOperationException(
                    $"Expected sharded input with last dim={expectedInputSize}, " +
                    $"but got {lastDim}. If input is not sharded, set inputIsSharded=false.");
            }
        }
        else
        {
            // Input is not sharded, we need to slice it to our shard
            int startIdx = _rank * (_inputSize / _worldSize);
            int endIdx = startIdx + (_inputSize / _worldSize);
            readyInput = input.Slice(-1, startIdx, endIdx);
        }

        // Perform local matrix multiplication
        // readyInput: [batch_size, ..., input_shard]
        // weight: [output_size, input_shard]
        // output_local: [batch_size, ..., output_size]
        var outputLocal = Tensor.MatMul(readyInput, _weight, transposeB: true);

        // All-reduce to sum results from all ranks
        var comm = _processGroup?.Communicator ?? TensorParallel.GetCommunicator();
        var output = comm.AllReduceAsync(outputLocal, ReduceOperation.Sum).Result;

        // Add bias if present (after all-reduce, so bias added once)
        if (_bias != null)
        {
            output += _bias;
        }

        return output;
    }

    /// <summary>
    /// Get the local weight shard (for inspection, not modification)
    /// </summary>
    public Tensor GetLocalWeight() => _weight;

    /// <summary>
    /// Get the local bias (shared across all ranks)
    /// </summary>
    public Tensor? GetLocalBias() => _bias;

    /// <summary>
    /// Get the shape of the local weight shard
    /// </summary>
    public (int rows, int cols) GetLocalWeightShape()
    {
        return (_outputSize, _inputSize / _worldSize);
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

### 2. Row Parallel Linear with Input Gathering

```csharp
public class RowParallelLinearWithInputGather : RowParallelLinear
{
    public RowParallelLinearWithInputGather(
        int inputSize,
        int outputSize,
        bool bias = true,
        bool inputIsSharded = false,
        TensorParallelGroup? processGroup = null)
        : base(inputSize, outputSize, bias, inputIsSharded, processGroup)
    {
    }

    /// <summary>
    /// Forward pass that automatically gathers sharded input if needed
    /// </summary>
    public override Tensor Forward(Tensor input)
    {
        Tensor readyInput = input;

        // If input is sharded (from previous column-parallel), gather it first
        // This is less efficient but more flexible
        if (_inputIsSharded)
        {
            var comm = _processGroup?.Communicator ?? TensorParallel.GetCommunicator();
            readyInput = comm.AllGatherAsync(input, dim: -1).Result;
        }

        // Now proceed with non-sharded input path
        // Slice our portion of the input
        int startIdx = _rank * (_inputSize / _worldSize);
        int endIdx = startIdx + (_inputSize / _worldSize);
        var slicedInput = readyInput.Slice(-1, startIdx, endIdx);

        // Local matmul
        var outputLocal = Tensor.MatMul(slicedInput, _weight, transposeB: true);

        // All-reduce
        var output = _processGroup?.Communicator?.AllReduceAsync(outputLocal, ReduceOperation.Sum).Result
                   ?? TensorParallel.GetCommunicator().AllReduceAsync(outputLocal, ReduceOperation.Sum).Result;

        // Add bias
        if (_bias != null)
        {
            output += _bias;
        }

        return output;
    }
}
```

### 3. Helper for Creating TP-Ready Layers

```csharp
public static class RowParallelLinearFactory
{
    /// <summary>
    /// Create a row parallel linear layer with sensible defaults
    /// </summary>
    public static RowParallelLinear Create(
        int inputSize,
        int outputSize,
        bool bias = true,
        bool inputIsSharded = true)
    {
        return new RowParallelLinear(inputSize, outputSize, bias, inputIsSharded);
    }

    /// <summary>
    /// Create for MLP output layer (typically after column-parallel hidden)
    /// </summary>
    public static RowParallelLinear CreateForMLPOutput(
        int hiddenSize,
        int outputSize,
        bool bias = true)
    {
        return new RowParallelLinear(
            hiddenSize,
            outputSize,
            bias: bias,
            inputIsSharded: true);
    }

    /// <summary>
    /// Create with input gathering (for flexibility)
    /// </summary>
    public static RowParallelLinearWithInputGather CreateWithGather(
        int inputSize,
        int outputSize,
        bool bias = true)
    {
        return new RowParallelLinearWithInputGather(inputSize, outputSize, bias, inputIsSharded: false);
    }
}
```

### 4. Combined Column+Row Pattern (MLP Block Helper)

```csharp
/// <summary>
/// Helper to create the standard column-then-row parallel MLP pattern
/// This is the most common pattern in Transformer models
/// </summary>
public static class TPMLPFactory
{
    public static (ColumnParallelLinear, RowParallelLinear) CreateMLPBlock(
        int inputSize,
        int hiddenSize,
        int outputSize,
        bool bias = true)
    {
        var columnLayer = new ColumnParallelLinear(
            inputSize,
            hiddenSize,
            bias: bias,
            gatherOutput: false);

        var rowLayer = new RowParallelLinear(
            hiddenSize,
            outputSize,
            bias: bias,
            inputIsSharded: true);

        return (columnLayer, rowLayer);
    }

    /// <summary>
    /// Forward pass through the combined MLP block
    /// </summary>
    public static Tensor ForwardMLP(
        Tensor input,
        ColumnParallelLinear columnLayer,
        RowParallelLinear rowLayer,
        Func<Tensor, Tensor> activation)
    {
        var hidden = columnLayer.Forward(input);
        var activated = activation(hidden);
        var output = rowLayer.Forward(activated);
        return output;
    }
}
```

## Files to Create

### Source Files
- `src/MLFramework/Layers/TensorParallel/RowParallelLinear.cs`
- `src/MLFramework/Layers/TensorParallel/RowParallelLinearWithInputGather.cs`
- `src/MLFramework/Layers/TensorParallel/RowParallelLinearFactory.cs`
- `src/MLFramework/Layers/TensorParallel/TPMLPFactory.cs`

### Test Files
- `tests/MLFramework.Tests/Layers/TensorParallel/RowParallelLinearTests.cs`

## Test Requirements

1. **Basic Functionality Tests**
   - Test forward pass with sharded input returns correct full output
   - Test forward pass with unsharded input (slices and processes correctly)
   - Test output dimensions: [batch_size, ..., output_size]

2. **All-Reduce Tests**
   - Verify all-reduce operation sums results across ranks
   - Test that output matches expected sum of rank-local computations

3. **Weight Sharding Tests**
   - Verify weight shard shape is [output_size, input_size/world_size]
   - Verify each rank has different weight values
   - Test that concatenated weights across ranks equal full weight (in test)

4. **Bias Tests**
   - Test that bias is added after all-reduce (added once, not multiple times)
   - Verify bias values are the same across all ranks

5. **Input Handling Tests**
   - Test with inputIsSharded=true (input should be sharded)
   - Test with inputIsSharded=false (input is sliced locally)
   - Test error thrown when sharded input has wrong dimensions

6. **Integration Tests**
   - Test ColumnParallelLinear → RowParallelLinear pipeline
   - Verify output of combined MLP is correct
   - Test with different world sizes

7. **Edge Cases**
   - Test inputSize not divisible by worldSize throws exception
   - Test with zero bias
   - Test with different batch sizes

## Dependencies
- `LinearLayer` base class from existing framework
- `TensorParallel` context manager
- `ICommunicator` and `TensorParallelGroup` from communication primitives
- `ColumnParallelLinear` from previous spec
- Tensor operations (MatMul, RandomNormal, Zeros, Slice, etc.)

## Success Criteria
- [ ] RowParallelLinear correctly shards weights across ranks
- [ ] Forward pass with sharded input produces correct all-reduced output
- [ ] Forward pass with unsharded input slices and processes correctly
- [ ] All-reduce correctly sums local outputs from all ranks
- [ ] Bias is added once after all-reduce (not per-rank)
- [ ] Works with default TP context and custom process groups
- [ ] Factory methods create appropriately configured layers
- [ ] MLP block helper works correctly
- [ ] Unit tests pass for all scenarios
- [ ] Exception thrown when inputSize not divisible by worldSize

## Estimated Time
45-60 minutes
