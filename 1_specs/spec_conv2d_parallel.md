# Spec: Conv2d Tensor Parallelism (Channel-wise)

## Overview
Implement tensor-parallel versions of `Conv2d` layers that split channels across multiple devices. This enables parallelizing convolutional layers in CNNs/ViT architectures where the number of output channels exceeds device memory.

## Context
For convolutional layers, we parallelize along the channel dimension:
- **Input channel parallelism**: Split input channels across devices
- **Output channel parallelism**: Split output filters across devices (more common)

## Implementation Details

### 1. Output Channel Parallel Conv2d (Most Common)

```csharp
namespace MLFramework.Layers.TensorParallel;

public class Conv2dOutputParallel : Conv2dLayer
{
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _worldSize;
    private readonly int _rank;
    private readonly bool _gatherOutput;
    private readonly TensorParallelGroup? _processGroup;

    // Sharded weight: [out_channels / world_size, in_channels, kernel_h, kernel_w]
    private Tensor _weight;

    private Tensor? _bias; // Optional bias: [out_channels / world_size]

    public Conv2dOutputParallel(
        int inChannels,
        int outChannels,
        int kernelSize,
        bool gatherOutput = false,
        TensorParallelGroup? processGroup = null,
        int stride = 1,
        int padding = 0,
        int dilation = 1,
        int groups = 1,
        bool bias = true)
        : base(inChannels, outChannels, kernelSize, stride, padding, dilation, groups, bias)
    {
        _inChannels = inChannels;
        _outChannels = outChannels;
        _worldSize = TensorParallel.GetWorldSize();
        _rank = TensorParallel.GetRank();
        _gatherOutput = gatherOutput;
        _processGroup = processGroup;

        // Validate dimensions
        if (outChannels % _worldSize != 0)
        {
            throw new ArgumentException(
                $"outChannels ({outChannels}) must be divisible by worldSize ({_worldSize})");
        }

        if (groups != 1 && groups != _worldSize)
        {
            throw new ArgumentException(
                "For channel parallel Conv2d, groups must be 1 or equal to worldSize");
        }

        // Calculate sharded dimensions
        int shardOutChannels = outChannels / _worldSize;
        int kernelH = kernelSize;
        int kernelW = kernelSize;

        // Initialize sharded weight
        // Shape: [out_channels / world_size, in_channels, kernel_h, kernel_w]
        _weight = InitializeWeight(shardOutChannels, inChannels, kernelH, kernelW);

        if (bias)
        {
            // Sharded bias: [out_channels / world_size]
            _bias = Tensor.Zeros(shardOutChannels);
        }
    }

    /// <summary>
    /// Forward pass for output-channel parallel Conv2d
    /// Input:  [batch, in_channels, height, width]
    /// Weight: [out_channels / world_size, in_channels, kernel_h, kernel_w]
    /// Output: [batch, out_channels / world_size, height_out, width_out] (if !gatherOutput)
    ///         [batch, out_channels, height_out, width_out] (if gatherOutput)
    /// </summary>
    public override Tensor Forward(Tensor input)
    {
        // Perform local convolution
        // input: [batch, in_channels, h, w]
        // weight: [out_shard, in_channels, kernel_h, kernel_w]
        // output_local: [batch, out_shard, h_out, w_out]
        var outputLocal = Tensor.Conv2d(
            input,
            _weight,
            stride: Stride,
            padding: Padding,
            dilation: Dilation);

        // Add bias if present
        if (_bias != null)
        {
            // bias: [out_shard]
            // broadcast to [batch, out_shard, h_out, w_out]
            var biasBroadcast = _bias.Reshape(1, _bias.Shape[0], 1, 1);
            outputLocal += biasBroadcast;
        }

        // Optionally gather output across ranks
        if (_gatherOutput)
        {
            // Gather along the channel dimension (dim=1)
            // Result: [batch, out_channels, h_out, w_out]
            var comm = _processGroup?.Communicator ?? TensorParallel.GetCommunicator();
            return comm.AllGatherAsync(outputLocal, dim: 1).Result;
        }

        return outputLocal;
    }

    private Tensor InitializeWeight(int outFeat, int inFeat, int kernelH, int kernelW)
    {
        // Use Kaiming/He initialization for ReLU-friendly networks
        double std = Math.Sqrt(2.0 / (inFeat * kernelH * kernelW));
        return Tensor.RandomNormal(outFeat, inFeat, kernelH, kernelW, mean: 0.0, std: std);
    }
}
```

### 2. Input Channel Parallel Conv2d (Less Common)

```csharp
public class Conv2dInputParallel : Conv2dLayer
{
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _worldSize;
    private readonly int _rank;
    private readonly TensorParallelGroup? _processGroup;

    // Sharded weight: [out_channels, in_channels / world_size, kernel_h, kernel_w]
    private Tensor _weight;

    private Tensor? _bias; // Bias: [out_channels] (not sharded)

    public Conv2dInputParallel(
        int inChannels,
        int outChannels,
        int kernelSize,
        TensorParallelGroup? processGroup = null,
        int stride = 1,
        int padding = 0,
        int dilation = 1,
        int groups = 1,
        bool bias = true)
        : base(inChannels, outChannels, kernelSize, stride, padding, dilation, groups, bias)
    {
        _inChannels = inChannels;
        _outChannels = outChannels;
        _worldSize = TensorParallel.GetWorldSize();
        _rank = TensorParallel.GetRank();
        _processGroup = processGroup;

        // Validate dimensions
        if (inChannels % _worldSize != 0)
        {
            throw new ArgumentException(
                $"inChannels ({inChannels}) must be divisible by worldSize ({_worldSize})");
        }

        // Calculate sharded dimensions
        int shardInChannels = inChannels / _worldSize;
        int kernelH = kernelSize;
        int kernelW = kernelSize;

        // Initialize sharded weight
        _weight = InitializeWeight(outChannels, shardInChannels, kernelH, kernelW);

        if (bias)
        {
            // Bias is NOT sharded: [out_channels]
            _bias = Tensor.Zeros(outChannels);
        }
    }

    /// <summary>
    /// Forward pass for input-channel parallel Conv2d
    /// Input:  [batch, in_channels / world_size, h, w] (sharded)
    /// Weight: [out_channels, in_channels / world_size, kernel_h, kernel_w]
    /// Output: [batch, out_channels, h_out, w_out] (after all-reduce)
    /// </summary>
    public override Tensor Forward(Tensor input)
    {
        // Verify input is sharded
        int expectedInChannels = _inChannels / _worldSize;
        if (input.Shape[1] != expectedInChannels)
        {
            throw new InvalidOperationException(
                $"Expected sharded input with {expectedInChannels} channels, got {input.Shape[1]}");
        }

        // Perform local convolution
        // input: [batch, in_shard, h, w]
        // weight: [out_channels, in_shard, kernel_h, kernel_w]
        // output_local: [batch, out_channels, h_out, w_out]
        var outputLocal = Tensor.Conv2d(
            input,
            _weight,
            stride: Stride,
            padding: Padding,
            dilation: Dilation);

        // All-reduce to sum results from all ranks
        var comm = _processGroup?.Communicator ?? TensorParallel.GetCommunicator();
        var output = comm.AllReduceAsync(outputLocal, ReduceOperation.Sum).Result;

        // Add bias if present (after all-reduce)
        if (_bias != null)
        {
            var biasBroadcast = _bias.Reshape(1, _bias.Shape[0], 1, 1);
            output += biasBroadcast;
        }

        return output;
    }

    private Tensor InitializeWeight(int outFeat, int inFeat, int kernelH, int kernelW)
    {
        double std = Math.Sqrt(2.0 / (inFeat * kernelH * kernelW));
        return Tensor.RandomNormal(outFeat, inFeat, kernelH, kernelW, mean: 0.0, std: std);
    }
}
```

### 3. Factory Methods

```csharp
public static class TPConv2dFactory
{
    /// <summary>
    /// Create output-channel parallel Conv2d (most common)
    /// </summary>
    public static Conv2dOutputParallel CreateOutputParallel(
        int inChannels,
        int outChannels,
        int kernelSize,
        bool gatherOutput = false,
        int stride = 1,
        int padding = 0,
        bool bias = true)
    {
        return new Conv2dOutputParallel(
            inChannels, outChannels, kernelSize,
            gatherOutput: gatherOutput,
            stride: stride,
            padding: padding,
            bias: bias);
    }

    /// <summary>
    /// Create input-channel parallel Conv2d
    /// </summary>
    public static Conv2dInputParallel CreateInputParallel(
        int inChannels,
        int outChannels,
        int kernelSize,
        int stride = 1,
        int padding = 0,
        bool bias = true)
    {
        return new Conv2dInputParallel(
            inChannels, outChannels, kernelSize,
            stride: stride,
            padding: padding,
            bias: bias);
    }

    /// <summary>
    /// Create standard pattern: output-parallel then input-parallel
    /// Useful for bottleneck blocks
    /// </summary>
    public static (Conv2dOutputParallel, Conv2dInputParallel) CreateBottleneckPair(
        int inChannels,
        int bottleneckChannels,
        int outChannels,
        int kernelSize = 1,
        int stride = 1)
    {
        var conv1 = new Conv2dOutputParallel(
            inChannels, bottleneckChannels, kernelSize,
            gatherOutput: false,
            stride: 1,
            padding: 0,
            bias: false);

        var conv2 = new Conv2dInputParallel(
            bottleneckChannels, outChannels, kernelSize,
            stride: stride,
            padding: 0,
            bias: false);

        return (conv1, conv2);
    }
}
```

## Files to Create

### Source Files
- `src/MLFramework/Layers/TensorParallel/Conv2dOutputParallel.cs`
- `src/MLFramework/Layers/TensorParallel/Conv2dInputParallel.cs`
- `src/MLFramework/Layers/TensorParallel/TPConv2dFactory.cs`

### Test Files
- `tests/MLFramework.Tests/Layers/TensorParallel/Conv2dParallelTests.cs`

## Test Requirements

1. **Output Channel Parallel Tests**
   - Test forward pass with sharded output returns correct dimensions
   - Test gatherOutput=true performs all-gather correctly
   - Test gatherOutput=false keeps output sharded
   - Verify output channel dimension: [batch, out/WorldSize, h, w] or [batch, out, h, w]

2. **Input Channel Parallel Tests**
   - Test forward pass with sharded input produces correct all-reduced output
   - Verify all-reduce sums results across ranks
   - Verify output dimension: [batch, out_channels, h_out, w_out]

3. **Weight Sharding Tests**
   - Verify output-parallel weight shape: [out/WorldSize, in, kH, kW]
   - Verify input-parallel weight shape: [out, in/WorldSize, kH, kW]
   - Verify bias shapes are correct

4. **Bias Tests**
   - Test that bias is correctly added
   - Test that input-parallel bias is added after all-reduce (once)

5. **Edge Cases**
   - Test outChannels not divisible by worldSize throws exception (output-parallel)
   - Test inChannels not divisible by worldSize throws exception (input-parallel)
   - Test different kernel sizes, strides, padding
   - Test with groups parameter

6. **Integration Tests**
   - Test Conv2dOutputParallel → Conv2dInputParallel pipeline
   - Test in a simple CNN architecture
   - Test combined with linear layers

## Dependencies
- `Conv2dLayer` base class from existing framework
- `TensorParallel` context manager
- `ICommunicator` and `TensorParallelGroup` from communication primitives
- Tensor operations (Conv2d, RandomNormal, Zeros, Reshape, etc.)

## Success Criteria
- [ ] Conv2dOutputParallel correctly shards output channels
- [ ] Conv2dInputParallel correctly shards input channels
- [ ] All-gather works correctly for output-parallel layers
- [ ] All-reduce works correctly for input-parallel layers
- [ ] Bias is handled correctly in both variants
- [ ] Works with different kernel sizes, strides, padding
- [ ] Factory methods create appropriately configured layers
- [ ] Bottleneck pattern works correctly
- [ ] Unit tests pass for all scenarios
- [ ] Exceptions thrown when dimensions are not divisible by worldSize

## Estimated Time
45-60 minutes
