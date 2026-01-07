# Spec: Gradient Synchronization for Tensor Parallelism

## Overview
Implement backward pass support for TP layers, ensuring gradients are correctly synchronized across devices during training. This includes computing gradients for sharded parameters and coordinating communication patterns (all-reduce, all-gather) during backpropagation.

## Context
During training, each TP layer needs to:
1. Compute local gradients for its parameter shard
2. Synchronize gradients appropriately across ranks
3. Ensure gradient flow is mathematically equivalent to the non-parallel version

## Implementation Details

### 1. TP Gradient Synchronization Strategy

```
Column-Parallel Linear (Forward):
- Input: x (full)
- Weight: W_sharded (column split)
- Output: y_sharded (local), then optionally all-gather to y (full)

Column-Parallel Linear (Backward):
- Gradient: dy_full (or dy_sharded if output not gathered)
- Compute: dW_local = x^T * dy_local
- Compute: dx = W_local * dy_local
- If output was gathered in forward: need to slice dy_full to dy_sharded
- Weight gradients stay local (no sync needed)
- Input gradients are already full, no sync needed
```

```
Row-Parallel Linear (Forward):
- Input: x_sharded
- Weight: W_sharded (row split)
- Output: y_local, then all-reduce to y_full

Row-Parallel Linear (Backward):
- Gradient: dy_full
- All-reduce (already done in forward, gradient is full)
- Compute: dW_local = x_sharded^T * dy_full
- Compute: dx_sharded = W_local^T * dy_full
- Weight gradients: dW_local (stay local, no sync)
- Input gradients: dx_sharded (already sharded, matches forward)
```

### 2. Base TP Layer with Gradient Support

```csharp
namespace MLFramework.Layers.TensorParallel;

public abstract class TPGradientLayer : Module
{
    protected readonly TensorParallelGroup? _processGroup;
    protected readonly int _worldSize;
    protected readonly int _rank;

    protected TPGradientLayer(TensorParallelGroup? processGroup)
    {
        _processGroup = processGroup;
        _worldSize = TensorParallel.GetWorldSize();
        _rank = TensorParallel.GetRank();
    }

    protected abstract Tensor ForwardInternal(Tensor input);

    protected abstract Tensor BackwardInternal(Tensor gradOutput);

    public override Tensor Forward(Tensor input)
    {
        return ForwardInternal(input);
    }

    public override Tensor Backward(Tensor gradOutput)
    {
        return BackwardInternal(gradOutput);
    }
}
```

### 3. ColumnParallelLinear with Gradients

```csharp
public class ColumnParallelLinearGrad : TPGradientLayer
{
    private readonly int _inputSize;
    private readonly int _outputSize;
    private readonly bool _gatherOutput;

    private Tensor _weight; // [out/WorldSize, in]
    private Tensor? _bias;

    // Saved for backward
    private Tensor _lastInput;
    private bool _outputWasGathered;

    public ColumnParallelLinearGrad(
        int inputSize,
        int outputSize,
        bool bias = true,
        bool gatherOutput = false,
        TensorParallelGroup? processGroup = null)
        : base(processGroup)
    {
        _inputSize = inputSize;
        _outputSize = outputSize;
        _gatherOutput = gatherOutput;

        int shardOutSize = outputSize / _worldSize;
        _weight = InitializeWeight(shardOutSize, inputSize);

        if (bias)
        {
            _bias = Tensor.Zeros(shardOutSize);
        }
    }

    protected override Tensor ForwardInternal(Tensor input)
    {
        _lastInput = input; // Save for backward
        var outputLocal = Tensor.MatMul(input, _weight, transposeB: true);

        if (_bias != null)
        {
            outputLocal += _bias;
        }

        if (_gatherOutput)
        {
            _outputWasGathered = true;
            var comm = _processGroup?.Communicator ?? TensorParallel.GetCommunicator();
            return comm.AllGatherAsync(outputLocal, dim: -1).Result;
        }

        _outputWasGathered = false;
        return outputLocal;
    }

    protected override Tensor BackwardInternal(Tensor gradOutput)
    {
        Tensor gradOutputLocal = gradOutput;

        // If output was gathered in forward, slice gradient back to local shard
        if (_outputWasGathered)
        {
            int startIdx = _rank * (_outputSize / _worldSize);
            int endIdx = startIdx + (_outputSize / _worldSize);
            gradOutputLocal = gradOutput.Slice(-1, startIdx, endIdx);
        }

        // Compute gradient w.r.t. weight: dW = x^T * dy_local
        // _lastInput: [batch, ..., in]
        // gradOutputLocal: [batch, ..., out_shard]
        // Result: [out_shard, in]
        var gradWeight = Tensor.MatMul(gradOutputLocal, _lastInput, transposeA: true);

        // Compute gradient w.r.t. input: dx = dy_local * W^T
        // gradOutputLocal: [batch, ..., out_shard]
        // _weight: [out_shard, in]
        // Result: [batch, ..., in]
        var gradInput = Tensor.MatMul(gradOutputLocal, _weight);

        // Compute gradient w.r.t. bias (if present)
        if (_bias != null)
        {
            // Sum over all dimensions except the last one
            var gradBias = gradOutputLocal.Sum(dimensions: new[] { 0, 1 }.Skip(gradOutputLocal.Shape.Length - 1).ToArray());
            _bias.Grad += gradBias;
        }

        // Accumulate gradient w.r.t. weight (no sync needed - each rank has its shard)
        _weight.Grad += gradWeight;

        // Return gradient w.r.t. input (full, no sync needed)
        return gradInput;
    }

    private Tensor InitializeWeight(int outFeat, int inFeat)
    {
        double std = Math.Sqrt(2.0 / (inFeat + outFeat));
        return Tensor.RandomNormal(outFeat, inFeat, mean: 0.0, std: std);
    }
}
```

### 4. RowParallelLinear with Gradients

```csharp
public class RowParallelLinearGrad : TPGradientLayer
{
    private readonly int _inputSize;
    private readonly int _outputSize;
    private readonly bool _inputIsSharded;

    private Tensor _weight; // [out, in/WorldSize]
    private Tensor? _bias;

    // Saved for backward
    private Tensor _lastInputSharded;

    public RowParallelLinearGrad(
        int inputSize,
        int outputSize,
        bool bias = true,
        bool inputIsSharded = true,
        TensorParallelGroup? processGroup = null)
        : base(processGroup)
    {
        _inputSize = inputSize;
        _outputSize = outputSize;
        _inputIsSharded = inputIsSharded;

        int shardInSize = inputSize / _worldSize;
        _weight = InitializeWeight(outputSize, shardInSize);

        if (bias)
        {
            _bias = Tensor.Zeros(outputSize);
        }
    }

    protected override Tensor ForwardInternal(Tensor input)
    {
        if (_inputIsSharded)
        {
            _lastInputSharded = input; // Save for backward
        }
        else
        {
            // Slice input to our shard
            int startIdx = _rank * (_inputSize / _worldSize);
            int endIdx = startIdx + (_inputSize / _worldSize);
            _lastInputSharded = input.Slice(-1, startIdx, endIdx);
        }

        var outputLocal = Tensor.MatMul(_lastInputSharded, _weight, transposeB: true);

        // All-reduce to sum results from all ranks
        var comm = _processGroup?.Communicator ?? TensorParallel.GetCommunicator();
        var output = comm.AllReduceAsync(outputLocal, ReduceOperation.Sum).Result;

        if (_bias != null)
        {
            output += _bias;
        }

        return output;
    }

    protected override Tensor BackwardInternal(Tensor gradOutput)
    {
        // Gradient w.r.t. output is full (no slicing needed)
        // It was all-reduced in forward, so gradient is already summed

        // Compute gradient w.r.t. weight: dW = x_sharded^T * dy_full
        // _lastInputSharded: [batch, ..., in_shard]
        // gradOutput: [batch, ..., out]
        // Result: [out, in_shard]
        var gradWeight = Tensor.MatMul(gradOutput, _lastInputSharded, transposeA: true);

        // Compute gradient w.r.t. input: dx_sharded = dy_full * W^T
        // gradOutput: [batch, ..., out]
        // _weight: [out, in_shard]
        // Result: [batch, ..., in_shard]
        var gradInputSharded = Tensor.MatMul(gradOutput, _weight);

        // Compute gradient w.r.t. bias (if present)
        if (_bias != null)
        {
            var gradBias = gradOutput.Sum(dimensions: new[] { 0, 1 }.Skip(gradOutput.Shape.Length - 1).ToArray());
            _bias.Grad += gradBias;
        }

        // Accumulate gradient w.r.t. weight (no sync needed - each rank has its shard)
        _weight.Grad += gradWeight;

        // Return gradient w.r.t. input (sharded, no sync needed)
        return gradInputSharded;
    }

    private Tensor InitializeWeight(int outFeat, int inFeat)
    {
        double std = Math.Sqrt(2.0 / (inFeat + outFeat));
        return Tensor.RandomNormal(outFeat, inFeat, mean: 0.0, std: std);
    }
}
```

### 5. Gradient Accumulation Manager

```csharp
public static class TPGradientManager
{
    /// <summary>
    /// Synchronize gradients across all ranks before optimizer step
    /// For TP, weight gradients stay local (no sync needed)
    /// But for hybrid TP+DP, we may need to sync across DP groups
    /// </summary>
    public static async Task SynchronizeGradientsAsync(
        IEnumerable<Parameter> parameters,
        ProcessGroup? dpProcessGroup = null)
    {
        // For pure TP: gradients are already local, no sync needed
        // For TP+DP: sync across DP groups but not TP groups

        if (dpProcessGroup != null)
        {
            // Sync parameters across DP process group
            foreach (var param in parameters)
            {
                if (param.Grad != null)
                {
                    param.Grad = await dpProcessGroup.AllReduceAsync(
                        param.Grad,
                        ReduceOperation.Sum);
                }
            }
        }
    }

    /// <summary>
    /// Check that all ranks have received gradients (for debugging)
    /// </summary>
    public static async Task VerifyGradientsAsync(IEnumerable<Parameter> parameters)
    {
        var comm = TensorParallel.GetCommunicator();

        foreach (var param in parameters)
        {
            if (param.Grad == null)
            {
                throw new InvalidOperationException($"Parameter {param.Name} has no gradient");
            }

            // Check that grad is not all zeros (simple sanity check)
            var gradNorm = param.Grad.Norm();
            await comm.AllReduceAsync(gradNorm.ToScalar(), ReduceOperation.Sum);
        }
    }
}
```

### 6. Parameter GradAccessor Helpers

```csharp
public static class TPGradExtensions
{
    /// <summary>
    /// Get all trainable parameters from TP layers
    /// </summary>
    public static List<Parameter> GetTrainableParameters(this Module module)
    {
        var parameters = new List<Parameter>();
        CollectParameters(module, parameters);
        return parameters;
    }

    private static void CollectParameters(Module module, List<Parameter> parameters)
    {
        foreach (var param in module.Parameters)
        {
            if (param.RequiresGrad)
            {
                parameters.Add(param);
            }
        }

        foreach (var submodule in module.Modules)
        {
            CollectParameters(submodule, parameters);
        }
    }

    /// <summary>
    /// Zero all gradients in a module
    /// </summary>
    public static void ZeroGrad(this Module module)
    {
        foreach (var param in module.Parameters)
        {
            if (param.Grad != null)
            {
                param.Grad.Fill(0);
            }
        }
    }
}
```

## Files to Create

### Source Files
- `src/MLFramework/Layers/TensorParallel/TPGradientLayer.cs`
- `src/MLFramework/Layers/TensorParallel/ColumnParallelLinearGrad.cs`
- `src/MLFramework/Layers/TensorParallel/RowParallelLinearGrad.cs`
- `src/MLFramework/Layers/TensorParallel/TPGradientManager.cs`
- `src/MLFramework/Layers/TensorParallel/TPGradExtensions.cs`

### Test Files
- `tests/MLFramework.Tests/Layers/TensorParallel/TPGradientTests.cs`

## Test Requirements

1. **Column Parallel Gradient Tests**
   - Test gradient w.r.t. weight is computed correctly
   - Test gradient w.r.t. bias is computed correctly
   - Test gradient w.r.t. input is correct
   - Test backward pass with gathered output slices gradient correctly

2. **Row Parallel Gradient Tests**
   - Test gradient w.r.t. weight is computed correctly
   - Test gradient w.r.t. bias is computed correctly
   - Test gradient w.r.t. input is sharded correctly

3. **Gradient Accumulation Tests**
   - Test gradients accumulate correctly over multiple batches
   - Test ZeroGrad clears all gradients

4. **Integration Tests**
   - Test full forward+backward pass through TP MLP
   - Test gradients match non-parallel version (numerically)
   - Test that training converges with TP layers

5. **Edge Cases**
   - Test with no bias
   - Test with different batch sizes
   - Test with multiple sequential TP layers

## Dependencies
- `Module` base class from existing framework
- `Parameter` class with gradient tracking
- `Tensor` operations (MatMul, Sum, Norm, etc.)
- TP layers from previous specs
- Communication primitives for gradient sync (if hybrid)

## Success Criteria
- [ ] ColumnParallelLinearGrad computes correct gradients
- [ ] RowParallelLinearGrad computes correct gradients
- [ ] Gradients accumulate correctly over multiple batches
- [ ] ZeroGrad clears all gradients
- [ ] Numerical gradient checking matches non-parallel version
- [ ] Training with TP layers converges correctly
- [ ] Unit tests pass for all scenarios

## Estimated Time
45-60 minutes
