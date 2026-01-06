# Spec: LoRAConv2d Layer Implementation

## Overview
Implement the LoRAConv2d layer, which wraps a standard Conv2d layer and injects low-rank adapter matrices for vision models (e.g., Vision Transformers, CNNs). This enables parameter-efficient fine-tuning of computer vision models.

## Implementation Details

### 1. LoRAConv2d Class
**File**: `src/LoRA/LoRAConv2d.cs`

```csharp
public class LoRAConv2d : LoRAAdapterBase, IModule
{
    private readonly Conv2d _convLayer;
    private readonly ITensor _loraA; // Rank x InChannels
    private readonly ITensor _loraB; // OutChannels x Rank
    private readonly float _dropoutRate;
    private readonly bool _useBias;
    private readonly ITensor? _loraBias;
    private readonly Random? _dropoutRandom;

    public int InChannels => _convLayer.InChannels;
    public int OutChannels => _convLayer.OutChannels;
    public int KernelSize => _convLayer.KernelSize;

    public LoRAConv2d(Conv2d convLayer, int rank, float alpha,
                      LoRAInitializationStrategy initialization = LoRAInitializationStrategy.Standard,
                      float dropout = 0.0f, bool useBias = false)
        : base(convLayer, rank, alpha)
    {
        _convLayer = convLayer ?? throw new ArgumentNullException(nameof(convLayer));
        _dropoutRate = dropout;
        _useBias = useBias;

        // Initialize LoRA matrices
        InitializeLoRAMatrices(initialization);

        if (_useBias)
        {
            _loraBias = Tensor.Zeros(new[] { OutChannels });
        }

        if (_dropoutRate > 0.0f)
        {
            _dropoutRandom = new Random(42);
        }
    }

    private void InitializeLoRAMatrices(LoRAInitializationStrategy strategy)
    {
        int inChannels = _convLayer.InChannels;
        int outChannels = _convLayer.OutChannels;
        int kernelSize = _convLayer.KernelSize;

        // Flatten spatial dimensions for LoRA
        // For Conv2d: [Out, In, K, K] -> [Out, In*K*K]
        // LoRA operates on [Out, In*K*K] dimension
        int flattenedInDim = inChannels * kernelSize * kernelSize;

        switch (strategy)
        {
            case LoRAInitializationStrategy.Standard:
                _loraA = Tensor.KaimingNormal(new[] { Rank, flattenedInDim });
                _loraB = Tensor.Zeros(new[] { outChannels, Rank });
                break;

            case LoRAInitializationStrategy.Xavier:
                _loraA = Tensor.XavierUniform(new[] { Rank, flattenedInDim });
                _loraB = Tensor.XavierUniform(new[] { outChannels, Rank });
                break;

            case LoRAInitializationStrategy.Zero:
                _loraA = Tensor.Zeros(new[] { Rank, flattenedInDim });
                _loraB = Tensor.Zeros(new[] { outChannels, Rank });
                break;

            default:
                throw new ArgumentException($"Unknown initialization strategy: {strategy}");
        }
    }

    public ITensor Forward(ITensor input)
    {
        // Standard forward pass through base layer
        var output = _convLayer.Forward(input);

        if (!_isEnabled)
            return output;

        // LoRA forward pass for Conv2d
        // Need to flatten spatial dims for LoRA computation
        int batchSize = input.Shape[0];
        int height = input.Shape[2];
        int width = input.Shape[3];
        int kernelSize = _convLayer.KernelSize;

        // Extract patches using im2col
        var patches = Im2Col(input, kernelSize, _convLayer.Padding, _convLayer.Stride);

        // Compute LoRA: W_patch + (alpha/r) * B * A * patch
        // Flatten for LoRA: [batch, out_positions, in*k*k]
        var patchesFlat = patches.Reshape(new[] { -1, _loraA.Shape[1] });

        // Apply LoRA: A * patch [rank, in*k*k] x [batch*positions, in*k*k]
        var loraInput = Tensor.MatMul(patchesFlat, _loraA.Transpose());

        // Apply dropout if enabled
        if (_dropoutRate > 0.0f && IsTrainingMode)
        {
            loraInput = ApplyDropout(loraInput);
        }

        // Apply B: [out, rank] x [batch*positions, rank]
        var loraOutput = Tensor.MatMul(loraInput, _loraB.Transpose());

        // Scale by alpha/r
        loraOutput = loraOutput.Mul(ScalingFactor);

        // Reshape and add to output
        loraOutput = loraOutput.Reshape(new[] { batchSize, OutChannels, height, width });

        // Add bias if present
        if (_loraBias != null)
        {
            output = output.Add(loraOutput.Add(_loraBias));
        }
        else
        {
            output = output.Add(loraOutput);
        }

        return output;
    }

    private ITensor Im2Col(ITensor input, int kernelSize, int padding, int stride)
    {
        // Extract sliding window patches from input
        // Implementation depends on framework's tensor operations
        // This is a standard operation in CNN frameworks
        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int outHeight = (height + 2 * padding - kernelSize) / stride + 1;
        int outWidth = (width + 2 * padding - kernelSize) / stride + 1;

        // Allocate output tensor: [batch, out_h, out_w, in_c, k, k]
        var output = Tensor.Zeros(new[] { batchSize, outHeight, outWidth, channels, kernelSize, kernelSize });

        // Extract patches (simplified - actual implementation varies by framework)
        for (int b = 0; b < batchSize; b++)
        {
            for (int oy = 0; oy < outHeight; oy++)
            {
                for (int ox = 0; ox < outWidth; ox++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        for (int ky = 0; ky < kernelSize; ky++)
                        {
                            for (int kx = 0; kx < kernelSize; kx++)
                            {
                                int iy = oy * stride - padding + ky;
                                int ix = ox * stride - padding + kx;

                                if (iy >= 0 && iy < height && ix >= 0 && ix < width)
                                {
                                    output[b, oy, ox, c, ky, kx] = input[b, c, iy, ix];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Reshape to [batch, out_h * out_w, in_c * k * k]
        return output.Reshape(new[] { batchSize, outHeight * outWidth, channels * kernelSize * kernelSize });
    }

    private ITensor ApplyDropout(ITensor tensor)
    {
        var mask = Tensor.Random(tensor.Shape, _dropoutRandom);
        mask = Tensor.Where(mask.GreaterThan(_dropoutRate), 1.0f / (1.0f - _dropoutRate), 0.0f);
        return tensor.Mul(mask);
    }

    public bool IsTrainingMode { get; set; } = false;

    public override void FreezeBaseLayer()
    {
        _convLayer.Weight.RequiresGrad = false;
        if (_convLayer.Bias != null)
        {
            _convLayer.Bias.RequiresGrad = false;
        }
        _isBaseLayerFrozen = true;
    }

    public override void UnfreezeBaseLayer()
    {
        _convLayer.Weight.RequiresGrad = true;
        if (_convLayer.Bias != null)
        {
            _convLayer.Bias.RequiresGrad = true;
        }
        _isBaseLayerFrozen = false;
    }

    public override IEnumerable<ITensor> TrainableParameters
    {
        get
        {
            if (!_isBaseLayerFrozen)
            {
                yield return _convLayer.Weight;
                if (_convLayer.Bias != null)
                    yield return _convLayer.Bias;
            }
            yield return _loraA;
            yield return _loraB;
            if (_loraBias != null)
                yield return _loraBias;
        }
    }

    public override IEnumerable<ITensor> FrozenParameters
    {
        get
        {
            if (_isBaseLayerFrozen)
            {
                yield return _convLayer.Weight;
                if (_convLayer.Bias != null)
                    yield return _convLayer.Bias;
            }
        }
    }

    public override void MergeAdapter()
    {
        // Backup original weights
        _baseLayerWeightsBackup = _convLayer.Weight.Clone();

        // Flatten and reshape for LoRA merge
        // Base weight: [out, in, k, k] -> [out, in*k*k]
        int outChannels = _convLayer.OutChannels;
        int kernelSize = _convLayer.KernelSize;

        var weightFlat = _convLayer.Weight.Reshape(new[] { outChannels, -1 });

        // W_new = W + (alpha/r) * B * A
        var deltaW = Tensor.MatMul(_loraB, _loraA);
        deltaW = deltaW.Mul(ScalingFactor);

        var newWeight = weightFlat.Add(deltaW);
        _convLayer.Weight = newWeight.Reshape(_convLayer.Weight.Shape);
    }

    public override void ResetBaseLayer()
    {
        if (_baseLayerWeightsBackup == null)
            throw new InvalidOperationException("No backup available. Cannot reset.");

        _convLayer.Weight = _baseLayerWeightsBackup;
        _baseLayerWeightsBackup = null;
    }

    public override (ITensor? MatrixA, ITensor? MatrixB) GetAdapterWeights()
    {
        return (_loraA, _loraB);
    }

    public override void SetAdapterWeights(ITensor? matrixA, ITensor? matrixB)
    {
        if (matrixA == null || matrixB == null)
            throw new ArgumentNullException("Adapter weights cannot be null");

        int flattenedInDim = _convLayer.InChannels * _convLayer.KernelSize * _convLayer.KernelSize;

        // Validate shapes
        if (matrixA.Shape.Length != 2 || matrixA.Shape[0] != Rank || matrixA.Shape[1] != flattenedInDim)
            throw new ArgumentException($"Matrix A shape must be [{Rank}, {flattenedInDim}]");

        if (matrixB.Shape.Length != 2 || matrixB.Shape[0] != OutChannels || matrixB.Shape[1] != Rank)
            throw new ArgumentException($"Matrix B shape must be [{OutChannels}, {Rank}]");

        _loraA.CopyFrom(matrixA);
        _loraB.CopyFrom(matrixB);
    }

    public ITensor? GetBias()
    {
        return _loraBias;
    }

    public void SetBias(ITensor? bias)
    {
        if (bias != null)
        {
            if (bias.Shape.Length != 1 || bias.Shape[0] != OutChannels)
                throw new ArgumentException($"Bias shape must be [{OutChannels}]");
            _loraBias.CopyFrom(bias);
        }
    }
}
```

### 2. Extension Method
**File**: `src/LoRA/LoRAExtensions.cs`

```csharp
public static class LoRAExtensions
{
    public static LoRAConv2d AsLoRA(this Conv2d conv, int rank, float alpha,
                                   LoRAInitializationStrategy initialization = LoRAInitializationStrategy.Standard,
                                   float dropout = 0.0f, bool useBias = false)
    {
        return new LoRAConv2d(conv, rank, alpha, initialization, dropout, useBias);
    }
}
```

## Testing Requirements

**File**: `tests/LoRA/LoRAConv2dTests.cs`

1. **Constructor Tests**
   - Test wrapping various Conv2d layer sizes
   - Test different initialization strategies
   - Test kernel size handling

2. **Forward Pass Tests**
   - Test output shape matches base layer
   - Test im2col patch extraction
   - Test that disabled adapter returns base output
   - Test dropout in training mode

3. **Freeze/Unfreeze Tests**
   - Test correct parameter freezing
   - Test TrainableParameters property

4. **Merge/Reset Tests**
   - Test MergeAdapter correctly updates conv weights
   - Test ResetBaseLayer restores original weights
   - Verify weight shape preservation

5. **Adapter Weight Tests**
   - Test GetAdapterWeights returns correct shapes
   - Test SetAdapterWeights validates input shapes

## Dependencies
- `Conv2d` layer (existing)
- `Tensor` class (existing)
- `ILoRAAdapter` interface (from spec 001)
- `LoRAAdapterBase` (from spec 001)

## Success Criteria
- LoRAConv2d correctly wraps Conv2d layers
- Im2col extraction works correctly for various kernel sizes
- Forward pass produces expected outputs with LoRA adaptation
- Adapter weights correctly flatten/unflatten
- All unit tests pass

## Estimated Time
60 minutes

## Notes
- Im2col implementation may need optimization for production use
- Consider using framework's built-in unfold/unroll operations if available
- Conv2d weight flattening must be consistent across merge/reset
- Dropout should only apply during training mode
