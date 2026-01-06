# Spec: LoRALinear Layer Implementation

## Overview
Implement the LoRALinear layer, which wraps a standard Linear layer and injects low-rank adapter matrices. This is the most critical LoRA component, as most LLM parameters are in linear layers (attention Q/K/V projections, MLP layers).

## Implementation Details

### 1. LoRALinear Class
**File**: `src/LoRA/LoRALinear.cs`

```csharp
public class LoRALinear : LoRAAdapterBase, IModule
{
    private readonly Linear _linearLayer;
    private readonly ITensor _loraA; // Rank x InDim
    private readonly ITensor _loraB; // OutDim x Rank
    private readonly float _dropoutRate;
    private readonly bool _useBias;
    private readonly ITensor? _loraBias; // Optional bias adapter
    private readonly Random? _dropoutRandom;

    public int InDim => _linearLayer.InFeatures;
    public int OutDim => _linearLayer.OutFeatures;

    public LoRALinear(Linear linearLayer, int rank, float alpha,
                      LoRAInitializationStrategy initialization = LoRAInitializationStrategy.Standard,
                      float dropout = 0.0f, bool useBias = false)
        : base(linearLayer, rank, alpha)
    {
        _linearLayer = linearLayer ?? throw new ArgumentNullException(nameof(linearLayer));
        _dropoutRate = dropout;
        _useBias = useBias;

        // Initialize LoRA matrices
        // A: Rank x InDim (initialized with Kaiming or Xavier)
        // B: OutDim x Rank (initialized with zeros)
        InitializeLoRAMatrices(initialization);

        if (_useBias)
        {
            _loraBias = Tensor.Zeros(new[] { OutDim });
        }

        if (_dropoutRate > 0.0f)
        {
            _dropoutRandom = new Random(42); // Fixed seed for reproducibility
        }
    }

    private void InitializeLoRAMatrices(LoRAInitializationStrategy strategy)
    {
        int inDim = _linearLayer.InFeatures;
        int outDim = _linearLayer.OutFeatures;

        switch (strategy)
        {
            case LoRAInitializationStrategy.Standard:
                // A: Kaiming normal, B: Zeros
                _loraA = Tensor.KaimingNormal(new[] { Rank, inDim });
                _loraB = Tensor.Zeros(new[] { outDim, Rank });
                break;

            case LoRAInitializationStrategy.Xavier:
                // Both: Xavier uniform
                _loraA = Tensor.XavierUniform(new[] { Rank, inDim });
                _loraB = Tensor.XavierUniform(new[] { outDim, Rank });
                break;

            case LoRAInitializationStrategy.Zero:
                // Both: Zeros
                _loraA = Tensor.Zeros(new[] { Rank, inDim });
                _loraB = Tensor.Zeros(new[] { outDim, Rank });
                break;

            default:
                throw new ArgumentException($"Unknown initialization strategy: {strategy}");
        }
    }

    public ITensor Forward(ITensor input)
    {
        // Standard forward pass through base layer
        var output = _linearLayer.Forward(input);

        if (!_isEnabled)
            return output;

        // LoRA forward pass: Wx + (alpha/r) * B(Ax)
        // Compute Ax (InDim -> Rank)
        var loraInput = Tensor.MatMul(input, _loraA.Transpose()); // [batch, in] x [in, rank] = [batch, rank]

        // Apply dropout if enabled (training mode)
        if (_dropoutRate > 0.0f && IsTrainingMode)
        {
            loraInput = ApplyDropout(loraInput);
        }

        // Compute B(Ax) (Rank -> OutDim)
        var loraOutput = Tensor.MatMul(loraInput, _loraB.Transpose()); // [batch, rank] x [rank, out] = [batch, out]

        // Scale by alpha/r
        loraOutput = loraOutput.Mul(ScalingFactor);

        // Add bias adapter if present
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

    private ITensor ApplyDropout(ITensor tensor)
    {
        var mask = Tensor.Random(tensor.Shape, _dropoutRandom);
        mask = Tensor.Where(mask.GreaterThan(_dropoutRate), 1.0f / (1.0f - _dropoutRate), 0.0f);
        return tensor.Mul(mask);
    }

    public bool IsTrainingMode { get; set; } = false;

    public override void FreezeBaseLayer()
    {
        // Mark base layer weights as frozen
        _linearLayer.Weight.RequiresGrad = false;
        if (_linearLayer.Bias != null)
        {
            _linearLayer.Bias.RequiresGrad = false;
        }
        _isBaseLayerFrozen = true;
    }

    public override void UnfreezeBaseLayer()
    {
        // Mark base layer weights as trainable
        _linearLayer.Weight.RequiresGrad = true;
        if (_linearLayer.Bias != null)
        {
            _linearLayer.Bias.RequiresGrad = true;
        }
        _isBaseLayerFrozen = false;
    }

    public override IEnumerable<ITensor> TrainableParameters
    {
        get
        {
            if (!_isBaseLayerFrozen)
            {
                yield return _linearLayer.Weight;
                if (_linearLayer.Bias != null)
                    yield return _linearLayer.Bias;
            }
            // Adapter parameters are always trainable
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
                yield return _linearLayer.Weight;
                if (_linearLayer.Bias != null)
                    yield return _linearLayer.Bias;
            }
        }
    }

    public override void MergeAdapter()
    {
        // Backup original weights
        _baseLayerWeightsBackup = _linearLayer.Weight.Clone();

        // W_new = W + (alpha/r) * B * A
        var deltaW = Tensor.MatMul(_loraB, _loraA); // [out, rank] x [rank, in] = [out, in]
        deltaW = deltaW.Mul(ScalingFactor);

        _linearLayer.Weight = _linearLayer.Weight.Add(deltaW);
    }

    public override void ResetBaseLayer()
    {
        if (_baseLayerWeightsBackup == null)
            throw new InvalidOperationException("No backup available. Cannot reset.");

        _linearLayer.Weight = _baseLayerWeightsBackup;
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

        // Validate shapes
        if (matrixA.Shape.Length != 2 || matrixA.Shape[0] != Rank || matrixA.Shape[1] != InDim)
            throw new ArgumentException($"Matrix A shape must be [{Rank}, {InDim}]");

        if (matrixB.Shape.Length != 2 || matrixB.Shape[0] != OutDim || matrixB.Shape[1] != Rank)
            throw new ArgumentException($"Matrix B shape must be [{OutDim}, {Rank}]");

        // Copy weights
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
            if (bias.Shape.Length != 1 || bias.Shape[0] != OutDim)
                throw new ArgumentException($"Bias shape must be [{OutDim}]");
            _loraBias.CopyFrom(bias);
        }
    }
}
```

### 2. Extension Method for Linear Layer
**File**: `src/LoRA/LoRAExtensions.cs`

```csharp
public static class LoRAExtensions
{
    public static LoRALinear AsLoRA(this Linear linear, int rank, float alpha,
                                     LoRAInitializationStrategy initialization = LoRAInitializationStrategy.Standard,
                                     float dropout = 0.0f, bool useBias = false)
    {
        return new LoRALinear(linear, rank, alpha, initialization, dropout, useBias);
    }
}
```

## Testing Requirements

**File**: `tests/LoRA/LoRALinearTests.cs`

1. **Constructor Tests**
   - Test wrapping various Linear layer sizes
   - Test different initialization strategies
   - Test configuration validation

2. **Forward Pass Tests**
   - Test output shape matches base layer
   - Test that disabled adapter returns base output
   - Test dropout functionality in training mode
   - Test scaling factor calculation

3. **Freeze/Unfreeze Tests**
   - Test FreezeBaseLayer correctly marks base weights
   - Test UnfreezeBaseLayer restores trainability
   - Test TrainableParameters property returns correct tensors

4. **Merge/Reset Tests**
   - Test MergeAdapter correctly updates base weights
   - Test ResetBaseLayer restores original weights
   - Test merge produces correct weight updates

5. **Adapter Weight Tests**
   - Test GetAdapterWeights returns correct tensors
   - Test SetAdapterWeights correctly updates matrices
   - Test shape validation in SetAdapterWeights

## Dependencies
- `Linear` layer (existing)
- `Tensor` class (existing)
- `ILoRAAdapter` interface (from spec 001)
- `LoRAAdapterBase` (from spec 001)

## Success Criteria
- LoRALinear correctly wraps Linear layers
- Forward pass produces expected outputs with LoRA adaptation
- Freeze/unfreeze functionality works correctly
- Adapter parameters are separate from base layer
- Merge/reset functionality preserves model integrity
- All unit tests pass

## Estimated Time
60 minutes

## Notes
- Ensure numerical stability for large models (FP32 for adapters even when base is FP16)
- Consider supporting mixed precision training
- The dropout should only apply during training mode
- Merge is destructive (modifies base weights) - must provide reset capability
