# Spec: LoRA Linear Layer

## Overview
Implement the core `LoraLinear` layer that wraps existing linear layers with low-rank adaptation.

## Requirements
- Wrap existing `Linear` layer with LoRA matrices A and B
- Compute scaling factor as alpha/rank
- Apply dropout to LoRA path
- Freeze base weights, make LoRA matrices trainable
- Support both forward pass and gradient computation

## Classes to Implement

### 1. LoraLinear
```csharp
public class LoraLinear : IModule
{
    private IModule _baseLinear;  // Frozen base layer
    private Parameter _loraA;     // [out_features, rank] - trainable
    private Parameter _loraB;     // [rank, in_features] - trainable
    private float _scaling;       // alpha / rank
    private float _dropout;
    private bool _merged;

    public LoraLinear(IModule baseLinear, int rank, int alpha, float dropout)
    {
        var inFeatures = baseLinear.InFeatures;
        var outFeatures = baseLinear.OutFeatures;

        _baseLinear = baseLinear;
        _baseLinear.Freeze();  // Freeze base weights

        // Initialize LoRA matrices
        _loraA = new Parameter(new Tensor(outFeatures, rank));
        _loraB = new Parameter(new Tensor(rank, inFeatures));

        // Initialization strategy:
        // - A: random normal with mean=0, std=0.01
        // - B: zeros (start with no effect)
        _loraA.Data.RandomNormal(0.0f, 0.01f);
        _loraB.Data.Zeros();

        _scaling = (float)alpha / rank;
        _dropout = dropout;
    }

    public Tensor Forward(Tensor x)
    {
        if (_merged)
        {
            return _baseLinear.Forward(x);
        }

        // Base computation (frozen)
        var baseOutput = _baseLinear.Forward(x);

        // LoRA computation: x @ B.T @ A.T * scaling
        var loraOutput = x.MatMul(_loraB.Data.Transpose());
        if (_dropout > 0)
        {
            loraOutput = Dropout(loraOutput, _dropout);
        }
        loraOutput = loraOutput.MatMul(_loraA.Data.Transpose()) * _scaling;

        return baseOutput + loraOutput;
    }

    /// <summary>Merge LoRA weights into base weights for inference</summary>
    public void Merge()
    {
        // W_merged = W_base + (B.T @ A.T) * scaling
        var deltaWeight = _loraB.Data.Transpose()
                               .MatMul(_loraA.Data.Transpose()) * _scaling;

        _baseLinear.Weight.Data += deltaWeight;
        _merged = true;
    }

    /// <summary>Unmerge LoRA weights (restore base weights)</summary>
    public void Unmerge()
    {
        if (!_merged) return;

        var deltaWeight = _loraB.Data.Transpose()
                               .MatMul(_loraA.Data.Transpose()) * _scaling;

        _baseLinear.Weight.Data -= deltaWeight;
        _merged = false;
    }

    /// <summary>Get trainable LoRA parameters</summary>
    public IEnumerable<Parameter> TrainableParameters()
    {
        if (_merged)
        {
            return Enumerable.Empty<Parameter>();
        }
        return new[] { _loraA, _loraB };
    }
}
```

## Implementation Details
- Use proper initialization: A with small random values, B with zeros
- Apply dropout only to LoRA path, not base path
- Merge/unmerge should be idempotent
- Support both 2D inputs [batch, in_features] and 3D [batch, seq, in_features]
- Ensure gradients flow correctly through LoRA matrices

## Deliverables
- `LoraLinear.cs` in `src/Core/LoRA/`
- Unit tests in `tests/Core/LoRA/`
