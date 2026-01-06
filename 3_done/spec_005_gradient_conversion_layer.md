# Spec: GradientConversionLayer Component

## Overview
Implement the gradient conversion layer that handles gradient precision management, loss scaling, and gradient clipping.

## Dependencies
- Spec 002: MixedPrecisionOptions
- Spec 003: DynamicLossScaler
- Spec 004: PrecisionManager

## Implementation Details

### GradientConversionLayer Class
Create the class in `src/MLFramework/Optimizers/MixedPrecision/GradientConversionLayer.cs`:

```csharp
using System;
using System.Collections.Generic;

namespace MLFramework.Optimizers.MixedPrecision;

/// <summary>
/// Handles gradient conversion, loss scaling, and clipping for mixed-precision training
/// </summary>
public class GradientConversionLayer
{
    private readonly MixedPrecisionOptions _options;
    private readonly DynamicLossScaler _lossScaler;
    private readonly PrecisionManager _precisionManager;

    #region Properties

    /// <summary>
    /// Current loss scale factor
    /// </summary>
    public float LossScale => _lossScaler.CurrentScale;

    /// <summary>
    /// Number of gradient clipping operations performed
    /// </summary>
    public int ClipCount { get; private set; }

    /// <summary>
    /// Total norm of last clipped gradients
    /// </summary>
    public float LastGradientNorm { get; private set; }

    #endregion

    #region Constructors

    public GradientConversionLayer(
        MixedPrecisionOptions options,
        DynamicLossScaler lossScaler,
        PrecisionManager precisionManager)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _lossScaler = lossScaler ?? throw new ArgumentNullException(nameof(lossScaler));
        _precisionManager = precisionManager ?? throw new ArgumentNullException(nameof(precisionManager));

        ClipCount = 0;
        LastGradientNorm = 0;
    }

    public GradientConversionLayer(MixedPrecisionOptions options)
        : this(
            options,
            new DynamicLossScaler(options),
            new PrecisionManager(options))
    {
    }

    public GradientConversionLayer()
        : this(MixedPrecisionOptions.ForFP16())
    {
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Processes gradients through the full pipeline:
    /// 1. Check for overflow
    /// 2. Unscale gradients
    /// 3. Clip gradients (if enabled)
    /// 4. Convert to FP32
    /// </summary>
    /// <returns>
    /// Tuple of (processed_gradients, should_skip_optimizer_step)
    /// </returns>
    public (Dictionary<string, ITensor> Gradients, bool ShouldSkip) ProcessGradients(
        Dictionary<string, ITensor> gradients)
    {
        if (gradients == null)
            throw new ArgumentNullException(nameof(gradients));

        // Step 1: Check for overflow
        bool hasOverflow = _lossScaler.CheckOverflow(gradients);

        // Step 2: Update loss scale
        bool shouldSkip = _lossScaler.UpdateScale(hasOverflow);

        if (shouldSkip)
        {
            // Skip this step, return empty gradients
            return (new Dictionary<string, ITensor>(), true);
        }

        // Step 3: Unscale gradients
        var unscaledGradients = _lossScaler.UnscaleGradients(gradients);

        // Step 4: Clip gradients (if enabled)
        if (_options.EnableGradientClipping && _options.MaxGradNorm > 0)
        {
            unscaledGradients = ClipGradients(unscaledGradients);
        }

        // Step 5: Convert to FP32 for optimizer update
        var fp32Gradients = _precisionManager.ConvertToFP32(unscaledGradients);

        return (fp32Gradients, false);
    }

    /// <summary>
    /// Scales loss before backward pass
    /// </summary>
    public ITensor ScaleLoss(ITensor loss)
    {
        if (loss == null)
            throw new ArgumentNullException(nameof(loss));

        return _lossScaler.ScaleLoss(loss);
    }

    /// <summary>
    /// Clips gradients to maximum norm
    /// </summary>
    public Dictionary<string, ITensor> ClipGradients(Dictionary<string, ITensor> gradients)
    {
        if (gradients == null)
            throw new ArgumentNullException(nameof(gradients));

        if (!_options.EnableGradientClipping || _options.MaxGradNorm <= 0)
            return gradients;

        // Compute total gradient norm
        float totalNorm = ComputeGradientNorm(gradients);
        LastGradientNorm = totalNorm;

        // Clip if norm exceeds threshold
        if (totalNorm > _options.MaxGradNorm)
        {
            float clipCoef = _options.MaxGradNorm / (totalNorm + 1e-6f);
            return ApplyClipCoefficient(gradients, clipCoef);
        }

        return gradients;
    }

    /// <summary>
    /// Gets statistics about gradient processing
    /// </summary>
    public GradientProcessingStats GetStats()
    {
        return new GradientProcessingStats
        {
            LossScale = _lossScaler.CurrentScale,
            ClipCount = ClipCount,
            LastGradientNorm = LastGradientNorm,
            ClippingEnabled = _options.EnableGradientClipping,
            MaxGradNorm = _options.MaxGradNorm,
            LossScalerStats = _lossScaler.GetStats()
        };
    }

    /// <summary>
    /// Resets internal counters and statistics
    /// </summary>
    public void Reset()
    {
        ClipCount = 0;
        LastGradientNorm = 0;
        _lossScaler.Reset();
    }

    #endregion

    #region Private Methods

    /// <summary>
    /// Computes the total L2 norm of all gradients
    /// </summary>
    private float ComputeGradientNorm(Dictionary<string, ITensor> gradients)
    {
        float totalNorm = 0;

        foreach (var grad in gradients.Values)
        {
            // Compute squared norm of each gradient
            // TODO: Implement based on tensor API
            // For now, return placeholder
            float gradNorm = 0;
            totalNorm += gradNorm * gradNorm;
        }

        return (float)Math.Sqrt(totalNorm);
    }

    /// <summary>
    /// Applies clipping coefficient to all gradients
    /// </summary>
    private Dictionary<string, ITensor> ApplyClipCoefficient(
        Dictionary<string, ITensor> gradients,
        float clipCoef)
    {
        var clipped = new Dictionary<string, ITensor>();

        foreach (var kvp in gradients)
        {
            // Clip gradient: grad = grad * clipCoef
            // TODO: Implement based on tensor API
            clipped[kvp.Key] = kvp.Value;  // Placeholder
        }

        ClipCount++;
        return clipped;
    }

    #endregion
}

/// <summary>
/// Statistics about gradient processing
/// </summary>
public class GradientProcessingStats
{
    public float LossScale { get; set; }
    public int ClipCount { get; set; }
    public float LastGradientNorm { get; set; }
    public bool ClippingEnabled { get; set; }
    public float MaxGradNorm { get; set; }
    public LossScalerStats LossScalerStats { get; set; }

    public override string ToString()
    {
        return $"LossScale: {LossScale:F2}, " +
               $"LastGradNorm: {LastGradientNorm:F4}, " +
               $"ClipCount: {ClipCount}, " +
               $"ClippingEnabled: {ClippingEnabled}, " +
               $"MaxGradNorm: {MaxGradNorm:F2}";
    }
}
```

## Requirements

### Functional Requirements
1. **Gradient Processing Pipeline**: Check overflow → Unscale → Clip → Convert to FP32
2. **Loss Scaling**: Scale loss before backward pass
3. **Gradient Clipping**: Clip gradients to max norm (optional)
4. **Norm Computation**: Compute L2 norm of gradients
5. **Skip Logic**: Return skip flag when overflow detected
6. **Statistics**: Track clipping count and gradient norms
7. **Reset**: Clear internal counters

### Non-Functional Requirements
1. **Performance**: Efficient norm computation (avoid extra copies)
2. **Numerical Stability**: Handle edge cases (empty gradients, zero norm)
3. **Memory**: Minimize temporary allocations
4. **Modularity**: Separable components for testing

## Gradient Processing Flow

### Standard Flow
1. Receive gradients from backward pass (in target precision)
2. Check for overflow (NaN/Inf)
3. Unscale gradients (divide by loss scale)
4. Clip gradients if norm exceeds threshold
5. Convert to FP32 for optimizer update
6. Return (fp32_gradients, false)

### Overflow Flow
1. Receive gradients from backward pass
2. Detect overflow
3. Update loss scale (reduce)
4. Return (empty_gradients, true) to skip optimizer step

## Clipping Algorithm

```
total_norm = sqrt(sum(grad^2 for all grads))
if total_norm > max_grad_norm:
    clip_coef = max_grad_norm / total_norm
    grad = grad * clip_coef for all grads
```

## Deliverables

### Source Files
1. `src/MLFramework/Optimizers/MixedPrecision/GradientConversionLayer.cs`

### Unit Tests
- Tests will be covered in spec 010 (PrecisionManager unit tests)

## Notes for Coder
- ComputeGradientNorm should stub for now (tensor API not ready)
- ApplyClipCoefficient should stub for now (tensor API not ready)
- Focus on the pipeline logic and state management
- Ensure proper handling of edge cases (empty gradients, zero norm)
- Clipping coefficient should be computed with epsilon to prevent division by zero
- Statistics should be useful for debugging and monitoring
- The skip flag is critical for training stability
