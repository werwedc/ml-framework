using System;
using System.Collections.Generic;
using RitterFramework.Core.Tensor;

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
    public (Dictionary<string, Tensor> Gradients, bool ShouldSkip) ProcessGradients(
        Dictionary<string, Tensor> gradients)
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
            return (new Dictionary<string, Tensor>(), true);
        }

        // Step 3: Unscale gradients
        var unscaledGradients = _lossScaler.UnscaleGradients(gradients);

        // Step 4: Clip gradients (if enabled)
        if (_options.EnableGradientClipping && _options.MaxGradNorm > 0)
        {
            unscaledGradients = ClipGradients(unscaledGradients);
        }

        // Step 5: Return gradients (already in appropriate format for optimizer)
        return (unscaledGradients, false);
    }

    /// <summary>
    /// Scales loss before backward pass
    /// </summary>
    public Tensor ScaleLoss(Tensor loss)
    {
        if (loss == null)
            throw new ArgumentNullException(nameof(loss));

        return _lossScaler.ScaleLoss(loss);
    }

    /// <summary>
    /// Clips gradients to maximum norm
    /// </summary>
    public Dictionary<string, Tensor> ClipGradients(Dictionary<string, Tensor> gradients)
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
    private float ComputeGradientNorm(Dictionary<string, Tensor> gradients)
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
    private Dictionary<string, Tensor> ApplyClipCoefficient(
        Dictionary<string, Tensor> gradients,
        float clipCoef)
    {
        var clipped = new Dictionary<string, Tensor>();

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
    public required LossScalerStats LossScalerStats { get; set; }

    public override string ToString()
    {
        return $"LossScale: {LossScale:F2}, " +
               $"LastGradNorm: {LastGradientNorm:F4}, " +
               $"ClipCount: {ClipCount}, " +
               $"ClippingEnabled: {ClippingEnabled}, " +
               $"MaxGradNorm: {MaxGradNorm:F2}";
    }
}
