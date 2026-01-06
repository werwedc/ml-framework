using System;
using System.Collections.Generic;
using RitterFramework.Core.Tensor;

namespace MLFramework.Optimizers.MixedPrecision;

/// <summary>
/// Dynamically adjusts loss scale to prevent underflow during mixed-precision training
/// </summary>
public class DynamicLossScaler
{
    private readonly MixedPrecisionOptions _options;
    private float _currentScale;
    private int _stepsSinceLastOverflow;
    private int _consecutiveOverflows;

    #region Properties

    /// <summary>
    /// Current loss scale factor
    /// </summary>
    public float CurrentScale => _currentScale;

    /// <summary>
    /// Number of consecutive overflows detected
    /// </summary>
    public int ConsecutiveOverflows => _consecutiveOverflows;

    /// <summary>
    /// Number of successful steps since last overflow
    /// </summary>
    public int StepsSinceLastOverflow => _stepsSinceLastOverflow;

    /// <summary>
    /// Total number of overflows detected since creation
    /// </summary>
    public int TotalOverflows { get; private set; }

    /// <summary>
    /// Whether dynamic loss scaling is enabled
    /// </summary>
    public bool IsEnabled => _options.EnableDynamicLossScaling;

    #endregion

    #region Constructors

    public DynamicLossScaler(MixedPrecisionOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _options.Validate();

        _currentScale = _options.InitialLossScale;
        _stepsSinceLastOverflow = 0;
        _consecutiveOverflows = 0;
        TotalOverflows = 0;
    }

    public DynamicLossScaler()
        : this(MixedPrecisionOptions.ForFP16())
    {
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Scales the loss tensor before backward pass
    /// </summary>
    public Tensor ScaleLoss(Tensor loss)
    {
        if (loss == null)
            throw new ArgumentNullException(nameof(loss));

        if (!IsEnabled)
            return loss;

        // Scale the loss by current scale factor
        return loss * _currentScale;
    }

    /// <summary>
    /// Unscales the gradients after backward pass
    /// </summary>
    public Dictionary<string, Tensor> UnscaleGradients(Dictionary<string, Tensor> gradients)
    {
        if (gradients == null)
            throw new ArgumentNullException(nameof(gradients));

        if (!IsEnabled)
            return gradients;

        var unscaled = new Dictionary<string, Tensor>();
        foreach (var kvp in gradients)
        {
            // Divide each gradient by current scale factor
            unscaled[kvp.Key] = DivideByScalar(kvp.Value, _currentScale);
        }

        return unscaled;
    }

    /// <summary>
    /// Checks if gradients contain overflow (NaN or Inf)
    /// </summary>
    public bool CheckOverflow(Dictionary<string, Tensor> gradients)
    {
        if (gradients == null)
            throw new ArgumentNullException(nameof(gradients));

        if (!IsEnabled)
            return false;

        foreach (var grad in gradients.Values)
        {
            if (HasOverflow(grad))
                return true;
        }

        return false;
    }

    /// <summary>
    /// Updates the scale factor based on overflow detection
    /// </summary>
    /// <returns>
    /// True if the step should be skipped due to overflow, false otherwise
    /// </returns>
    public bool UpdateScale(bool hadOverflow)
    {
        if (!IsEnabled)
            return false;

        if (hadOverflow)
        {
            return HandleOverflow();
        }
        else
        {
            return HandleSuccess();
        }
    }

    /// <summary>
    /// Convenience method: Check overflow and update scale in one call
    /// </summary>
    public bool CheckOverflowAndUpdate(Dictionary<string, Tensor> gradients)
    {
        bool hasOverflow = CheckOverflow(gradients);
        return UpdateScale(hasOverflow);
    }

    /// <summary>
    /// Resets the scaler to initial state
    /// </summary>
    public void Reset()
    {
        _currentScale = _options.InitialLossScale;
        _stepsSinceLastOverflow = 0;
        _consecutiveOverflows = 0;
        TotalOverflows = 0;
    }

    /// <summary>
    /// Gets statistics about scaler behavior
    /// </summary>
    public LossScalerStats GetStats()
    {
        return new LossScalerStats
        {
            CurrentScale = _currentScale,
            StepsSinceLastOverflow = _stepsSinceLastOverflow,
            ConsecutiveOverflows = _consecutiveOverflows,
            TotalOverflows = TotalOverflows,
            GrowthInterval = _options.GrowthInterval,
            MaxConsecutiveOverflows = _options.MaxConsecutiveOverflows,
            IsStable = _consecutiveOverflows < _options.MaxConsecutiveOverflows
        };
    }

    #endregion

    #region Private Methods

    private bool HandleOverflow()
    {
        TotalOverflows++;
        _consecutiveOverflows++;

        // Reduce scale by backoff factor
        _currentScale = Math.Max(
            _currentScale * _options.BackoffFactor,
            _options.MinLossScale
        );

        // Reset successful step counter
        _stepsSinceLastOverflow = 0;

        // Step should be skipped
        return true;
    }

    private bool HandleSuccess()
    {
        _consecutiveOverflows = 0;
        _stepsSinceLastOverflow++;

        // Grow scale after growth interval steps
        if (_stepsSinceLastOverflow >= _options.GrowthInterval)
        {
            _currentScale = Math.Min(
                _currentScale * _options.GrowthFactor,
                _options.MaxLossScale
            );

            _stepsSinceLastOverflow = 0;
        }

        // Step should proceed
        return false;
    }

    private bool HasOverflow(Tensor tensor)
    {
        // Check for NaN or Inf values by iterating through tensor data
        // We'll check each element in the tensor
        int[] shape = tensor.Shape;
        int totalElements = 1;
        foreach (int dim in shape)
        {
            totalElements *= dim;
        }

        // For each element, we need to construct indices
        int[] indices = new int[shape.Length];
        for (int i = 0; i < totalElements; i++)
        {
            // Convert flat index to multi-dimensional indices
            int temp = i;
            for (int j = shape.Length - 1; j >= 0; j--)
            {
                indices[j] = temp % shape[j];
                temp /= shape[j];
            }

            float value = tensor[indices];

            // Check for NaN or Infinity
            if (float.IsNaN(value) || float.IsInfinity(value))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Divides a tensor by a scalar value
    /// </summary>
    private Tensor DivideByScalar(Tensor tensor, float scalar)
    {
        // Since Tensor class doesn't have division operator, we multiply by reciprocal
        if (scalar == 0)
            throw new DivideByZeroException("Cannot divide by zero");

        return tensor * (1.0f / scalar);
    }

    #endregion
}

/// <summary>
/// Statistics about the loss scaler
/// </summary>
public class LossScalerStats
{
    public float CurrentScale { get; set; }
    public int StepsSinceLastOverflow { get; set; }
    public int ConsecutiveOverflows { get; set; }
    public int TotalOverflows { get; set; }
    public int GrowthInterval { get; set; }
    public int MaxConsecutiveOverflows { get; set; }
    public bool IsStable { get; set; }

    public override string ToString()
    {
        return $"Scale: {CurrentScale:F2}, " +
               $"Steps since overflow: {StepsSinceLastOverflow}, " +
               $"Consecutive overflows: {ConsecutiveOverflows}, " +
               $"Total overflows: {TotalOverflows}, " +
               $"Stable: {IsStable}";
    }
}
