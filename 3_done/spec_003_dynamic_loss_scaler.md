# Spec: DynamicLossScaler Component

## Overview
Implement the dynamic loss scaler that automatically adjusts the loss scale factor to prevent underflow while maximizing precision.

## Dependencies
- Spec 002: MixedPrecisionOptions

## Implementation Details

### DynamicLossScaler Class
Create the class in `src/MLFramework/Optimizers/MixedPrecision/DynamicLossScaler.cs`:

```csharp
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
        options.Validate();

        _currentScale = options.InitialLossScale;
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
    public ITensor ScaleLoss(ITensor loss)
    {
        if (loss == null)
            throw new ArgumentNullException(nameof(loss));

        if (!IsEnabled)
            return loss;

        // Scale the loss by current scale factor
        return loss.Multiply(_currentScale);
    }

    /// <summary>
    /// Unscales the gradients after backward pass
    /// </summary>
    public Dictionary<string, ITensor> UnscaleGradients(Dictionary<string, ITensor> gradients)
    {
        if (gradients == null)
            throw new ArgumentNullException(nameof(gradients));

        if (!IsEnabled)
            return gradients;

        var unscaled = new Dictionary<string, ITensor>();
        foreach (var kvp in gradients)
        {
            // Divide each gradient by current scale factor
            unscaled[kvp.Key] = kvp.Value.Divide(_currentScale);
        }

        return unscaled;
    }

    /// <summary>
    /// Checks if gradients contain overflow (NaN or Inf)
    /// </summary>
    public bool CheckOverflow(Dictionary<string, ITensor> gradients)
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
    public bool CheckOverflowAndUpdate(Dictionary<string, ITensor> gradients)
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

    private bool HasOverflow(ITensor tensor)
    {
        // Check for NaN or Inf values
        // TODO: Implement based on tensor API
        return false;
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
```

## Requirements

### Functional Requirements
1. **Loss Scaling**: Must scale loss before backward pass
2. **Gradient Unscaling**: Must unscale gradients after backward pass
3. **Overflow Detection**: Must detect NaN/Inf in gradients
4. **Dynamic Adjustment**: Must adjust scale based on overflow events
5. **Growth Logic**: Must grow scale after successful interval
6. **Backoff Logic**: Must reduce scale after overflow
7. **Statistics**: Must track scaler behavior
8. **Reset**: Must support resetting to initial state

### Non-Functional Requirements
1. **Thread Safety**: Not required (single-threaded training)
2. **Performance**: Operations should be O(n) where n = number of gradients
3. **Memory**: Minimal memory overhead
4. **Error Handling**: Handle null inputs gracefully

## Algorithm Details

### Loss Scaling Flow
1. Before backward pass: `scaled_loss = loss * current_scale`
2. Backward pass computes gradients
3. After backward pass: `unscaled_grads = grads / current_scale`
4. Check for overflow (NaN/Inf) in gradients
5. If overflow: decrease scale, skip optimizer step
6. If no overflow: increase scale after growth interval, proceed

### Scale Adjustment Rules
- **Overflow**: `scale = max(scale * backoff_factor, min_scale)`
- **Success**: After N steps, `scale = min(scale * growth_factor, max_scale)`
- **Bounds**: Scale always stays in [min_scale, max_scale]

## Deliverables

### Source Files
1. `src/MLFramework/Optimizers/MixedPrecision/DynamicLossScaler.cs`

### Unit Tests
- Tests will be covered in spec 009 (DynamicLossScaler unit tests)

## Notes for Coder
- HasOverflow() should be a stub for now (tensor API not ready)
- Tensor operations (Multiply, Divide) should be stubbed for now
- Focus on the scaling logic and state management
- Ensure scale factor is properly bounded by min/max settings
- Consecutive overflow counter is important for auto-fallback logic
- Statistics class should be useful for debugging and monitoring
