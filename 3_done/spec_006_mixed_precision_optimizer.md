# Spec: MixedPrecisionOptimizer Base Class

## Overview
Implement the main MixedPrecisionOptimizer that wraps existing optimizers and orchestrates mixed-precision training.

## Dependencies
- Spec 001: Precision enum and detection utilities
- Spec 002: MixedPrecisionOptions
- Spec 003: DynamicLossScaler
- Spec 004: PrecisionManager
- Spec 005: GradientConversionLayer

## Implementation Details

### MixedPrecisionOptimizer Class
Create the class in `src/MLFramework/Optimizers/MixedPrecision/MixedPrecisionOptimizer.cs`:

```csharp
using System;
using System.Collections.Generic;

namespace MLFramework.Optimizers.MixedPrecision;

/// <summary>
/// Wrapper optimizer that enables mixed-precision training for any base optimizer
/// </summary>
public class MixedPrecisionOptimizer : IOptimizer
{
    private readonly IOptimizer _baseOptimizer;
    private readonly MixedPrecisionOptions _options;
    private readonly DynamicLossScaler _lossScaler;
    private readonly PrecisionManager _precisionManager;
    private readonly GradientConversionLayer _gradientLayer;
    private readonly PerformanceMonitor _performanceMonitor;

    private Dictionary<string, ITensor> _masterWeights;
    private Dictionary<string, ITensor> _trainingWeights;
    private bool _fallbackToFP32;
    private int _stepCount;

    #region Properties

    /// <summary>
    /// The underlying optimizer being wrapped
    /// </summary>
    public IOptimizer BaseOptimizer => _baseOptimizer;

    /// <summary>
    /// Mixed precision configuration
    /// </summary>
    public MixedPrecisionOptions Options => _options;

    /// <summary>
    /// Master FP32 weights
    /// </summary>
    public IReadOnlyDictionary<string, ITensor> MasterWeights => _masterWeights;

    /// <summary>
    /// Training weights (in target precision)
    /// </summary>
    public IReadOnlyDictionary<string, ITensor> TrainingWeights => _trainingWeights;

    /// <summary>
    /// Current training precision
    /// </summary>
    public Precision TargetPrecision => _precisionManager.TargetPrecision;

    /// <summary>
    /// Whether optimizer has fallen back to FP32
    /// </summary>
    public bool HasFallback => _fallbackToFP32;

    /// <summary>
    /// Total number of optimizer steps performed
    /// </summary>
    public int StepCount => _stepCount;

    /// <summary>
    /// Number of skipped steps (due to overflow)
    /// </summary>
    public int SkippedSteps { get; private set; }

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a mixed-precision wrapper around an existing optimizer
    /// </summary>
    public MixedPrecisionOptimizer(
        IOptimizer baseOptimizer,
        MixedPrecisionOptions options = null)
    {
        _baseOptimizer = baseOptimizer ?? throw new ArgumentNullException(nameof(baseOptimizer));
        _options = options ?? MixedPrecisionOptions.ForFP16();
        _options.Validate();

        _lossScaler = new DynamicLossScaler(_options);
        _precisionManager = new PrecisionManager(_options);
        _gradientLayer = new GradientConversionLayer(_options, _lossScaler, _precisionManager);

        if (_options.EnablePerformanceMonitoring)
        {
            _performanceMonitor = new PerformanceMonitor(_options);
        }

        _fallbackToFP32 = false;
        _stepCount = 0;
        SkippedSteps = 0;
    }

    #endregion

    #region IOptimizer Implementation

    /// <summary>
    /// Sets the parameters to optimize
    /// </summary>
    public void SetParameters(Dictionary<string, ITensor> parameters)
    {
        if (parameters == null)
            throw new ArgumentNullException(nameof(parameters));

        if (_fallbackToFP32)
        {
            // Fallback: just pass through to base optimizer
            _baseOptimizer.SetParameters(parameters);
            _masterWeights = parameters;
            _trainingWeights = parameters;
            return;
        }

        // Create master weights (FP32)
        _masterWeights = _precisionManager.CreateMasterWeights(parameters);

        // Create training weights (target precision)
        _trainingWeights = _precisionManager.ConvertWeights(parameters);

        // Set training weights in base optimizer
        _baseOptimizer.SetParameters(_trainingWeights);
    }

    /// <summary>
    /// Performs an optimizer step with the given gradients
    /// </summary>
    public void Step(Dictionary<string, ITensor> gradients)
    {
        if (gradients == null)
            throw new ArgumentNullException(nameof(gradients));

        if (_fallbackToFP32)
        {
            // Fallback: just pass through to base optimizer
            _baseOptimizer.Step(gradients);
            _stepCount++;
            return;
        }

        _stepCount++;

        if (_options.EnablePerformanceMonitoring)
        {
            _performanceMonitor.StartStep();
        }

        try
        {
            // Process gradients through conversion layer
            var (processedGrads, shouldSkip) = _gradientLayer.ProcessGradients(gradients);

            if (shouldSkip)
            {
                // Check for consecutive overflow
                HandleOverflow();
                return;
            }

            // Update master weights with processed gradients
            _baseOptimizer.Step(processedGrads);

            // Sync training weights from master weights
            _trainingWeights = _precisionManager.SyncTrainingWeights(_masterWeights);

            // Reset overflow counter on success
            _lossScaler.ConsecutiveOverflows = 0;
        }
        finally
        {
            if (_options.EnablePerformanceMonitoring)
            {
                _performanceMonitor.EndStep();
            }
        }
    }

    /// <summary>
    /// Applies a specific gradient to a specific parameter
    /// </summary>
    public void StepParameter(string parameterName, ITensor gradient)
    {
        if (string.IsNullOrEmpty(parameterName))
            throw new ArgumentException("Parameter name cannot be null or empty", nameof(parameterName));

        if (gradient == null)
            throw new ArgumentNullException(nameof(gradient));

        if (_fallbackToFP32)
        {
            _baseOptimizer.StepParameter(parameterName, gradient);
            return;
        }

        var gradients = new Dictionary<string, ITensor> { { parameterName, gradient } };
        Step(gradients);
    }

    /// <summary>
    /// Zeroes out gradients
    /// </summary>
    public void ZeroGrad()
    {
        _baseOptimizer.ZeroGrad();
    }

    /// <summary>
    /// Gets the current learning rate
    /// </summary>
    public float LearningRate => _baseOptimizer.LearningRate;

    /// <summary>
    /// Sets the learning rate
    /// </summary>
    public void SetLearningRate(float lr)
    {
        _baseOptimizer.SetLearningRate(lr);
    }

    #endregion

    #region Mixed Precision Specific Methods

    /// <summary>
    /// Scales loss before backward pass
    /// </summary>
    public ITensor ScaleLoss(ITensor loss)
    {
        if (_fallbackToFP32)
            return loss;

        return _gradientLayer.ScaleLoss(loss);
    }

    /// <summary>
    /// Manually triggers fallback to FP32
    /// </summary>
    public void FallbackToFP32()
    {
        if (_fallbackToFP32)
            return;

        _fallbackToFP32 = true;

        if (_options.LogFallbackEvents)
        {
            Console.WriteLine($"[MixedPrecisionOptimizer] Falling back to FP32 at step {_stepCount}");
        }

        // Switch master weights to be the training weights
        _masterWeights = _trainingWeights;
        _baseOptimizer.SetParameters(_masterWeights);
    }

    /// <summary>
    /// Gets comprehensive statistics about optimizer behavior
    /// </summary>
    public MixedPrecisionOptimizerStats GetStats()
    {
        return new MixedPrecisionOptimizerStats
        {
            StepCount = _stepCount,
            SkippedSteps = SkippedSteps,
            HasFallback = _fallbackToFP32,
            TargetPrecision = TargetPrecision,
            GradientStats = _gradientLayer.GetStats(),
            LossScalerStats = _lossScaler.GetStats(),
            PerformanceStats = _performanceMonitor?.GetStats()
        };
    }

    /// <summary>
    /// Resets optimizer to initial state
    /// </summary>
    public void Reset()
    {
        _lossScaler.Reset();
        _gradientLayer.Reset();
        _performanceMonitor?.Reset();
        _stepCount = 0;
        SkippedSteps = 0;
    }

    #endregion

    #region Private Methods

    private void HandleOverflow()
    {
        SkippedSteps++;

        if (_options.LogFallbackEvents)
        {
            Console.WriteLine($"[MixedPrecisionOptimizer] Skipped step {_stepCount} due to overflow");
        }

        // Check if we should fallback
        if (_options.EnableAutoFallback &&
            _lossScaler.ConsecutiveOverflows >= _options.MaxConsecutiveOverflows)
        {
            FallbackToFP32();
        }
    }

    #endregion
}

/// <summary>
/// Comprehensive statistics for mixed-precision optimizer
/// </summary>
public class MixedPrecisionOptimizerStats
{
    public int StepCount { get; set; }
    public int SkippedSteps { get; set; }
    public bool HasFallback { get; set; }
    public Precision TargetPrecision { get; set; }
    public GradientProcessingStats GradientStats { get; set; }
    public LossScalerStats LossScalerStats { get; set; }
    public PerformanceStats PerformanceStats { get; set; }

    public float SkipRate => StepCount > 0 ? (float)SkippedSteps / StepCount : 0;

    public override string ToString()
    {
        return $"Steps: {StepCount}, " +
               $"Skipped: {SkippedSteps} ({SkipRate:P2}), " +
               $"Fallback: {HasFallback}, " +
               $"Precision: {TargetPrecision}";
    }
}
```

## Requirements

### Functional Requirements
1. **Optimizer Wrapping**: Wrap any IOptimizer implementation
2. **Parameter Management**: Maintain master and training weights
3. **Gradient Processing**: Orchestrate full mixed-precision pipeline
4. **Loss Scaling**: Scale loss before backward pass
5. **Overflow Handling**: Skip steps and track consecutive overflows
6. **Auto Fallback**: Automatically switch to FP32 if unstable
7. **Statistics**: Comprehensive tracking of optimizer behavior
8. **IOptimizer Interface**: Implement all required methods

### Non-Functional Requirements
1. **Transparency**: API should be identical to base optimizer
2. **Performance**: Minimal overhead for precision management
3. **Thread Safety**: Not required (single-threaded training)
4. **Memory**: Master weights double memory usage (expected)

## Training Flow

### Standard Mixed Precision
1. Scale loss → backward pass
2. Receive gradients in target precision
3. Process: Check overflow → Unscale → Clip → Convert to FP32
4. Update master weights (FP32)
5. Sync training weights from master weights

### Fallback Mode
1. Pass through all operations to base optimizer
2. No loss scaling, no precision conversion
3. Standard FP32 training

## Deliverables

### Source Files
1. `src/MLFramework/Optimizers/MixedPrecision/MixedPrecisionOptimizer.cs`

### Unit Tests
- Tests will be covered in spec 011 (MixedPrecisionOptimizer unit tests)

## Notes for Coder
- IOptimizer is an interface that needs to exist (base type for all optimizers)
- _performanceMonitor will be implemented in spec 008
- Assume the base optimizer implements IOptimizer interface
- Focus on the orchestration logic between components
- Ensure proper handling of fallback mode
- Statistics should be comprehensive for debugging
- The wrapper pattern should be transparent to the user
