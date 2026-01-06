using System;
using System.Collections.Generic;
using RitterFramework.Core.Tensor;

namespace MLFramework.Optimizers.MixedPrecision;

/// <summary>
/// Wrapper optimizer that enables mixed-precision training for any base optimizer
/// </summary>
public class MixedPrecisionOptimizer : IOptimizer
{
    private readonly IOptimizer _baseOptimizer;
    private readonly MixedPrecisionOptions _options;
    private readonly DynamicLossScaler _lossScaler;
    private readonly GradientConversionLayer _gradientLayer;
    private readonly PerformanceMonitor? _performanceMonitor;

    private Dictionary<string, Tensor>? _masterWeights;
    private Dictionary<string, Tensor>? _trainingWeights;
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
    public IReadOnlyDictionary<string, Tensor>? MasterWeights => _masterWeights;

    /// <summary>
    /// Training weights (in target precision)
    /// </summary>
    public IReadOnlyDictionary<string, Tensor>? TrainingWeights => _trainingWeights;

    /// <summary>
    /// Current training precision
    /// </summary>
    public Precision TargetPrecision => _options.AutoDetectPrecision
        ? HardwareDetector.GetRecommendedPrecision()
        : _options.Precision;

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

    /// <summary>
    /// Current loss scale factor (for monitoring)
    /// </summary>
    public float LossScale => _gradientLayer.LossScale;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a mixed-precision wrapper around an existing optimizer
    /// </summary>
    public MixedPrecisionOptimizer(
        IOptimizer baseOptimizer,
        MixedPrecisionOptions? options = null)
    {
        _baseOptimizer = baseOptimizer ?? throw new ArgumentNullException(nameof(baseOptimizer));
        _options = options ?? MixedPrecisionOptions.ForFP16();
        _options.Validate();

        _lossScaler = new DynamicLossScaler(_options);
        var precisionManager = new PrecisionManager(_options);
        _gradientLayer = new GradientConversionLayer(_options, _lossScaler, precisionManager);

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
    public void SetParameters(Dictionary<string, Tensor> parameters)
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

        // In fallback mode or for now, just pass through
        // TODO: Implement proper precision conversion when tensor infrastructure is ready
        _masterWeights = parameters;
        _trainingWeights = parameters;
        _baseOptimizer.SetParameters(parameters);
    }

    /// <summary>
    /// Performs an optimizer step with the given gradients
    /// </summary>
    public void Step(Dictionary<string, Tensor> gradients)
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
            _performanceMonitor?.StartStep();
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
            // TODO: Implement proper sync when tensor infrastructure is ready
            _trainingWeights = _masterWeights;

            // Reset overflow counter on success - use Reset() instead
            _lossScaler.Reset();
        }
        finally
        {
            if (_options.EnablePerformanceMonitoring)
            {
                _performanceMonitor?.EndStep();
            }
        }
    }

    /// <summary>
    /// Applies a specific gradient to a specific parameter
    /// </summary>
    public void StepParameter(string parameterName, Tensor gradient)
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

        var gradients = new Dictionary<string, Tensor> { { parameterName, gradient } };
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
    public Tensor ScaleLoss(Tensor loss)
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
        if (_masterWeights != null)
        {
            _baseOptimizer.SetParameters(_masterWeights);
        }
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
        var stats = _lossScaler.GetStats();
        if (_options.EnableAutoFallback &&
            stats.ConsecutiveOverflows >= _options.MaxConsecutiveOverflows)
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
    public required GradientProcessingStats GradientStats { get; set; }
    public required LossScalerStats LossScalerStats { get; set; }
    public PerformanceStats? PerformanceStats { get; set; }

    public float SkipRate => StepCount > 0 ? (float)SkippedSteps / StepCount : 0;

    public override string ToString()
    {
        return $"Steps: {StepCount}, " +
               $"Skipped: {SkippedSteps} ({SkipRate:P2}), " +
               $"Fallback: {HasFallback}, " +
               $"Precision: {TargetPrecision}";
    }
}
