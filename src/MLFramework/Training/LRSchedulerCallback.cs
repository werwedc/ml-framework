using System;
using System.Collections.Generic;
using MLFramework.Schedulers;

namespace MLFramework.Training;

/// <summary>
/// Callback that automatically steps learning rate schedulers during training.
/// Handles step-based, epoch-based, and metric-based schedulers.
/// </summary>
public class LRSchedulerCallback : Callback
{
    private readonly ILearningRateScheduler _scheduler;
    private readonly bool _stepOnBatch;
    private readonly bool _stepOnEpoch;
    private readonly string? _metricName;

    /// <summary>
    /// Creates a callback for the given scheduler.
    /// Automatically detects scheduler type and steps appropriately.
    /// </summary>
    /// <param name="scheduler">The scheduler to step.</param>
    public LRSchedulerCallback(ILearningRateScheduler scheduler)
    {
        _scheduler = scheduler ?? throw new ArgumentNullException(nameof(scheduler));

        // Auto-detect stepping behavior
        _stepOnBatch = scheduler is IStepScheduler;
        _stepOnEpoch = scheduler is IEpochScheduler;
        _metricName = null;

        if (scheduler is IMetricBasedScheduler)
        {
            // For metric-based schedulers, metric name should be set manually
            // or use the alternative constructor
        }
    }

    /// <summary>
    /// Creates a callback with explicit stepping behavior.
    /// </summary>
    /// <param name="scheduler">The scheduler to step.</param>
    /// <param name="stepOnBatch">Whether to step on each batch.</param>
    /// <param name="stepOnEpoch">Whether to step on each epoch.</param>
    /// <param name="metricName">Metric name for metric-based schedulers.</param>
    public LRSchedulerCallback(
        ILearningRateScheduler scheduler,
        bool stepOnBatch = false,
        bool stepOnEpoch = false,
        string? metricName = null)
    {
        _scheduler = scheduler ?? throw new ArgumentNullException(nameof(scheduler));
        _stepOnBatch = stepOnBatch;
        _stepOnEpoch = stepOnEpoch;
        _metricName = metricName;
    }

    /// <summary>
    /// Called at the end of each batch. Steps step-based schedulers if configured.
    /// </summary>
    /// <param name="batch">The current batch index (0-based).</param>
    /// <param name="metrics">Dictionary of metrics collected during this batch.</param>
    public override void OnBatchEnd(int batch, Dictionary<string, float> metrics)
    {
        if (_stepOnBatch && _scheduler is IStepScheduler stepScheduler)
        {
            stepScheduler.Step();
        }
    }

    /// <summary>
    /// Called at the end of each epoch. Steps epoch-based schedulers and updates metric-based schedulers.
    /// </summary>
    /// <param name="epoch">The current epoch index (0-based).</param>
    /// <param name="metrics">Dictionary of metrics collected during this epoch.</param>
    public override void OnEpochEnd(int epoch, Dictionary<string, float> metrics)
    {
        if (_stepOnEpoch && _scheduler is IEpochScheduler epochScheduler)
        {
            epochScheduler.StepEpoch();
        }

        // Handle metric-based schedulers
        if (_scheduler is IMetricBasedScheduler metricScheduler && !string.IsNullOrEmpty(_metricName))
        {
            if (metrics.TryGetValue(_metricName, out float value))
            {
                metricScheduler.UpdateMetric(_metricName, value);
            }
        }
    }

    /// <summary>
    /// Called at the end of validation. Updates metric-based schedulers with validation metrics.
    /// </summary>
    /// <param name="metrics">Dictionary of validation metrics.</param>
    public override void OnValidationEnd(Dictionary<string, float> metrics)
    {
        // Handle metric-based schedulers with validation metrics
        if (_scheduler is IMetricBasedScheduler metricScheduler && !string.IsNullOrEmpty(_metricName))
        {
            if (metrics.TryGetValue(_metricName, out float value))
            {
                metricScheduler.UpdateMetric(_metricName, value);
            }
        }
    }

    /// <summary>
    /// Gets the scheduler associated with this callback.
    /// </summary>
    public ILearningRateScheduler Scheduler => _scheduler;
}
