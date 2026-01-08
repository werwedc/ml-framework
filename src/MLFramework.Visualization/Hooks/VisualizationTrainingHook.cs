using System;
using System.Collections.Generic;

namespace MLFramework.Visualization.Hooks;

/// <summary>
/// Training hook that integrates visualization into the training loop.
/// Automatically logs metrics and profiles training phases without manual intervention.
/// </summary>
public class VisualizationTrainingHook : ITrainingHook
{
    private readonly IVisualizer _visualizer;
    private readonly Dictionary<string, long> _phaseStartTimes = new();
    private int _batchesSinceLastLog = 0;

    /// <summary>
    /// Creates a new visualization training hook
    /// </summary>
    /// <param name="visualizer">The visualizer to use for logging</param>
    public VisualizationTrainingHook(IVisualizer visualizer)
    {
        _visualizer = visualizer ?? throw new ArgumentNullException(nameof(visualizer));
    }

    /// <summary>
    /// Whether to log loss values
    /// </summary>
    public bool LogLoss { get; set; } = true;

    /// <summary>
    /// Whether to log custom metrics
    /// </summary>
    public bool LogMetrics { get; set; } = true;

    /// <summary>
    /// Whether to log learning rate
    /// </summary>
    public bool LogLearningRate { get; set; } = true;

    /// <summary>
    /// Whether to profile forward pass
    /// </summary>
    public bool ProfileForwardPass { get; set; } = true;

    /// <summary>
    /// Whether to profile backward pass
    /// </summary>
    public bool ProfileBackwardPass { get; set; } = true;

    /// <summary>
    /// Whether to profile optimizer step
    /// </summary>
    public bool ProfileOptimizerStep { get; set; } = true;

    /// <summary>
    /// Prefix to add to all logged metric names (e.g., "train/")
    /// </summary>
    public string LogPrefix { get; set; } = "train/";

    /// <summary>
    /// How often to log metrics (in batches). 1 means log every batch.
    /// </summary>
    public int LogFrequencyBatches { get; set; } = 1;

    /// <summary>
    /// Called when a training phase starts
    /// </summary>
    public void OnPhaseStart(TrainingPhase phase, TrainingContext context)
    {
        // Start profiling for specific phases
        switch (phase)
        {
            case TrainingPhase.ForwardPassStart:
                if (ProfileForwardPass)
                {
                    _visualizer.StartProfile($"{LogPrefix}forward_pass");
                }
                break;

            case TrainingPhase.BackwardPassStart:
                if (ProfileBackwardPass)
                {
                    _visualizer.StartProfile($"{LogPrefix}backward_pass");
                }
                break;

            case TrainingPhase.OptimizerStep:
                if (ProfileOptimizerStep)
                {
                    _visualizer.StartProfile($"{LogPrefix}optimizer_step");
                }
                break;

            case TrainingPhase.EpochStart:
                // Reset batch counter at start of epoch
                _batchesSinceLastLog = 0;
                break;
        }
    }

    /// <summary>
    /// Called when a training phase ends
    /// </summary>
    public void OnPhaseEnd(TrainingPhase phase, TrainingContext context)
    {
        // End profiling for specific phases
        switch (phase)
        {
            case TrainingPhase.ForwardPassEnd:
                if (ProfileForwardPass)
                {
                    _visualizer.EndProfile($"{LogPrefix}forward_pass");
                }
                break;

            case TrainingPhase.BackwardPassEnd:
                if (ProfileBackwardPass)
                {
                    _visualizer.EndProfile($"{LogPrefix}backward_pass");
                }
                break;

            case TrainingPhase.OptimizerStep:
                if (ProfileOptimizerStep)
                {
                    _visualizer.EndProfile($"{LogPrefix}optimizer_step");
                }
                break;

            case TrainingPhase.BatchEnd:
                _batchesSinceLastLog++;

                // Log at configured frequency
                if (_batchesSinceLastLog >= LogFrequencyBatches)
                {
                    LogMetricsBatch(context);
                    _batchesSinceLastLog = 0;
                }
                break;

            case TrainingPhase.EpochEnd:
                // Always log at epoch end
                LogMetricsBatch(context);
                break;
        }
    }

    /// <summary>
    /// Called when a metric is updated
    /// </summary>
    public void OnMetricUpdate(string metricName, float value, TrainingContext context)
    {
        if (!LogMetrics)
            return;

        var fullName = $"{LogPrefix}{metricName}";
        _visualizer.LogScalar(fullName, value, context.CurrentStep);
    }

    /// <summary>
    /// Called when an exception occurs during training
    /// </summary>
    public void OnException(Exception exception, TrainingContext context)
    {
        // Log the exception information
        Console.Error.WriteLine($"[VisualizationHook] Exception at step {context.CurrentStep}, epoch {context.CurrentEpoch}, batch {context.CurrentBatch}:");
        Console.Error.WriteLine($"  {exception.Message}");

        // Flush any pending logs
        _visualizer.Flush();
    }

    /// <summary>
    /// Logs a batch of metrics
    /// </summary>
    private void LogMetricsBatch(TrainingContext context)
    {
        // Log loss
        if (LogLoss)
        {
            _visualizer.LogScalar($"{LogPrefix}loss", context.Loss, context.CurrentStep);
        }

        // Log learning rate
        if (LogLearningRate)
        {
            _visualizer.LogScalar($"{LogPrefix}learning_rate", context.LearningRate, context.CurrentStep);
        }

        // Log all custom metrics
        if (LogMetrics)
        {
            foreach (var metric in context.Metrics)
            {
                var fullName = $"{LogPrefix}{metric.Key}";
                _visualizer.LogScalar(fullName, metric.Value, context.CurrentStep);
            }
        }

        // Flush logs
        _visualizer.Flush();
    }
}
