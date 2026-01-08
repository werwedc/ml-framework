using System;

namespace MLFramework.Visualization.Hooks;

/// <summary>
/// Interface for training hooks that can be attached to a training loop.
/// Hooks receive callbacks at various points during training and can perform
/// actions such as logging, profiling, checkpointing, etc.
/// </summary>
public interface ITrainingHook
{
    /// <summary>
    /// Called when a training phase starts.
    /// </summary>
    /// <param name="phase">The phase that is starting</param>
    /// <param name="context">Current training context</param>
    void OnPhaseStart(TrainingPhase phase, TrainingContext context);

    /// <summary>
    /// Called when a training phase ends.
    /// </summary>
    /// <param name="phase">The phase that ended</param>
    /// <param name="context">Current training context</param>
    void OnPhaseEnd(TrainingPhase phase, TrainingContext context);

    /// <summary>
    /// Called when a metric is updated.
    /// </summary>
    /// <param name="metricName">Name of the metric</param>
    /// <param name="value">New value of the metric</param>
    /// <param name="context">Current training context</param>
    void OnMetricUpdate(string metricName, float value, TrainingContext context);

    /// <summary>
    /// Called when an exception occurs during training.
    /// </summary>
    /// <param name="exception">The exception that occurred</param>
    /// <param name="context">Current training context</param>
    void OnException(Exception exception, TrainingContext context);
}
