namespace MLFramework.Schedulers;

/// <summary>
/// Marker interface for schedulers that react to training metrics.
/// </summary>
public interface IMetricBasedScheduler : ILearningRateScheduler
{
    /// <summary>
    /// Updates the scheduler with a new metric value.
    /// </summary>
    /// <param name="metricName">Name of the metric (e.g., "val_loss").</param>
    /// <param name="value">Current metric value.</param>
    void UpdateMetric(string metricName, float value);
}
