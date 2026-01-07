using RitterFramework.Core.Tensor;

namespace MLFramework.Quantization.Evaluation.Metrics;

/// <summary>
/// Interface for accuracy metrics used to evaluate model performance.
/// </summary>
public interface IAccuracyMetric
{
    /// <summary>
    /// Computes the metric value for a batch of predictions and labels.
    /// </summary>
    /// <param name="predictions">Model predictions tensor</param>
    /// <param name="labels">Ground truth labels tensor</param>
    /// <returns>Metric value for this batch</returns>
    float Compute(Tensor predictions, Tensor labels);

    /// <summary>
    /// Gets the name of the metric.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets whether higher values are better (true) or worse (false).
    /// </summary>
    bool HigherIsBetter { get; }
}
