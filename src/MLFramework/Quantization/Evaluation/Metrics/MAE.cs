using RitterFramework.Core.Tensor;

namespace MLFramework.Quantization.Evaluation.Metrics;

/// <summary>
/// Computes Mean Absolute Error (MAE) for regression tasks.
/// </summary>
public class MAE : IAccuracyMetric
{
    /// <summary>
    /// Computes Mean Absolute Error for a batch of predictions and labels.
    /// </summary>
    /// <param name="predictions">Model predictions tensor</param>
    /// <param name="labels">Ground truth labels tensor (must match predictions shape)</param>
    /// <returns>MAE value (lower is better)</returns>
    public float Compute(Tensor predictions, Tensor labels)
    {
        if (predictions.Dimensions != labels.Dimensions)
            throw new ArgumentException("Predictions and labels must have the same number of dimensions", nameof(labels));

        if (!predictions.Shape.SequenceEqual(labels.Shape))
            throw new ArgumentException("Predictions and labels must have the same shape", nameof(labels));

        var predictionData = predictions.Data;
        var labelData = labels.Data;
        int size = predictions.Size;

        float sumError = 0;
        for (int i = 0; i < size; i++)
        {
            sumError += Math.Abs(predictionData[i] - labelData[i]);
        }

        return sumError / size;
    }

    /// <summary>
    /// Gets the name of the metric.
    /// </summary>
    public string Name => "Mean Absolute Error (MAE)";

    /// <summary>
    /// Gets whether higher values are better.
    /// </summary>
    public bool HigherIsBetter => false;
}
