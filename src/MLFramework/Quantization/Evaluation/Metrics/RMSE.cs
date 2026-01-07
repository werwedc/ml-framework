using RitterFramework.Core.Tensor;

namespace MLFramework.Quantization.Evaluation.Metrics;

/// <summary>
/// Computes Root Mean Square Error (RMSE) for regression tasks.
/// </summary>
public class RMSE : IAccuracyMetric
{
    /// <summary>
    /// Computes Root Mean Square Error for a batch of predictions and labels.
    /// </summary>
    /// <param name="predictions">Model predictions tensor</param>
    /// <param name="labels">Ground truth labels tensor (must match predictions shape)</param>
    /// <returns>RMSE value (lower is better)</returns>
    public float Compute(Tensor predictions, Tensor labels)
    {
        if (predictions.Dimensions != labels.Dimensions)
            throw new ArgumentException("Predictions and labels must have the same number of dimensions", nameof(labels));

        if (!predictions.Shape.SequenceEqual(labels.Shape))
            throw new ArgumentException("Predictions and labels must have the same shape", nameof(labels));

        var predictionData = predictions.Data;
        var labelData = labels.Data;
        int size = predictions.Size;

        float sumSquaredError = 0;
        for (int i = 0; i < size; i++)
        {
            float error = predictionData[i] - labelData[i];
            sumSquaredError += error * error;
        }

        float mse = sumSquaredError / size;
        return (float)Math.Sqrt(mse);
    }

    /// <summary>
    /// Gets the name of the metric.
    /// </summary>
    public string Name => "Root Mean Square Error (RMSE)";

    /// <summary>
    /// Gets whether higher values are better.
    /// </summary>
    public bool HigherIsBetter => false;
}
