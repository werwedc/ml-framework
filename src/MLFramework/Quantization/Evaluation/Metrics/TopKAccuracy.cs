using RitterFramework.Core.Tensor;

namespace MLFramework.Quantization.Evaluation.Metrics;

/// <summary>
/// Computes Top-K accuracy for classification tasks.
/// </summary>
public class TopKAccuracy : IAccuracyMetric
{
    private readonly int _k;

    /// <summary>
    /// Initializes a new instance of the TopKAccuracy metric.
    /// </summary>
    /// <param name="k">Number of top predictions to consider (default: 1)</param>
    public TopKAccuracy(int k = 1)
    {
        if (k <= 0)
            throw new ArgumentOutOfRangeException(nameof(k), "K must be positive");

        _k = k;
    }

    /// <summary>
    /// Computes Top-K accuracy for a batch of predictions and labels.
    /// </summary>
    /// <param name="predictions">Model predictions tensor of shape [batch_size, num_classes]</param>
    /// <param name="labels">Ground truth labels tensor of shape [batch_size] or [batch_size, 1]</param>
    /// <returns>Accuracy value between 0 and 1</returns>
    public float Compute(Tensor predictions, Tensor labels)
    {
        if (predictions.Dimensions != 2)
            throw new ArgumentException("Predictions must be 2D tensor [batch_size, num_classes]", nameof(predictions));

        if (labels.Dimensions != 1 && labels.Dimensions != 2)
            throw new ArgumentException("Labels must be 1D or 2D tensor [batch_size] or [batch_size, 1]", nameof(labels));

        int batchSize = predictions.Shape[0];
        int numClasses = predictions.Shape[1];

        if (labels.Shape[0] != batchSize)
            throw new ArgumentException("Batch size mismatch between predictions and labels");

        var predictionData = predictions.Data;
        var labelData = labels.Data;

        int correct = 0;

        for (int i = 0; i < batchSize; i++)
        {
            // Get the true label
            int trueLabel = (int)labelData[i];

            // Get predictions for this sample
            var samplePredictions = new float[numClasses];
            Array.Copy(predictionData, i * numClasses, samplePredictions, 0, numClasses);

            // Get top-K predictions
            var topKIndices = GetTopKIndices(samplePredictions, _k);

            // Check if true label is in top-K
            if (topKIndices.Contains(trueLabel))
            {
                correct++;
            }
        }

        return (float)correct / batchSize;
    }

    /// <summary>
    /// Gets the indices of the top-K values in an array.
    /// </summary>
    private int[] GetTopKIndices(float[] values, int k)
    {
        var indexedValues = values
            .Select((value, index) => new { Value = value, Index = index })
            .OrderByDescending(x => x.Value)
            .Take(k)
            .Select(x => x.Index)
            .ToArray();

        return indexedValues;
    }

    /// <summary>
    /// Gets the name of the metric.
    /// </summary>
    public string Name => $"Top-{_k} Accuracy";

    /// <summary>
    /// Gets whether higher values are better.
    /// </summary>
    public bool HigherIsBetter => true;
}
