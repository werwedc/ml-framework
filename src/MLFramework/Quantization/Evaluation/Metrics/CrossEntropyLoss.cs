using RitterFramework.Core.Tensor;

namespace MLFramework.Quantization.Evaluation.Metrics;

/// <summary>
/// Computes Cross-Entropy Loss for classification tasks.
/// </summary>
public class CrossEntropyLoss : IAccuracyMetric
{
    /// <summary>
    /// Computes Cross-Entropy Loss for a batch of predictions and labels.
    /// </summary>
    /// <param name="predictions">Model predictions tensor of shape [batch_size, num_classes] (logits or softmax probabilities)</param>
    /// <param name="labels">Ground truth labels tensor of shape [batch_size] (class indices)</param>
    /// <returns>Cross-Entropy Loss value (lower is better)</returns>
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

        float totalLoss = 0;

        for (int i = 0; i < batchSize; i++)
        {
            // Get the true label
            int trueLabel = (int)labelData[i];

            // Get predictions for this sample
            var samplePredictions = new float[numClasses];
            Array.Copy(predictionData, i * numClasses, samplePredictions, 0, numClasses);

            // Apply softmax if needed (check if sum is close to 1)
            float sum = samplePredictions.Sum();
            float[] probabilities = sum > 1.5f ? ApplySoftmax(samplePredictions) : samplePredictions;

            // Get probability of true class with numerical stability
            float prob = probabilities[trueLabel];
            float logProb = (float)Math.Log(Math.Max(prob, 1e-10f)); // Avoid log(0)

            totalLoss -= logProb;
        }

        return totalLoss / batchSize;
    }

    /// <summary>
    /// Applies softmax function to convert logits to probabilities.
    /// </summary>
    private float[] ApplySoftmax(float[] logits)
    {
        // Find max for numerical stability
        float maxLogit = logits.Max();

        // Compute softmax
        float expSum = 0;
        var expValues = new float[logits.Length];

        for (int i = 0; i < logits.Length; i++)
        {
            expValues[i] = (float)Math.Exp(logits[i] - maxLogit);
            expSum += expValues[i];
        }

        // Normalize
        for (int i = 0; i < logits.Length; i++)
        {
            expValues[i] /= expSum;
        }

        return expValues;
    }

    /// <summary>
    /// Gets the name of the metric.
    /// </summary>
    public string Name => "Cross-Entropy Loss";

    /// <summary>
    /// Gets whether higher values are better.
    /// </summary>
    public bool HigherIsBetter => false;
}
