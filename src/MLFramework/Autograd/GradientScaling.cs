using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Provides static methods for scaling and normalizing gradients based on batch sizes.
/// Ensures fair gradient accumulation when dealing with variable batch sizes.
/// </summary>
public static class GradientScaling
{
    /// <summary>
    /// Scales a gradient tensor by the ratio of batch size to reference size.
    /// Used to weight gradients by their contribution to the overall batch.
    /// </summary>
    /// <param name="gradient">The gradient tensor to scale.</param>
    /// <param name="batchSize">The actual batch size.</param>
    /// <param name="referenceSize">The reference batch size for scaling.</param>
    /// <returns>A new scaled gradient tensor.</returns>
    /// <exception cref="ArgumentNullException">Thrown when gradient is null.</exception>
    /// <exception cref="ArgumentException">Thrown when referenceSize is less than 1.</exception>
    public static Tensor ScaleByBatchSize(Tensor gradient, int batchSize, int referenceSize)
    {
        if (gradient == null)
            throw new ArgumentNullException(nameof(gradient));

        if (referenceSize < 1)
            throw new ArgumentException("Reference size must be at least 1", nameof(referenceSize));

        if (batchSize == 0)
            return Tensor.Zeros(gradient.Shape);

        // Calculate scaling factor: (batchSize / referenceSize)
        double scalingFactor = (double)batchSize / referenceSize;

        var scaledData = new float[gradient.Data.Length];
        for (int i = 0; i < gradient.Data.Length; i++)
        {
            scaledData[i] = gradient.Data[i] * (float)scalingFactor;
        }

        return new Tensor(scaledData, gradient.Shape);
    }

    /// <summary>
    /// Averages an accumulated gradient by dividing by the total batch size.
    /// Normalizes gradients to be comparable regardless of accumulation strategy.
    /// </summary>
    /// <param name="accumulated">The accumulated gradient tensor.</param>
    /// <param name="totalBatchSize">The total batch size accumulated.</param>
    /// <param name="referenceSize">The reference batch size for normalization.</param>
    /// <returns>A new normalized gradient tensor.</returns>
    /// <exception cref="ArgumentNullException">Thrown when accumulated is null.</exception>
    /// <exception cref="ArgumentException">Thrown when totalBatchSize or referenceSize is less than 1.</exception>
    public static Tensor AverageAccumulated(Tensor accumulated, int totalBatchSize, int referenceSize)
    {
        if (accumulated == null)
            throw new ArgumentNullException(nameof(accumulated));

        if (totalBatchSize < 1)
            throw new ArgumentException("Total batch size must be at least 1", nameof(totalBatchSize));

        if (referenceSize < 1)
            throw new ArgumentException("Reference size must be at least 1", nameof(referenceSize));

        // Calculate normalization factor: (totalBatchSize / referenceSize)
        double normalizationFactor = (double)totalBatchSize / referenceSize;

        var normalizedData = new float[accumulated.Data.Length];
        for (int i = 0; i < accumulated.Data.Length; i++)
        {
            normalizedData[i] = accumulated.Data[i] / (float)normalizationFactor;
        }

        return new Tensor(normalizedData, accumulated.Shape);
    }

    /// <summary>
    /// Normalizes a batch gradient by dividing by its batch size.
    /// Ensures that gradients from different batch sizes are on the same scale.
    /// </summary>
    /// <param name="gradient">The gradient tensor to normalize.</param>
    /// <param name="batchSize">The batch size of the gradient.</param>
    /// <returns>A new normalized gradient tensor.</returns>
    /// <exception cref="ArgumentNullException">Thrown when gradient is null.</exception>
    /// <exception cref="ArgumentException">Thrown when batchSize is less than 1.</exception>
    public static Tensor NormalizeBatchGradient(Tensor gradient, int batchSize)
    {
        if (gradient == null)
            throw new ArgumentNullException(nameof(gradient));

        if (batchSize < 1)
            throw new ArgumentException("Batch size must be at least 1", nameof(batchSize));

        // Normalize by dividing by batch size
        var normalizedData = new float[gradient.Data.Length];
        for (int i = 0; i < gradient.Data.Length; i++)
        {
            normalizedData[i] = gradient.Data[i] / batchSize;
        }

        return new Tensor(normalizedData, gradient.Shape);
    }

    /// <summary>
    /// Combines multiple gradients using weighted averaging based on batch sizes.
    /// </summary>
    /// <param name="gradients">Array of gradient tensors to combine.</param>
    /// <param name="batchSizes">Array of batch sizes corresponding to each gradient.</param>
    /// <returns>A new combined gradient tensor.</returns>
    /// <exception cref="ArgumentNullException">Thrown when gradients or batchSizes is null.</exception>
    /// <exception cref="ArgumentException">Thrown when arrays have different lengths or contain invalid values.</exception>
    public static Tensor WeightedAverage(Tensor[] gradients, int[] batchSizes)
    {
        if (gradients == null)
            throw new ArgumentNullException(nameof(gradients));

        if (batchSizes == null)
            throw new ArgumentNullException(nameof(batchSizes));

        if (gradients.Length != batchSizes.Length)
            throw new ArgumentException("Gradients and batch sizes must have the same length");

        if (gradients.Length == 0)
            throw new ArgumentException("Gradients array cannot be empty");

        // Initialize with the first gradient
        var combined = Tensor.Zeros(gradients[0].Shape);
        int totalBatchSize = 0;

        // Accumulate weighted gradients
        for (int i = 0; i < gradients.Length; i++)
        {
            if (gradients[i] == null)
                throw new ArgumentException($"Gradient at index {i} is null");

            if (batchSizes[i] < 1)
                throw new ArgumentException($"Batch size at index {i} must be at least 1");

            var normalized = NormalizeBatchGradient(gradients[i], batchSizes[i]);
            var weighted = ScaleByBatchSize(normalized, batchSizes[i], 1);

            for (int j = 0; j < combined.Data.Length; j++)
            {
                combined.Data[j] += weighted.Data[j];
            }

            totalBatchSize += batchSizes[i];
        }

        // Return normalized result
        return AverageAccumulated(combined, totalBatchSize, 1);
    }
}
