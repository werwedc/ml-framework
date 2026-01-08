namespace MachineLearning.Visualization.Scalars.Smoothing;

/// <summary>
/// Provides moving average smoothing for scalar series
/// </summary>
public class MovingAverageSmoother
{
    /// <summary>
    /// Applies moving average smoothing to a series of values
    /// </summary>
    /// <param name="values">Values to smooth</param>
    /// <param name="windowSize">Size of the moving average window</param>
    /// <returns>Smoothed values</returns>
    public static float[] Smooth(float[] values, int windowSize)
    {
        if (values == null) throw new ArgumentNullException(nameof(values));
        if (windowSize <= 0) throw new ArgumentException("Window size must be positive", nameof(windowSize));

        if (values.Length == 0) return Array.Empty<float>();

        float[] smoothed = new float[values.Length];
        int halfWindow = windowSize / 2;

        for (int i = 0; i < values.Length; i++)
        {
            int start = Math.Max(0, i - halfWindow);
            int end = Math.Min(values.Length - 1, i + halfWindow);
            int count = end - start + 1;

            float sum = 0;
            for (int j = start; j <= end; j++)
            {
                sum += values[j];
            }

            smoothed[i] = sum / count;
        }

        return smoothed;
    }

    /// <summary>
    /// Applies exponential moving average smoothing to a series of values
    /// </summary>
    /// <param name="values">Values to smooth</param>
    /// <param name="alpha">Smoothing factor (0 < alpha <= 1)</param>
    /// <returns>Smoothed values</returns>
    public static float[] ExponentialSmooth(float[] values, float alpha)
    {
        if (values == null) throw new ArgumentNullException(nameof(values));
        if (alpha <= 0 || alpha > 1) throw new ArgumentException("Alpha must be in (0, 1]", nameof(alpha));

        if (values.Length == 0) return Array.Empty<float>();

        float[] smoothed = new float[values.Length];
        smoothed[0] = values[0];

        for (int i = 1; i < values.Length; i++)
        {
            smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1];
        }

        return smoothed;
    }

    /// <summary>
    /// Applies weighted moving average smoothing to a series of values
    /// </summary>
    /// <param name="values">Values to smooth</param>
    /// <param name="weights">Weights for the window (must be odd length)</param>
    /// <returns>Smoothed values</returns>
    public static float[] WeightedSmooth(float[] values, float[] weights)
    {
        if (values == null) throw new ArgumentNullException(nameof(values));
        if (weights == null) throw new ArgumentNullException(nameof(weights));
        if (weights.Length % 2 == 0) throw new ArgumentException("Weights must have odd length", nameof(weights));

        if (values.Length == 0) return Array.Empty<float>();

        float[] smoothed = new float[values.Length];
        int halfWindow = weights.Length / 2;
        float weightSum = weights.Sum();

        for (int i = 0; i < values.Length; i++)
        {
            float sum = 0;
            int actualWeights = 0;

            for (int j = -halfWindow; j <= halfWindow; j++)
            {
                int idx = i + j;
                if (idx >= 0 && idx < values.Length)
                {
                    sum += values[idx] * weights[j + halfWindow];
                    actualWeights++;
                }
            }

            // Normalize by actual weights used (handles edges)
            int weightStart = Math.Max(0, halfWindow - i);
            int weightEnd = Math.Min(weights.Length - 1, halfWindow + (values.Length - 1 - i));
            float actualWeightSum = weights.Skip(weightStart).Take(weightEnd - weightStart + 1).Sum();

            smoothed[i] = actualWeightSum > 0 ? sum / actualWeightSum : 0;
        }

        return smoothed;
    }
}
