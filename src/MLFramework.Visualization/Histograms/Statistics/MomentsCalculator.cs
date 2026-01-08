namespace MLFramework.Visualization.Histograms.Statistics;

/// <summary>
/// Calculator for statistical moments (skewness, kurtosis) of data distributions
/// </summary>
public static class MomentsCalculator
{
    /// <summary>
    /// Calculates the skewness (third standardized moment) of the data
    /// Skewness measures asymmetry of the probability distribution
    /// </summary>
    /// <param name="data">Input data array</param>
    /// <param name="mean">Pre-calculated mean of the data</param>
    /// <param name="std">Pre-calculated standard deviation of the data</param>
    /// <returns>The skewness value</returns>
    public static float CalculateSkewness(float[] data, float mean, float std)
    {
        if (data == null || data.Length == 0)
        {
            throw new ArgumentException("Data array cannot be null or empty", nameof(data));
        }

        if (std == 0)
        {
            return 0; // No skewness if all values are identical
        }

        if (data.Length < 3)
        {
            return 0; // Skewness is undefined for n < 3
        }

        int n = data.Length;
        float m3 = 0f; // Third central moment

        for (int i = 0; i < n; i++)
        {
            float deviation = data[i] - mean;
            m3 += deviation * deviation * deviation;
        }

        m3 /= n;

        // Use the adjusted Fisher-Pearson coefficient of skewness
        // G1 = [sqrt(n(n-1)) / (n-2)] * (m3 / std^3)
        float skewness = (float)(Math.Sqrt(n * (n - 1.0)) / (n - 2.0)) * (m3 / (float)Math.Pow(std, 3));

        return skewness;
    }

    /// <summary>
    /// Calculates the skewness without pre-calculated mean and std
    /// </summary>
    /// <param name="data">Input data array</param>
    /// <returns>The skewness value</returns>
    public static float CalculateSkewness(float[] data)
    {
        (float mean, float std) = CalculateMeanAndStd(data);
        return CalculateSkewness(data, mean, std);
    }

    /// <summary>
    /// Calculates the kurtosis (fourth standardized moment) of the data
    /// Kurtosis measures the "tailedness" of the probability distribution
    /// </summary>
    /// <param name="data">Input data array</param>
    /// <param name="mean">Pre-calculated mean of the data</param>
    /// <param name="std">Pre-calculated standard deviation of the data</param>
    /// <returns>The kurtosis value (excess kurtosis)</returns>
    public static float CalculateKurtosis(float[] data, float mean, float std)
    {
        if (data == null || data.Length == 0)
        {
            throw new ArgumentException("Data array cannot be null or empty", nameof(data));
        }

        if (std == 0)
        {
            return 0; // No kurtosis if all values are identical
        }

        if (data.Length < 4)
        {
            return 0; // Kurtosis is undefined for n < 4
        }

        int n = data.Length;
        float m4 = 0f; // Fourth central moment

        for (int i = 0; i < n; i++)
        {
            float deviation = data[i] - mean;
            float deviationSquared = deviation * deviation;
            m4 += deviationSquared * deviationSquared;
        }

        m4 /= n;

        // Calculate excess kurtosis (subtract 3 to compare to normal distribution)
        // G2 = [(n-1) / ((n-2)(n-3))] * [(n+1)g2 + 6]
        // where g2 = m4 / std^4 - 3
        float g2 = m4 / (float)Math.Pow(std, 4) - 3.0f;
        float correction = (float)((n - 1.0) / ((n - 2.0) * (n - 3.0)));
        float kurtosis = correction * ((n + 1) * g2 + 6.0f);

        return kurtosis;
    }

    /// <summary>
    /// Calculates the kurtosis without pre-calculated mean and std
    /// </summary>
    /// <param name="data">Input data array</param>
    /// <returns>The kurtosis value (excess kurtosis)</returns>
    public static float CalculateKurtosis(float[] data)
    {
        (float mean, float std) = CalculateMeanAndStd(data);
        return CalculateKurtosis(data, mean, std);
    }

    /// <summary>
    /// Calculates both skewness and kurtosis in a single pass
    /// </summary>
    /// <param name="data">Input data array</param>
    /// <returns>Tuple containing (skewness, kurtosis)</returns>
    public static (float Skewness, float Kurtosis) CalculateBoth(float[] data)
    {
        if (data == null || data.Length == 0)
        {
            throw new ArgumentException("Data array cannot be null or empty", nameof(data));
        }

        if (data.Length < 4)
        {
            return (0, 0);
        }

        (float mean, float std) = CalculateMeanAndStd(data);

        if (std == 0)
        {
            return (0, 0);
        }

        return (CalculateSkewness(data, mean, std), CalculateKurtosis(data, mean, std));
    }

    /// <summary>
    /// Calculates mean and standard deviation in a single pass
    /// Uses Welford's algorithm for numerical stability
    /// </summary>
    private static (float Mean, float Std) CalculateMeanAndStd(float[] data)
    {
        int n = data.Length;

        if (n == 1)
        {
            return (data[0], 0);
        }

        float mean = 0f;
        float m2 = 0f; // Sum of squared differences from current mean

        // Welford's algorithm
        for (int i = 0; i < n; i++)
        {
            float delta = data[i] - mean;
            mean += delta / (i + 1);
            float delta2 = data[i] - mean;
            m2 += delta * delta2;
        }

        float variance = m2 / n;
        float std = (float)Math.Sqrt(variance);

        return (mean, std);
    }
}
