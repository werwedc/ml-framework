namespace MLFramework.Visualization.Histograms.Statistics;

/// <summary>
/// Detector for identifying outliers in data distributions
/// </summary>
public static class OutlierDetector
{
    /// <summary>
    /// Detects outliers using the standard deviation method
    /// Values more than threshold standard deviations from the mean are considered outliers
    /// </summary>
    /// <param name="data">Input data array</param>
    /// <param name="threshold">Number of standard deviations (default: 3.0)</param>
    /// <returns>Tuple containing (outlierCount, outlierIndices)</returns>
    public static (int Count, int[] Indices) DetectOutliersByStd(float[] data, float threshold = 3.0f)
    {
        if (data == null || data.Length == 0)
        {
            throw new ArgumentException("Data array cannot be null or empty", nameof(data));
        }

        if (threshold <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold), "Threshold must be greater than 0");
        }

        if (data.Length < 2)
        {
            return (0, Array.Empty<int>());
        }

        // Calculate mean and standard deviation
        (float mean, float std) = CalculateMeanAndStd(data);

        if (std == 0)
        {
            return (0, Array.Empty<int>()); // No outliers if all values are identical
        }

        // Find outliers
        var outlierIndices = new List<int>();
        float upperBound = mean + threshold * std;
        float lowerBound = mean - threshold * std;

        for (int i = 0; i < data.Length; i++)
        {
            if (data[i] > upperBound || data[i] < lowerBound)
            {
                outlierIndices.Add(i);
            }
        }

        return (outlierIndices.Count, outlierIndices.ToArray());
    }

    /// <summary>
    /// Detects outliers using the interquartile range (IQR) method
    /// Values below Q1 - k*IQR or above Q3 + k*IQR are considered outliers
    /// </summary>
    /// <param name="data">Input data array</param>
    /// <param name="iqrMultiplier">IQR multiplier (default: 1.5 for standard Tukey fences)</param>
    /// <returns>Tuple containing (outlierCount, outlierIndices)</returns>
    public static (int Count, int[] Indices) DetectOutliersByIQR(float[] data, float iqrMultiplier = 1.5f)
    {
        if (data == null || data.Length == 0)
        {
            throw new ArgumentException("Data array cannot be null or empty", nameof(data));
        }

        if (iqrMultiplier <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(iqrMultiplier), "IQR multiplier must be greater than 0");
        }

        if (data.Length < 4)
        {
            return (0, Array.Empty<int>());
        }

        // Calculate quartiles
        float q1 = QuantileCalculator.Calculate(data, 0.25f);
        float q3 = QuantileCalculator.Calculate(data, 0.75f);
        float iqr = q3 - q1;

        if (iqr == 0)
        {
            return (0, Array.Empty<int>()); // No outliers if all values are identical
        }

        // Find outliers
        var outlierIndices = new List<int>();
        float upperBound = q3 + iqrMultiplier * iqr;
        float lowerBound = q1 - iqrMultiplier * iqr;

        for (int i = 0; i < data.Length; i++)
        {
            if (data[i] > upperBound || data[i] < lowerBound)
            {
                outlierIndices.Add(i);
            }
        }

        return (outlierIndices.Count, outlierIndices.ToArray());
    }

    /// <summary>
    /// Counts the number of dead neurons (zero-valued entries) in the data
    /// </summary>
    /// <param name="data">Input data array</param>
    /// <param name="nearZeroThreshold">Threshold for considering a value "near zero" (default: 0.0, exact zeros only)</param>
    /// <returns>Count of dead neurons</returns>
    public static int CountDeadNeurons(float[] data, float nearZeroThreshold = 0.0f)
    {
        if (data == null || data.Length == 0)
        {
            throw new ArgumentException("Data array cannot be null or empty", nameof(data));
        }

        if (nearZeroThreshold < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(nearZeroThreshold), "Near-zero threshold cannot be negative");
        }

        int deadCount = 0;

        for (int i = 0; i < data.Length; i++)
        {
            if (Math.Abs(data[i]) <= nearZeroThreshold)
            {
                deadCount++;
            }
        }

        return deadCount;
    }

    /// <summary>
    /// Gets the values of outliers
    /// </summary>
    /// <param name="data">Input data array</param>
    /// <param name="threshold">Number of standard deviations (default: 3.0)</param>
    /// <returns>Array of outlier values</returns>
    public static float[] GetOutlierValues(float[] data, float threshold = 3.0f)
    {
        if (data == null || data.Length == 0)
        {
            throw new ArgumentException("Data array cannot be null or empty", nameof(data));
        }

        (_, int[] indices) = DetectOutliersByStd(data, threshold);

        if (indices.Length == 0)
        {
            return Array.Empty<float>();
        }

        var values = new float[indices.Length];
        for (int i = 0; i < indices.Length; i++)
        {
            values[i] = data[indices[i]];
        }

        return values;
    }

    /// <summary>
    /// Gets the outlier bounds (lower and upper) for the given threshold
    /// </summary>
    /// <param name="data">Input data array</param>
    /// <param name="threshold">Number of standard deviations (default: 3.0)</param>
    /// <returns>Tuple containing (lowerBound, upperBound)</returns>
    public static (float LowerBound, float UpperBound) GetOutlierBounds(float[] data, float threshold = 3.0f)
    {
        if (data == null || data.Length == 0)
        {
            throw new ArgumentException("Data array cannot be null or empty", nameof(data));
        }

        if (threshold <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold), "Threshold must be greater than 0");
        }

        if (data.Length < 2)
        {
            return (data[0], data[0]);
        }

        (float mean, float std) = CalculateMeanAndStd(data);
        float upperBound = mean + threshold * std;
        float lowerBound = mean - threshold * std;

        return (lowerBound, upperBound);
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
