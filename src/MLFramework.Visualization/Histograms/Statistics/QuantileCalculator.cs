namespace MLFramework.Visualization.Histograms.Statistics;

/// <summary>
/// Efficient calculator for quantiles (percentiles) of data distributions
/// </summary>
public static class QuantileCalculator
{
    /// <summary>
    /// Calculates the specified quantile from the data
    /// </summary>
    /// <param name="data">Input data array</param>
    /// <param name="quantile">Quantile to calculate (0.0 to 1.0)</param>
    /// <returns>The quantile value</returns>
    public static float Calculate(float[] data, float quantile)
    {
        if (data == null || data.Length == 0)
        {
            throw new ArgumentException("Data array cannot be null or empty", nameof(data));
        }

        if (quantile < 0.0f || quantile > 1.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(quantile), "Quantile must be between 0.0 and 1.0");
        }

        if (data.Length == 1)
        {
            return data[0];
        }

        // For small arrays, sorting is faster than quickselect
        if (data.Length < 1000)
        {
            return CalculateSorted(SortCopy(data), quantile);
        }

        // Use quickselect for larger arrays
        return CalculateQuickSelect(data, quantile);
    }

    /// <summary>
    /// Calculates multiple quantiles efficiently (reuses sorted data)
    /// </summary>
    /// <param name="data">Input data array</param>
    /// <param name="quantiles">Array of quantiles to calculate</param>
    /// <returns>Array of quantile values</returns>
    public static float[] CalculateMultiple(float[] data, float[] quantiles)
    {
        if (data == null || data.Length == 0)
        {
            throw new ArgumentException("Data array cannot be null or empty", nameof(data));
        }

        if (quantiles == null || quantiles.Length == 0)
        {
            throw new ArgumentException("Quantiles array cannot be null or empty", nameof(quantiles));
        }

        if (data.Length == 1)
        {
            var result = new float[quantiles.Length];
            Array.Fill(result, data[0]);
            return result;
        }

        // Sort data once and use it for all quantiles
        var sorted = SortCopy(data);
        var results = new float[quantiles.Length];

        for (int i = 0; i < quantiles.Length; i++)
        {
            results[i] = CalculateSorted(sorted, quantiles[i]);
        }

        return results;
    }

    /// <summary>
    /// Calculates quantile from pre-sorted data using linear interpolation
    /// </summary>
    private static float CalculateSorted(float[] sortedData, float quantile)
    {
        int n = sortedData.Length;
        float position = quantile * (n - 1);
        int lowerIndex = (int)Math.Floor(position);
        int upperIndex = (int)Math.Ceiling(position);

        if (lowerIndex == upperIndex)
        {
            return sortedData[lowerIndex];
        }

        float lowerValue = sortedData[lowerIndex];
        float upperValue = sortedData[upperIndex];
        float fraction = position - lowerIndex;

        return lowerValue + fraction * (upperValue - lowerValue);
    }

    /// <summary>
    /// Creates a sorted copy of the data
    /// </summary>
    private static float[] SortCopy(float[] data)
    {
        var copy = new float[data.Length];
        Array.Copy(data, copy, data.Length);
        Array.Sort(copy);
        return copy;
    }

    /// <summary>
    /// Uses quickselect algorithm to find quantile without full sorting
    /// More efficient for large arrays when only one quantile is needed
    /// </summary>
    private static float CalculateQuickSelect(float[] data, float quantile)
    {
        int n = data.Length;
        float position = quantile * (n - 1);
        int k = (int)Math.Round(position);

        // Make a copy to avoid modifying original data
        var copy = new float[n];
        Array.Copy(data, copy, n);

        return QuickSelect(copy, 0, n - 1, k);
    }

    /// <summary>
    /// Quickselect algorithm implementation
    /// </summary>
    private static float QuickSelect(float[] arr, int left, int right, int k)
    {
        if (left == right)
        {
            return arr[left];
        }

        // Randomly select pivot to avoid worst-case performance
        int pivotIndex = new Random().Next(left, right + 1);
        pivotIndex = Partition(arr, left, right, pivotIndex);

        if (k == pivotIndex)
        {
            return arr[k];
        }
        else if (k < pivotIndex)
        {
            return QuickSelect(arr, left, pivotIndex - 1, k);
        }
        else
        {
            return QuickSelect(arr, pivotIndex + 1, right, k);
        }
    }

    /// <summary>
    /// Partition helper for quickselect
    /// </summary>
    private static int Partition(float[] arr, int left, int right, int pivotIndex)
    {
        float pivotValue = arr[pivotIndex];

        // Move pivot to end
        (arr[pivotIndex], arr[right]) = (arr[right], arr[pivotIndex]);

        int storeIndex = left;
        for (int i = left; i < right; i++)
        {
            if (arr[i] < pivotValue)
            {
                (arr[storeIndex], arr[i]) = (arr[i], arr[storeIndex]);
                storeIndex++;
            }
        }

        // Move pivot to its final place
        (arr[right], arr[storeIndex]) = (arr[storeIndex], arr[right]);

        return storeIndex;
    }
}
