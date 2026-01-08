namespace MachineLearning.Visualization.Scalars.Smoothing;

/// <summary>
/// Provides resampling functionality for scalar series to reduce data points while preserving trends
/// </summary>
public class Resampler
{
    /// <summary>
    /// Resamples a series to a target number of points using linear interpolation
    /// </summary>
    /// <param name="values">Values to resample</param>
    /// <param name="targetCount">Target number of points</param>
    /// <returns>Resampled values</returns>
    public static float[] Resample(float[] values, int targetCount)
    {
        if (values == null) throw new ArgumentNullException(nameof(values));
        if (targetCount <= 0) throw new ArgumentException("Target count must be positive", nameof(targetCount));

        if (values.Length == 0) return Array.Empty<float>();
        if (values.Length <= targetCount) return (float[])values.Clone();

        float[] resampled = new float[targetCount];
        float stepSize = (float)(values.Length - 1) / (targetCount - 1);

        for (int i = 0; i < targetCount; i++)
        {
            float exactIndex = i * stepSize;
            int lowerIndex = (int)Math.Floor(exactIndex);
            int upperIndex = Math.Min(lowerIndex + 1, values.Length - 1);

            if (lowerIndex == upperIndex)
            {
                resampled[i] = values[lowerIndex];
            }
            else
            {
                float alpha = exactIndex - lowerIndex;
                resampled[i] = values[lowerIndex] * (1 - alpha) + values[upperIndex] * alpha;
            }
        }

        return resampled;
    }

    /// <summary>
    /// Resamples a series with steps using linear interpolation
    /// </summary>
    /// <param name="values">Values to resample</param>
    /// <param name="steps">Step numbers for each value</param>
    /// <param name="targetCount">Target number of points</param>
    /// <returns>Tuple of (resampled values, resampled steps)</returns>
    public static (float[] values, long[] steps) Resample(
        float[] values,
        long[] steps,
        int targetCount)
    {
        if (values == null) throw new ArgumentNullException(nameof(values));
        if (steps == null) throw new ArgumentNullException(nameof(steps));
        if (values.Length != steps.Length)
            throw new ArgumentException("Values and steps must have the same length");
        if (targetCount <= 0) throw new ArgumentException("Target count must be positive", nameof(targetCount));

        if (values.Length == 0) return (Array.Empty<float>(), Array.Empty<long>());
        if (values.Length <= targetCount) return ((float[])values.Clone(), (long[])steps.Clone());

        float[] resampledValues = new float[targetCount];
        long[] resampledSteps = new long[targetCount];
        float stepSize = (float)(values.Length - 1) / (targetCount - 1);

        for (int i = 0; i < targetCount; i++)
        {
            float exactIndex = i * stepSize;
            int lowerIndex = (int)Math.Floor(exactIndex);
            int upperIndex = Math.Min(lowerIndex + 1, values.Length - 1);

            if (lowerIndex == upperIndex)
            {
                resampledValues[i] = values[lowerIndex];
                resampledSteps[i] = steps[lowerIndex];
            }
            else
            {
                float alpha = exactIndex - lowerIndex;
                resampledValues[i] = values[lowerIndex] * (1 - alpha) + values[upperIndex] * alpha;
                resampledSteps[i] = steps[lowerIndex] + (long)((steps[upperIndex] - steps[lowerIndex]) * alpha);
            }
        }

        return (resampledValues, resampledSteps);
    }

    /// <summary>
    /// Resamples a series using the Largest Triangle Three Buckets (LTTB) algorithm
    /// which preserves important peaks and valleys better than linear interpolation
    /// </summary>
    /// <param name="values">Values to resample</param>
    /// <param name="targetCount">Target number of points</param>
    /// <returns>Resampled values</returns>
    public static float[] ResampleLTTB(float[] values, int targetCount)
    {
        if (values == null) throw new ArgumentNullException(nameof(values));
        if (targetCount <= 0) throw new ArgumentException("Target count must be positive", nameof(targetCount));

        if (values.Length == 0) return Array.Empty<float>();
        if (values.Length <= targetCount) return (float[])values.Clone();

        // LTTB algorithm
        float[] resampled = new float[targetCount];
        int bucketSize = (values.Length - 2) / (targetCount - 2);

        // First point is always included
        resampled[0] = values[0];

        int a = 0;
        int nextA = 0;

        for (int i = 0; i < targetCount - 2; i++)
        {
            // Calculate average of next bucket
            int avgRangeStart = nextA + bucketSize + 1;
            int avgRangeEnd = Math.Min(avgRangeStart + bucketSize, values.Length - 1);

            float avgX = 0;
            float avgY = 0;
            for (int j = avgRangeStart; j <= avgRangeEnd; j++)
            {
                avgX += j;
                avgY += values[j];
            }
            avgX /= (avgRangeEnd - avgRangeStart + 1);
            avgY /= (avgRangeEnd - avgRangeStart + 1);

            // Find point in current bucket with largest triangle area
            int rangeStart = nextA + 1;
            int rangeEnd = Math.Min(rangeStart + bucketSize, avgRangeStart - 1);

            float maxArea = -1;
            int maxIdx = rangeStart;

            for (int j = rangeStart; j <= rangeEnd; j++)
            {
                // Area of triangle formed by (a, values[a]), (j, values[j]), (avgX, avgY)
                float area = Math.Abs(
                    (a - avgX) * (values[j] - avgY) -
                    (a - j) * (values[a] - avgY)
                ) * 0.5f;

                if (area > maxArea)
                {
                    maxArea = area;
                    maxIdx = j;
                }
            }

            resampled[i + 1] = values[maxIdx];
            a = maxIdx;
            nextA = avgRangeEnd;
        }

        // Last point is always included
        resampled[targetCount - 1] = values[values.Length - 1];

        return resampled;
    }

    /// <summary>
    /// Downsamples a series by taking every nth point
    /// </summary>
    /// <param name="values">Values to downsample</param>
    /// <param name="factor">Downsampling factor (n)</param>
    /// <returns>Downsampled values</returns>
    public static float[] Downsample(float[] values, int factor)
    {
        if (values == null) throw new ArgumentNullException(nameof(values));
        if (factor <= 0) throw new ArgumentException("Factor must be positive", nameof(factor));

        if (values.Length == 0) return Array.Empty<float>();

        int targetCount = (values.Length + factor - 1) / factor;
        float[] downsampled = new float[targetCount];

        for (int i = 0; i < targetCount; i++)
        {
            downsampled[i] = values[i * factor];
        }

        return downsampled;
    }
}
