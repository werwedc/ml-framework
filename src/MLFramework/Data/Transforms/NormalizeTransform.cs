namespace MLFramework.Data.Transforms;

/// <summary>
/// Transform that applies mean and standard deviation normalization to 2D array data.
/// </summary>
public class NormalizeTransform : ITransform<float[,], float[,]>
{
    private readonly float[] _mean;
    private readonly float[] _std;

    /// <summary>
    /// Creates a new NormalizeTransform with the specified mean and standard deviation values.
    /// </summary>
    /// <param name="mean">Array of mean values for each channel.</param>
    /// <param name="std">Array of standard deviation values for each channel.</param>
    /// <exception cref="ArgumentNullException">Thrown if mean or std is null.</exception>
    /// <exception cref="ArgumentException">Thrown if mean and std arrays have different lengths.</exception>
    public NormalizeTransform(float[] mean, float[] std)
    {
        if (mean == null)
            throw new ArgumentNullException(nameof(mean));

        if (std == null)
            throw new ArgumentNullException(nameof(std));

        if (mean.Length != std.Length)
            throw new ArgumentException("Mean and std must have the same length", nameof(std));

        _mean = mean;
        _std = std;
    }

    /// <summary>
    /// Applies normalization to the input array using the formula: (x - mean) / std
    /// </summary>
    /// <param name="input">The input 2D array to normalize.</param>
    /// <returns>The normalized 2D array.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    public float[,] Apply(float[,] input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        int height = input.GetLength(0);
        int width = input.GetLength(1);
        var output = new float[height, width];
        int numChannels = _mean.Length;

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                int channel = j % numChannels; // Simple channel mapping
                output[i, j] = (input[i, j] - _mean[channel]) / _std[channel];
            }
        }

        return output;
    }
}
