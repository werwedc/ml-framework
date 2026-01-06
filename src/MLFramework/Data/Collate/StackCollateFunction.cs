namespace MLFramework.Data.Collate;

/// <summary>
/// Collate function that stacks 2D arrays into a 3D batched array.
/// </summary>
public class StackCollateFunction
{
    /// <summary>
    /// Stacks an array of 2D arrays into a 3D batched array (batch, height, width).
    /// </summary>
    /// <param name="batch">An array of 2D float arrays (jagged 3D array) to stack.</param>
    /// <returns>A 3D array with dimensions (batchSize, height, width).</returns>
    /// <exception cref="ArgumentException">Thrown when batch is null, empty, or samples have inconsistent dimensions.</exception>
    public float[,,] Collate(float[][][] batch)
    {
        if (batch == null || batch.Length == 0)
            throw new ArgumentException("Batch cannot be empty");

        // Get dimensions from first sample
        int batchSize = batch.Length;
        int height = batch[0].Length;
        int width = batch[0][0].Length;

        // Validate all samples have same dimensions
        for (int i = 1; i < batch.Length; i++)
        {
            if (batch[i].Length != height)
                throw new ArgumentException($"All samples must have the same dimensions. Sample 0 has height {height}, but sample {i} has height {batch[i].Length}");

            for (int j = 0; j < batch[i].Length; j++)
            {
                if (batch[i][j].Length != width)
                    throw new ArgumentException($"All samples must have the same dimensions. Sample 0 has width {width}, but sample {i} row {j} has width {batch[i][j].Length}");
            }
        }

        // Stack into 3D array
        var result = new float[batchSize, height, width];

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    result[b, h, w] = batch[b][h][w];
                }
            }
        }

        return result;
    }
}
