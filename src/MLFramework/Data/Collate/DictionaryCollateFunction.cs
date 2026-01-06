using System.Collections.Generic;

namespace MLFramework.Data.Collate;

/// <summary>
/// Collate function that handles batches where each sample is a dictionary of arrays.
/// </summary>
public class DictionaryCollateFunction : ICollateFunction<Dictionary<string, object>>
{
    /// <summary>
    /// Collates a batch of dictionaries into a dictionary of collated arrays.
    /// </summary>
    /// <param name="batch">An array of dictionaries to collate.</param>
    /// <returns>A dictionary where each value is a collated array.</returns>
    /// <exception cref="ArgumentException">Thrown when batch is null, empty, or samples have missing/invalid keys.</exception>
    public object Collate(Dictionary<string, object>[] batch)
    {
        if (batch == null || batch.Length == 0)
            throw new ArgumentException("Batch cannot be empty");

        var result = new Dictionary<string, object>();
        var keys = batch[0].Keys;

        foreach (var key in keys)
        {
            var values = new object[batch.Length];

            for (int i = 0; i < batch.Length; i++)
            {
                if (!batch[i].ContainsKey(key))
                    throw new ArgumentException($"Sample {i} missing key '{key}'");

                values[i] = batch[i][key];
            }

            // Determine collation strategy based on first value type
            if (values[0] is float[][])
            {
                // Convert object[] to float[][][] for stacking
                var imageBatch = new float[batch.Length][][];
                for (int i = 0; i < batch.Length; i++)
                {
                    imageBatch[i] = (float[][])values[i];
                }

                var stacker = new StackCollateFunction();
                result[key] = stacker.Collate(imageBatch);
            }
            else
            {
                // Default: just array of values
                result[key] = values;
            }
        }

        return result;
    }
}
