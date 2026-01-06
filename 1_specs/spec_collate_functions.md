# Spec: Basic Collate Functions

## Overview
Implement collate functions that combine individual samples into batches.

## Requirements

### Interface

#### ICollateFunction<T>
```csharp
public interface ICollateFunction<T>
{
    object Collate(T[] batch);
}
```

### Implementations

#### DefaultCollateFunction
- Stacks samples into batch by creating a jagged array
- Simple pass-through for primitive types
- Placeholder behavior for complex types

```csharp
public class DefaultCollateFunction<T> : ICollateFunction<T>
{
    public object Collate(T[] batch)
    {
        if (batch == null)
            throw new ArgumentNullException(nameof(batch));

        return batch;
    }
}
```

#### StackCollateFunction
- Stacks arrays into a 3D array (batch, height, width)
- Assumes all samples have same dimensions
- Input: `float[][,]` (array of 2D arrays)
- Output: `float[,,]` (3D batched array)

```csharp
public class StackCollateFunction : ICollateFunction<float[,]>
{
    public object Collate(float[,][] batch)
    {
        if (batch == null || batch.Length == 0)
            throw new ArgumentException("Batch cannot be empty");

        // Get dimensions from first sample
        int batchSize = batch.Length;
        int height = batch[0].GetLength(0);
        int width = batch[0].GetLength(1);

        // Validate all samples have same dimensions
        for (int i = 1; i < batch.Length; i++)
        {
            if (batch[i].GetLength(0) != height || batch[i].GetLength(1) != width)
                throw new ArgumentException("All samples must have the same dimensions");
        }

        // Stack into 3D array
        var result = new float[batchSize, height, width];

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    result[b, h, w] = batch[b][h, w];
                }
            }
        }

        return result;
    }
}
```

#### DictionaryCollateFunction
- Handles batches where each sample is a dictionary of arrays
- Collates each value independently
- Input: `Dictionary<string, object>[]` (array of dictionaries)
- Output: `Dictionary<string, object>` (dictionary of collated arrays)

```csharp
public class DictionaryCollateFunction : ICollateFunction<Dictionary<string, object>>
{
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
            if (values[0] is float[,][] imageBatch)
            {
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
```

### Error Handling
- `ArgumentNullException` for null batches
- `ArgumentException` for empty batches
- `ArgumentException` for dimension mismatches in StackCollateFunction
- `ArgumentException` for missing dictionary keys

## Acceptance Criteria
1. DefaultCollateFunction returns input array unchanged
2. StackCollateFunction creates 3D array from 2D samples
3. StackCollateFunction validates uniform dimensions
4. DictionaryCollateFunction collates each dictionary value
5. DictionaryCollateFunction validates all samples have same keys
6. All functions handle edge cases (null, empty, single sample)
7. Unit tests cover various batch sizes and dimensions

## Files to Create
- `src/Data/ICollateFunction.cs`
- `src/Data/Collate/DefaultCollateFunction.cs`
- `src/Data/Collate/StackCollateFunction.cs`
- `src/Data/Collate/DictionaryCollateFunction.cs`

## Tests
- `tests/Data/Collate/DefaultCollateFunctionTests.cs`
- `tests/Data/Collate/StackCollateFunctionTests.cs`
- `tests/Data/Collate/DictionaryCollateFunctionTests.cs`

## Usage Example
```csharp
// Stack 2D images into 3D batch
var collator = new StackCollateFunction();
var batchedImages = collator.Collate(imageArray); // float[,,]

// Handle dictionary samples
var dictCollator = new DictionaryCollateFunction();
var batchedDict = dictCollator.Collate(dictArray); // Dict<string, object>
```

## Notes
- Collate functions are critical for efficient batching
- Future specs will add padding collate for variable-length sequences
- StackCollateFunction assumes all samples are same size
- DictionaryCollateFunction delegates to specialized collators when possible
- Performance matters - avoid unnecessary copies
- Consider adding async collation for large batches
