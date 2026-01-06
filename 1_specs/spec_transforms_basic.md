# Spec: Transform Interface and Basic Transforms

## Overview
Define the transform abstraction and implement basic preprocessing transforms.

## Requirements

### Interface

#### ITransform
```csharp
public interface ITransform
{
    object Apply(object input);
}
```

#### ITransform<TInput, TOutput>
```csharp
public interface ITransform<TInput, TOutput>
{
    TOutput Apply(TInput input);
}
```

### Basic Transforms

#### ToTensorTransform
- Converts input (image/array) to tensor representation
- Placeholder for now - actual tensor conversion requires tensor library
- Input: `float[,]` (2D array as image)
- Output: `Tensor` (placeholder - will be defined in tensor spec)

```csharp
public class ToTensorTransform : ITransform<float[,], object>
{
    public object Apply(float[,] input)
    {
        // Placeholder - will integrate with tensor library later
        return new { Data = input, Shape = new[] { input.GetLength(0), input.GetLength(1) } };
    }
}
```

#### NormalizeTransform
- Applies mean and standard deviation normalization
- Input: `float[,]` (2D array)
- Output: `float[,]` (normalized array)

```csharp
public class NormalizeTransform : ITransform<float[,], float[,]>
{
    private readonly float[] _mean;
    private readonly float[] _std;

    public NormalizeTransform(float[] mean, float[] std)
    {
        if (mean == null || std == null)
            throw new ArgumentNullException();

        if (mean.Length != std.Length)
            throw new ArgumentException("Mean and std must have same length");

        _mean = mean;
        _std = std;
    }

    public float[,] Apply(float[,] input)
    {
        int height = input.GetLength(0);
        int width = input.GetLength(1);
        var output = new float[height, width];

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                int channel = j % _mean.Length; // Simple channel mapping
                output[i, j] = (input[i, j] - _mean[channel]) / _std[channel];
            }
        }

        return output;
    }
}
```

#### LambdaTransform
- Generic wrapper for user-defined functions
- Allows custom transforms without creating new class
- Input: `TInput`, Output: `TOutput`

```csharp
public class LambdaTransform<TInput, TOutput> : ITransform<TInput, TOutput>
{
    private readonly Func<TInput, TOutput> _func;

    public LambdaTransform(Func<TInput, TOutput> func)
    {
        _func = func ?? throw new ArgumentNullException(nameof(func));
    }

    public TOutput Apply(TInput input) => _func(input);
}
```

### Error Handling
- `ArgumentNullException` for null inputs
- `ArgumentException` for mismatched dimensions in NormalizeTransform
- Validation in constructors

## Acceptance Criteria
1. ITransform interface allows generic input/output types
2. ToTensorTransform wraps input in placeholder structure
3. NormalizeTransform correctly applies mean/std normalization
4. LambdaTransform executes provided function correctly
5. All transforms handle edge cases (null, empty)
6. Unit tests verify normalization formula: (x - mean) / std
7. Unit tests verify LambdaTransform with various functions

## Files to Create
- `src/Data/ITransform.cs`
- `src/Data/ITransform{TInput,TOutput}.cs`
- `src/Data/Transforms/ToTensorTransform.cs`
- `src/Data/Transforms/NormalizeTransform.cs`
- `src/Data/Transforms/LambdaTransform.cs`

## Tests
- `tests/Data/Transforms/ToTensorTransformTests.cs`
- `tests/Data/Transforms/NormalizeTransformTests.cs`
- `tests/Data/Transforms/LambdaTransformTests.cs`

## Notes
- Keep transforms simple and composable
- ToTensorTransform is placeholder until tensor library exists
- NormalizeTransform assumes channel-last format for now
- LambdaTransform provides flexibility for custom preprocessing
- Future specs will add image-specific transforms (resize, crop, flip)
- Consider adding support for multi-channel images properly
