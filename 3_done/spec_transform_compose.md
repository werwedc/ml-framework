# Spec: ComposeTransform

## Overview
Implement a transform that chains multiple transforms together in sequence.

## Requirements

### Implementation

#### ComposeTransform
- Chains multiple transforms in sequence
- Output of one transform becomes input to the next
- Supports any number of transforms
- Type-safe when using ITransform<TInput, TOutput>

**Key Constructor:**
```csharp
public class ComposeTransform : ITransform
{
    private readonly ITransform[] _transforms;

    public ComposeTransform(params ITransform[] transforms)
    {
        if (transforms == null || transforms.Length == 0)
            throw new ArgumentException("At least one transform is required");

        // Check for null transforms
        if (transforms.Any(t => t == null))
            throw new ArgumentNullException(nameof(transforms));

        _transforms = transforms;
    }

    public object Apply(object input)
    {
        object current = input;

        foreach (var transform in _transforms)
        {
            current = transform.Apply(current);
        }

        return current;
    }
}
```

**Typed Version:**
```csharp
public class ComposeTransform<TInput, TOutput> : ITransform<TInput, TOutput>
{
    private readonly Func<TInput, TOutput> _composedFunc;

    public ComposeTransform(params ITransform[] transforms)
    {
        if (transforms == null || transforms.Length == 0)
            throw new ArgumentException("At least one transform is required");

        if (transforms.Any(t => t == null))
            throw new ArgumentNullException(nameof(transforms));

        _composedFunc = BuildChain(transforms);
    }

    private Func<TInput, TOutput> BuildChain(ITransform[] transforms)
    {
        return input =>
        {
            object current = input;

            foreach (var transform in transforms)
            {
                current = transform.Apply(current);
            }

            return (TOutput)current;
        };
    }

    public TOutput Apply(TInput input) => _composedFunc(input);
}
```

### Error Handling
- `ArgumentException` if transforms array is null or empty
- `ArgumentNullException` if any transform is null
- `InvalidCastException` if type mismatch occurs in chain

## Acceptance Criteria
1. ComposeTransform chains transforms in the order provided
2. Output of transform N becomes input to transform N+1
3. Supports single transform (trivial case)
4. Supports multiple transforms (2+)
5. Empty transform list throws exception
6. Null transform in list throws exception
7. Type-casting exceptions propagate correctly
8. Unit tests verify sequential application of transforms
9. Unit tests verify type safety in typed version

## Files to Create
- `src/Data/Transforms/ComposeTransform.cs`

## Tests
- `tests/Data/Transforms/ComposeTransformTests.cs`

## Usage Example
```csharp
var transform = new ComposeTransform(
    new NormalizeTransform(mean: new[] {0.5f}, std: new[] {0.5f}),
    new LambdaTransform<float[,], float[,]>(arr => MultiplyByTwo(arr))
);

var result = transform.Apply(inputImage);
```

## Notes
- Transforms are applied in array order: transforms[0] -> transforms[1] -> ... -> transforms[n]
- No automatic type checking - relies on runtime casting
- Consider adding type-safe builder pattern in future
- Common pattern: Normalize -> ToTensor -> (future GPU transforms)
- Keep implementation simple - defer to individual transforms for validation
- Immutable once created - transforms cannot be modified after construction
