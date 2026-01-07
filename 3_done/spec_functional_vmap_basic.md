# Spec: Basic Vectorization (vmap)

## Overview
Implement the basic vmap transformation that automatically transforms functions operating on single data points to work on batches.

## Scope
- Implement `Vectorize<T>` method
- Support single-axis vectorization
- Handle batch dimension insertion
- Preserve function semantics

## Technical Requirements

### 1. Vectorize Extension Method

```csharp
namespace MLFramework.Functional
{
    public static class Functional
    {
        /// <summary>
        /// Transforms a function that operates on single tensors to work on batches.
        /// </summary>
        /// <typeparam name="TInput">Input tensor type</typeparam>
        /// <typeparam name="TOutput">Output tensor type</typeparam>
        /// <param name="func">Function to vectorize</param>
        /// <param name="axis">Batch axis (default: 0)</param>
        /// <returns>Batched function</returns>
        public static Func<Tensor, Tensor> Vectorize(
            Func<Tensor, Tensor> func,
            int axis = 0)
        {
            var transform = new VMapTransform(func, axis);
            return (Func<Tensor, Tensor>)transform.Transform(func);
        }

        /// <summary>
        /// Vectorize a function with multiple input tensors.
        /// </summary>
        public static Func<Tensor, Tensor, Tensor> Vectorize(
            Func<Tensor, Tensor, Tensor> func,
            int axis = 0)
        {
            var transform = new VMapTransform(func, axis);
            return (Func<Tensor, Tensor, Tensor>)transform.Transform(func);
        }
    }
}
```

### 2. VMapTransform Implementation

```csharp
public class VMapTransform : BaseTransformation
{
    private readonly int _axis;

    public VMapTransform(Delegate original, int axis = 0)
        : base("vmap", TransformationType.Vectorization)
    {
        _axis = axis;
        ValidateDelegate(original);
    }

    public override Delegate Transform(Delegate original)
    {
        // Based on delegate type, create appropriate wrapper
        var method = original.Method;
        var returnType = method.ReturnType;

        if (!typeof(Tensor).IsAssignableFrom(returnType))
        {
            throw new NotSupportedException("vmap only supports functions returning Tensor");
        }

        // Handle single input: Func<Tensor, Tensor>
        if (method.GetParameters().Length == 1 &&
            method.GetParameters()[0].ParameterType == typeof(Tensor))
        {
            return CreateSingleInputWrapper((Func<Tensor, Tensor>)original);
        }

        // Handle double input: Func<Tensor, Tensor, Tensor>
        if (method.GetParameters().Length == 2 &&
            method.GetParameters()[0].ParameterType == typeof(Tensor) &&
            method.GetParameters()[1].ParameterType == typeof(Tensor))
        {
            return CreateDoubleInputWrapper((Func<Tensor, Tensor, Tensor>)original);
        }

        throw new NotSupportedException("Unsupported delegate signature for vmap");
    }

    private Func<Tensor, Tensor> CreateSingleInputWrapper(Func<Tensor, Tensor> original)
    {
        return (Tensor batchInput) =>
        {
            // Validate input has batch dimension
            if (batchInput.Rank <= _axis)
            {
                throw new ArgumentException($"Input tensor must have at least {_axis + 1} dimensions for axis={_axis}");
            }

            var batchSize = batchInput.Shape[_axis];
            var outputs = new List<Tensor>();

            // Iterate over batch dimension
            for (int i = 0; i < batchSize; i++)
            {
                // Extract single element from batch
                var singleInput = batchInput.Take(_axis, i);
                var output = original(singleInput);
                outputs.Add(output);
            }

            // Stack outputs along batch dimension
            return Tensor.Stack(outputs, _axis);
        };
    }

    private Func<Tensor, Tensor, Tensor> CreateDoubleInputWrapper(Func<Tensor, Tensor, Tensor> original)
    {
        return (Tensor batchInput1, Tensor batchInput2) =>
        {
            // Validate inputs have same batch size
            var batchSize1 = batchInput1.Shape[_axis];
            var batchSize2 = batchInput2.Shape[_axis];

            if (batchSize1 != batchSize2)
            {
                throw new ArgumentException($"Batch dimensions must match: {batchSize1} != {batchSize2}");
            }

            var batchSize = batchSize1;
            var outputs = new List<Tensor>();

            for (int i = 0; i < batchSize; i++)
            {
                var singleInput1 = batchInput1.Take(_axis, i);
                var singleInput2 = batchInput2.Take(_axis, i);
                var output = original(singleInput1, singleInput2);
                outputs.Add(output);
            }

            return Tensor.Stack(outputs, _axis);
        };
    }
}
```

### 3. Tensor Extension Methods (if needed)

```csharp
public static class VMapTensorExtensions
{
    /// <summary>
    /// Takes a slice from tensor along specified axis.
    /// </summary>
    public static Tensor Take(this Tensor tensor, int axis, int index)
    {
        // Implementation: extract tensor[axis == index]
        // This should use existing tensor indexing operations
        // Example: tensor[index] for axis=0
        return tensor[Slice.At(index, axis)];
    }

    /// <summary>
    /// Stacks tensors along specified axis.
    /// </summary>
    public static Tensor Stack(IEnumerable<Tensor> tensors, int axis)
    {
        // Implementation: concatenate tensors along new axis
        // If axis=0, creates new first dimension
        return Tensor.Stack(tensors.ToArray(), axis);
    }
}
```

## Files to Create
1. `src/MLFramework/Functional/VMapTransform.cs`
2. `src/MLFramework/Functional/VMapTensorExtensions.cs`
3. Update `src/MLFramework/Functional/Functional.cs` with Vectorize methods

## Dependencies
- spec_functional_core_interfaces.md (must be completed first)
- MLFramework.Tensor with basic indexing and concatenation operations

## Success Criteria
- Can vectorize simple single-input functions
- Can vectorize double-input functions
- Batch dimensions are correctly handled
- Function semantics are preserved
- Errors are thrown for incompatible inputs

## Notes for Coder
- Start with the simplest implementation using loops
- Tensor indexing/concatenation operations may need to be implemented if not available
- Optimize in later specs (this is basic implementation)
- Include comprehensive error messages
- Test with both 1D and multidimensional tensors
