# Spec: Multi-Axis Vectorization

## Overview
Extend vmap to support mapping over multiple axes simultaneously, enabling more flexible batch transformations.

## Scope
- Add in_axes parameter to Vectorize
- Support mapping different inputs on different axes
- Handle None/in_axes for non-batched parameters
- Support tuple/array in_axes for multi-parameter functions

## Technical Requirements

### 1. Update Vectorize with in_axes Parameter

```csharp
namespace MLFramework.Functional
{
    public static class Functional
    {
        // Overload for single input with explicit axis
        public static Func<Tensor, Tensor> Vectorize(
            Func<Tensor, Tensor> func,
            int axis = 0)
        {
            return Vectorize(func, new object[] { axis });
        }

        // Overload for double input with explicit axis
        public static Func<Tensor, Tensor, Tensor> Vectorize(
            Func<Tensor, Tensor, Tensor> func,
            int axis = 0)
        {
            return Vectorize(func, new object[] { axis, axis });
        }

        /// <summary>
        /// Vectorize with per-parameter axis specification.
        /// </summary>
        /// <param name="func">Function to vectorize</param>
        /// <param name="in_axes">Array of axes for each parameter. Use null for non-batched parameters.</param>
        /// <returns>Batched function</returns>
        public static Func<Tensor, Tensor> Vectorize(
            Func<Tensor, Tensor> func,
            object[] in_axes)
        {
            var transform = new VMapTransform(func, in_axes);
            return (Func<Tensor, Tensor>)transform.Transform(func);
        }

        public static Func<Tensor, Tensor, Tensor> Vectorize(
            Func<Tensor, Tensor, Tensor> func,
            object[] in_axes)
        {
            var transform = new VMapTransform(func, in_axes);
            return (Func<Tensor, Tensor, Tensor>)transform.Transform(func);
        }
    }
}
```

### 2. Enhanced VMapTransform Constructor

```csharp
public class VMapTransform : BaseTransformation
{
    private readonly int[] _axes;  // -1 means no vectorization for that param

    // New constructor accepting in_axes
    public VMapTransform(Delegate original, object[] in_axes)
        : base("vmap", TransformationType.Vectorization)
    {
        ValidateDelegate(original);

        // Normalize in_axes to int array
        var paramCount = original.Method.GetParameters().Length;
        _axes = new int[paramCount];

        if (in_axes == null)
        {
            // Default: vectorize all params on axis 0
            for (int i = 0; i < paramCount; i++)
                _axes[i] = 0;
        }
        else
        {
            if (in_axes.Length != paramCount)
            {
                throw new ArgumentException($"in_axes length ({in_axes.Length}) must match parameter count ({paramCount})");
            }

            for (int i = 0; i < paramCount; i++)
            {
                if (in_axes[i] == null)
                {
                    _axes[i] = -1;  // Don't vectorize this parameter
                }
                else if (in_axes[i] is int axis)
                {
                    _axes[i] = axis;
                }
                else
                {
                    throw new ArgumentException($"in_axes[{i}] must be int or null");
                }
            }
        }
    }

    // Legacy constructor for backward compatibility
    public VMapTransform(Delegate original, int axis = 0)
        : this(original, null)
    {
        // Set all axes to the specified axis
        for (int i = 0; i < _axes.Length; i++)
            _axes[i] = axis;
    }
```

### 3. Multi-Axis Wrapper Implementation

```csharp
private Func<Tensor, Tensor> CreateSingleInputWrapper(Func<Tensor, Tensor> original)
{
    int axis = _axes[0];

    return (Tensor input) =>
    {
        if (axis == -1)
        {
            // No vectorization needed
            return original(input);
        }

        if (input.Rank <= axis)
        {
            throw new ArgumentException($"Input tensor must have at least {axis + 1} dimensions for axis={axis}");
        }

        var batchSize = input.Shape[axis];
        var outputs = new List<Tensor>();

        for (int i = 0; i < batchSize; i++)
        {
            var singleInput = input.Take(axis, i);
            var output = original(singleInput);
            outputs.Add(output);
        }

        return Tensor.Stack(outputs, axis);
    };
}

private Func<Tensor, Tensor, Tensor> CreateDoubleInputWrapper(Func<Tensor, Tensor, Tensor> original)
{
    int axis1 = _axes[0];
    int axis2 = _axes[1];

    return (Tensor input1, Tensor input2) =>
    {
        // Determine batch dimension
        int? batchSize = null;

        if (axis1 != -1)
        {
            if (input1.Rank <= axis1)
                throw new ArgumentException($"Input1 must have at least {axis1 + 1} dimensions");
            batchSize = input1.Shape[axis1];
        }

        if (axis2 != -1)
        {
            if (input2.Rank <= axis2)
                throw new ArgumentException($"Input2 must have at least {axis2 + 1} dimensions");
            int batchSize2 = input2.Shape[axis2];
            if (batchSize.HasValue && batchSize.Value != batchSize2)
                throw new ArgumentException($"Batch dimensions must match: {batchSize.Value} != {batchSize2}");
            batchSize = batchSize ?? batchSize2;
        }

        // Handle case where no vectorization is needed
        if (batchSize == null)
        {
            return original(input1, input2);
        }

        int finalBatchSize = batchSize.Value;
        var outputs = new List<Tensor>();

        for (int i = 0; i < finalBatchSize; i++)
        {
            var singleInput1 = axis1 != -1 ? input1.Take(axis1, i) : input1;
            var singleInput2 = axis2 != -1 ? input2.Take(axis2, i) : input2;
            var output = original(singleInput1, singleInput2);
            outputs.Add(output);
        }

        // Determine output axis (first non -1 axis)
        int outputAxis = axis1 != -1 ? axis1 : axis2;
        return Tensor.Stack(outputs, outputAxis);
    };
}
```

### 4. Validation Helper

```csharp
private void ValidateInAxes()
{
    // Check that at least one axis is vectorized
    if (_axes.All(axis => axis == -1))
    {
        throw new ArgumentException("At least one parameter must be vectorized (axis != null)");
    }

    // Check that all vectorized axes are non-negative
    for (int i = 0; i < _axes.Length; i++)
    {
        if (_axes[i] < -1)
        {
            throw new ArgumentException($"Invalid axis {_axes[i]} for parameter {i}");
        }
    }
}
```

## Files to Modify
1. `src/MLFramework/Functional/VMapTransform.cs` (update existing file)
2. `src/MLFramework/Functional/Functional.cs` (add new overloads)

## Dependencies
- spec_functional_vmap_basic.md (must be completed first)

## Success Criteria
- Can specify different axes for each parameter
- Can exclude parameters from vectorization using null
- Backward compatible with single-axis Vectorize calls
- Correctly handles mixed batched/non-batched inputs

## Notes for Coder
- Maintain backward compatibility with existing code
- Use -1 internally to represent "no vectorization"
- Ensure validation is performed early with clear error messages
- Consider extending to support more than 2 parameters in future specs

## Example Usage
```csharp
// Vectorize first parameter only (parameter 2 is not batched)
var batchedFunc = Functional.Vectorize(
    (Tensor x, Tensor weight) => MatMul(x, weight),
    new object[] { 0, null }
);

// Vectorize different parameters on different axes
var multiAxisFunc = Functional.Vectorize(
    (Tensor x, Tensor y) => x + y,
    new object[] { 0, 1 }
);
```
