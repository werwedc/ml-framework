using RitterFramework.Core;

namespace RitterFramework.Core.Tensor;

/// <summary>
/// Extension methods for Tensor class
/// </summary>
public static class TensorExtensions
{
    /// <summary>
    /// Gets a float value from the tensor at the given flat index
    /// </summary>
    public static float GetFloat(this Tensor tensor, int index)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));
        if (index < 0 || index >= tensor.Size)
            throw new ArgumentOutOfRangeException(nameof(index), "Index is out of range");

        return tensor.Data[index];
    }

    /// <summary>
    /// Sets a float value at the given flat index
    /// </summary>
    public static void SetFloat(this Tensor tensor, int index, float value)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));
        if (index < 0 || index >= tensor.Size)
            throw new ArgumentOutOfRangeException(nameof(index), "Index is out of range");

        tensor.Data[index] = value;
    }

    /// <summary>
    /// Computes the mean of all elements in the tensor.
    /// </summary>
    /// <param name="tensor">The tensor to compute mean for.</param>
    /// <returns>A scalar tensor containing the mean.</returns>
    public static Tensor Mean(this Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var data = tensor.Data;
        float sum = 0f;
        for (int i = 0; i < tensor.Size; i++)
        {
            sum += data[i];
        }
        float mean = sum / tensor.Size;

        return new Tensor(new[] { mean }, new[] { 1 });
    }

    /// <summary>
    /// Computes the sum of all elements in the tensor.
    /// </summary>
    /// <param name="tensor">The tensor to sum.</param>
    /// <returns>A scalar tensor containing the sum.</returns>
    public static Tensor Sum(this Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var data = tensor.Data;
        float sum = 0f;
        for (int i = 0; i < tensor.Size; i++)
        {
            sum += data[i];
        }

        return new Tensor(new[] { sum }, new[] { 1 });
    }

    /// <summary>
    /// Clamps all elements in the tensor to be within the specified range.
    /// </summary>
    /// <param name="tensor">The tensor to clamp.</param>
    /// <param name="min">The minimum value.</param>
    /// <param name="max">The maximum value.</param>
    /// <returns>A new tensor with clamped values.</returns>
    public static Tensor Clamp(this Tensor tensor, float min, float max)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var data = tensor.Data;
        var result = new float[tensor.Size];
        for (int i = 0; i < tensor.Size; i++)
        {
            result[i] = Math.Max(min, Math.Min(max, data[i]));
        }

        return new Tensor(result, tensor.Shape, tensor.RequiresGrad);
    }

    /// <summary>
    /// Element-wise subtraction of two tensors.
    /// </summary>
    /// <param name="tensorA">The first tensor.</param>
    /// <param name="tensorB">The second tensor.</param>
    /// <returns>A new tensor containing the element-wise difference.</returns>
    public static Tensor Subtract(this Tensor tensorA, Tensor tensorB)
    {
        if (tensorA == null)
            throw new ArgumentNullException(nameof(tensorA));
        if (tensorB == null)
            throw new ArgumentNullException(nameof(tensorB));

        if (!tensorA.HasSameShape(tensorB))
            throw new ArgumentException("Tensors must have the same shape for subtraction");

        var result = new float[tensorA.Size];
        for (int i = 0; i < tensorA.Size; i++)
        {
            result[i] = tensorA.Data[i] - tensorB.Data[i];
        }

        return new Tensor(result, tensorA.Shape, tensorA.RequiresGrad || tensorB.RequiresGrad);
    }

    /// <summary>
    /// Element-wise multiplication of two tensors.
    /// </summary>
    /// <param name="tensorA">The first tensor.</param>
    /// <param name="tensorB">The second tensor.</param>
    /// <returns>A new tensor containing the element-wise product.</returns>
    public static Tensor Multiply(this Tensor tensorA, Tensor tensorB)
    {
        if (tensorA == null)
            throw new ArgumentNullException(nameof(tensorA));
        if (tensorB == null)
            throw new ArgumentNullException(nameof(tensorB));

        if (!tensorA.HasSameShape(tensorB))
            throw new ArgumentException("Tensors must have the same shape for multiplication");

        var result = new float[tensorA.Size];
        for (int i = 0; i < tensorA.Size; i++)
        {
            result[i] = tensorA.Data[i] * tensorB.Data[i];
        }

        return new Tensor(result, tensorA.Shape, tensorA.RequiresGrad || tensorB.RequiresGrad);
    }

    /// <summary>
    /// Element-wise division of two tensors.
    /// </summary>
    /// <param name="tensorA">The numerator tensor.</param>
    /// <param name="tensorB">The denominator tensor.</param>
    /// <returns>A new tensor containing the element-wise quotient.</returns>
    public static Tensor Divide(this Tensor tensorA, Tensor tensorB)
    {
        if (tensorA == null)
            throw new ArgumentNullException(nameof(tensorA));
        if (tensorB == null)
            throw new ArgumentNullException(nameof(tensorB));

        if (!tensorA.HasSameShape(tensorB))
            throw new ArgumentException("Tensors must have the same shape for division");

        var result = new float[tensorA.Size];
        for (int i = 0; i < tensorA.Size; i++)
        {
            result[i] = tensorA.Data[i] / tensorB.Data[i];
        }

        return new Tensor(result, tensorA.Shape, tensorA.RequiresGrad || tensorB.RequiresGrad);
    }

    /// <summary>
    /// Divides a tensor by a scalar.
    /// </summary>
    /// <param name="tensor">The tensor to divide.</param>
    /// <param name="scalar">The scalar divisor.</param>
    /// <returns>A new tensor with each element divided by the scalar.</returns>
    public static Tensor Divide(this Tensor tensor, float scalar)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var result = new float[tensor.Size];
        for (int i = 0; i < tensor.Size; i++)
        {
            result[i] = tensor.Data[i] / scalar;
        }

        return new Tensor(result, tensor.Shape, tensor.RequiresGrad);
    }

    /// <summary>
    /// Creates a tensor with element-wise comparison: result[i] = 1 if tensorA[i] >= value, else 0.
    /// </summary>
    /// <param name="tensor">The tensor to compare.</param>
    /// <param name="value">The value to compare against.</param>
    /// <returns>A new tensor with 1.0f or 0.0f values.</returns>
    public static Tensor GreaterThanOrEqual(this Tensor tensor, float value)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var result = new float[tensor.Size];
        for (int i = 0; i < tensor.Size; i++)
        {
            result[i] = tensor.Data[i] >= value ? 1.0f : 0.0f;
        }

        return new Tensor(result, tensor.Shape, false);
    }

    /// <summary>
    /// Creates a tensor with element-wise comparison: result[i] = 1 if tensorA[i] <= value, else 0.
    /// </summary>
    /// <param name="tensor">The tensor to compare.</param>
    /// <param name="value">The value to compare against.</param>
    /// <returns>A new tensor with 1.0f or 0.0f values.</returns>
    public static Tensor LessThanOrEqual(this Tensor tensor, float value)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var result = new float[tensor.Size];
        for (int i = 0; i < tensor.Size; i++)
        {
            result[i] = tensor.Data[i] <= value ? 1.0f : 0.0f;
        }

        return new Tensor(result, tensor.Shape, false);
    }

    /// <summary>
    /// Element-wise logical AND of two tensors (treated as boolean masks).
    /// </summary>
    /// <param name="tensorA">The first tensor.</param>
    /// <param name="tensorB">The second tensor.</param>
    /// <returns>A new tensor with 1.0f where both inputs are non-zero, else 0.0f.</returns>
    public static Tensor And(this Tensor tensorA, Tensor tensorB)
    {
        if (tensorA == null)
            throw new ArgumentNullException(nameof(tensorA));
        if (tensorB == null)
            throw new ArgumentNullException(nameof(tensorB));

        if (!tensorA.HasSameShape(tensorB))
            throw new ArgumentException("Tensors must have the same shape for AND operation");

        var result = new float[tensorA.Size];
        for (int i = 0; i < tensorA.Size; i++)
        {
            result[i] = (tensorA.Data[i] != 0.0f && tensorB.Data[i] != 0.0f) ? 1.0f : 0.0f;
        }

        return new Tensor(result, tensorA.Shape, false);
    }

    /// <summary>
    /// Negates a tensor element-wise.
    /// </summary>
    /// <param name="tensor">The tensor to negate.</param>
    /// <returns>A new tensor with all elements negated.</returns>
    public static Tensor Negate(this Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var result = new float[tensor.Size];
        for (int i = 0; i < tensor.Size; i++)
        {
            result[i] = -tensor.Data[i];
        }

        return new Tensor(result, tensor.Shape, tensor.RequiresGrad);
    }

    /// <summary>
    /// Gets the number of elements in the tensor.
    /// </summary>
    /// <param name="tensor">The tensor.</param>
    /// <returns>The total number of elements.</returns>
    public static int NumberOfElements(this Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        return tensor.Size;
    }

    /// <summary>
    /// Multiplies all elements of a tensor by a scalar.
    /// </summary>
    /// <param name="tensor">The tensor to multiply.</param>
    /// <param name="scalar">The scalar multiplier.</param>
    /// <returns>A new tensor with each element multiplied by the scalar.</returns>
    public static Tensor MultiplyScalar(this Tensor tensor, float scalar)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var result = new float[tensor.Size];
        for (int i = 0; i < tensor.Size; i++)
        {
            result[i] = tensor.Data[i] * scalar;
        }

        return new Tensor(result, tensor.Shape, tensor.RequiresGrad);
    }

    /// <summary>
    /// Creates a scalar tensor from a float value.
    /// </summary>
    /// <param name="value">The scalar value.</param>
    /// <param name="requiresGrad">Whether to enable gradient tracking (default: false).</param>
    /// <returns>A scalar tensor.</returns>
    public static Tensor FromScalar(float value, bool requiresGrad = false)
    {
        return new Tensor(new[] { value }, new[] { 1 }, requiresGrad);
    }
}
