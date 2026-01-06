using System;
using System.Linq;
using System.Reflection;
using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Internal helper class for accessing private members of the Tensor class.
/// This is necessary because the Tensor class has private fields that we need for higher-order derivative computation.
/// </summary>
internal static class TensorAccessor
{
    private static readonly FieldInfo DataField = typeof(Tensor).GetField("_data", BindingFlags.NonPublic | BindingFlags.Instance)!;
    private static readonly MethodInfo BackwardMethod = typeof(Tensor).GetMethod("Backward", new[] { typeof(Tensor) })!;

    /// <summary>
    /// Gets the underlying data array of a tensor.
    /// </summary>
    public static float[] GetData(Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        return (float[])DataField.GetValue(tensor)!;
    }

    /// <summary>
    /// Sets the underlying data array of a tensor.
    /// </summary>
    public static void SetData(Tensor tensor, float[] data)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));
        if (data == null)
            throw new ArgumentNullException(nameof(data));

        DataField.SetValue(tensor, data);
    }

    /// <summary>
    /// Creates a clone of a tensor with gradient tracking enabled.
    /// </summary>
    public static Tensor CloneWithGrad(Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var data = (float[])GetData(tensor).Clone();
        var shape = (int[])tensor.Shape.Clone();
        return new Tensor(data, shape, requiresGrad: true);
    }

    /// <summary>
    /// Creates a clone of a tensor without gradient tracking.
    /// </summary>
    public static Tensor CloneWithoutGrad(Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var data = (float[])GetData(tensor).Clone();
        var shape = (int[])tensor.Shape.Clone();
        return new Tensor(data, shape, requiresGrad: false);
    }

    /// <summary>
    /// Creates a scalar tensor from a double value.
    /// </summary>
    public static Tensor CreateScalar(double value)
    {
        return new Tensor(new[] { (float)value }, new[] { 1 }, requiresGrad: true);
    }
}
