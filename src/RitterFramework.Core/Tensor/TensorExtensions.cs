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
}
