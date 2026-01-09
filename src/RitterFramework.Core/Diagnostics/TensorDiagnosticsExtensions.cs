namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Extension methods for Tensor diagnostics.
/// </summary>
public static class TensorDiagnosticsExtensions
{
    /// <summary>
    /// Gets the shape of a tensor as a formatted string.
    /// </summary>
    public static string GetShapeString(this global::RitterFramework.Core.Tensor.Tensor? tensor)
    {
        if (tensor == null)
        {
            return "null";
        }

        return $"[{string.Join(", ", tensor.Shape)}]";
    }

    /// <summary>
    /// Gets the total element count of a tensor.
    /// </summary>
    public static long GetElementCount(this global::RitterFramework.Core.Tensor.Tensor? tensor)
    {
        if (tensor == null)
        {
            return 0;
        }

        long count = 1;
        foreach (var dim in tensor.Shape)
        {
            count *= dim;
        }
        return count;
    }
}
