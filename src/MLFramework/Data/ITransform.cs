namespace MLFramework.Data;

/// <summary>
/// Base interface for data transforms.
/// </summary>
public interface ITransform
{
    /// <summary>
    /// Applies the transform to the input data.
    /// </summary>
    /// <param name="input">The input data to transform.</param>
    /// <returns>The transformed data.</returns>
    object Apply(object input);
}
