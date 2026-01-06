namespace MLFramework.Data;

/// <summary>
/// Generic interface for type-safe data transforms.
/// </summary>
/// <typeparam name="TInput">The type of input data.</typeparam>
/// <typeparam name="TOutput">The type of output data.</typeparam>
public interface ITransform<TInput, TOutput>
{
    /// <summary>
    /// Applies the transform to the input data.
    /// </summary>
    /// <param name="input">The input data to transform.</param>
    /// <returns>The transformed data.</returns>
    TOutput Apply(TInput input);
}
