namespace MLFramework.Data.Transforms;

/// <summary>
/// Generic wrapper for user-defined transform functions.
/// Allows creating custom transforms without defining a new class.
/// </summary>
/// <typeparam name="TInput">The type of input data.</typeparam>
/// <typeparam name="TOutput">The type of output data.</typeparam>
public class LambdaTransform<TInput, TOutput> : ITransform<TInput, TOutput>
{
    private readonly Func<TInput, TOutput> _func;

    /// <summary>
    /// Creates a new LambdaTransform with the specified function.
    /// </summary>
    /// <param name="func">The function to apply to input data.</param>
    /// <exception cref="ArgumentNullException">Thrown if func is null.</exception>
    public LambdaTransform(Func<TInput, TOutput> func)
    {
        _func = func ?? throw new ArgumentNullException(nameof(func));
    }

    /// <summary>
    /// Applies the lambda function to the input data.
    /// </summary>
    /// <param name="input">The input data to transform.</param>
    /// <returns>The transformed data.</returns>
    public TOutput Apply(TInput input)
    {
        return _func(input);
    }
}
