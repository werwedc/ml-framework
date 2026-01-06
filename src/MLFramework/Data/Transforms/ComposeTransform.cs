namespace MLFramework.Data.Transforms;

/// <summary>
/// Transform that chains multiple transforms together in sequence.
/// The output of one transform becomes the input to the next.
/// </summary>
public class ComposeTransform : ITransform
{
    private readonly ITransform[] _transforms;

    /// <summary>
    /// Creates a new ComposeTransform that chains the specified transforms in order.
    /// </summary>
    /// <param name="transforms">The transforms to chain. Must contain at least one transform.</param>
    /// <exception cref="ArgumentNullException">Thrown if transforms array is null.</exception>
    /// <exception cref="ArgumentException">Thrown if transforms array is empty or contains null elements.</exception>
    public ComposeTransform(params ITransform[] transforms)
    {
        if (transforms == null)
            throw new ArgumentNullException(nameof(transforms));

        if (transforms.Length == 0)
            throw new ArgumentException("At least one transform is required", nameof(transforms));

        for (int i = 0; i < transforms.Length; i++)
        {
            if (transforms[i] == null)
                throw new ArgumentNullException($"Transform at index {i} cannot be null");
        }

        _transforms = transforms;
    }

    /// <summary>
    /// Applies all transforms in sequence to the input data.
    /// </summary>
    /// <param name="input">The input data to transform.</param>
    /// <returns>The transformed data after applying all transforms in sequence.</returns>
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

/// <summary>
/// Type-safe transform that chains multiple transforms together in sequence.
/// The output of one transform becomes the input to the next.
/// </summary>
/// <typeparam name="TInput">The type of input data.</typeparam>
/// <typeparam name="TOutput">The type of output data.</typeparam>
public class ComposeTransform<TInput, TOutput> : ITransform<TInput, TOutput>
{
    private readonly Func<TInput, TOutput> _composedFunc;

    /// <summary>
    /// Creates a new ComposeTransform that chains the specified transforms in order.
    /// </summary>
    /// <param name="transforms">The transforms to chain. Must contain at least one transform.</param>
    /// <exception cref="ArgumentNullException">Thrown if transforms array is null.</exception>
    /// <exception cref="ArgumentException">Thrown if transforms array is empty or contains null elements.</exception>
    public ComposeTransform(params ITransform[] transforms)
    {
        if (transforms == null)
            throw new ArgumentNullException(nameof(transforms));

        if (transforms.Length == 0)
            throw new ArgumentException("At least one transform is required", nameof(transforms));

        for (int i = 0; i < transforms.Length; i++)
        {
            if (transforms[i] == null)
                throw new ArgumentNullException($"Transform at index {i} cannot be null");
        }

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

    /// <summary>
    /// Applies all transforms in sequence to the input data.
    /// </summary>
    /// <param name="input">The input data to transform.</param>
    /// <returns>The transformed data after applying all transforms in sequence.</returns>
    public TOutput Apply(TInput input)
    {
        return _composedFunc(input);
    }
}
