using System;
using System.Collections.Generic;

namespace MLFramework.Autograd.Exceptions;

/// <summary>
/// Exception thrown when a gradient tensor has an incompatible shape with its corresponding input tensor.
/// </summary>
public class GradientShapeException : InvalidOperationException
{
    /// <summary>
    /// Gets the expected shape of the gradient tensor.
    /// </summary>
    public IReadOnlyList<int> ExpectedShape { get; }

    /// <summary>
    /// Gets the actual shape of the gradient tensor.
    /// </summary>
    public IReadOnlyList<int> ActualShape { get; }

    /// <summary>
    /// Gets the name of the parameter that caused the shape mismatch.
    /// </summary>
    public string ParameterName { get; }

    public GradientShapeException(
        IReadOnlyList<int> expectedShape,
        IReadOnlyList<int> actualShape,
        string parameterName)
        : base($"Gradient shape [{string.Join(", ", actualShape)}] does not match input shape [{string.Join(", ", expectedShape)}] for parameter '{parameterName}'")
    {
        ExpectedShape = expectedShape ?? throw new ArgumentNullException(nameof(expectedShape));
        ActualShape = actualShape ?? throw new ArgumentNullException(nameof(actualShape));
        ParameterName = parameterName ?? throw new ArgumentNullException(nameof(parameterName));
    }

    public GradientShapeException(
        IReadOnlyList<int> expectedShape,
        IReadOnlyList<int> actualShape,
        string parameterName,
        Exception innerException)
        : base($"Gradient shape [{string.Join(", ", actualShape)}] does not match input shape [{string.Join(", ", expectedShape)}] for parameter '{parameterName}'", innerException)
    {
        ExpectedShape = expectedShape ?? throw new ArgumentNullException(nameof(expectedShape));
        ActualShape = actualShape ?? throw new ArgumentNullException(nameof(actualShape));
        ParameterName = parameterName ?? throw new ArgumentNullException(nameof(parameterName));
    }
}
