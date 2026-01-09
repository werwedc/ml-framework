using System;
using RitterFramework.Core;

namespace MLFramework.Autograd.Exceptions;

/// <summary>
/// Exception thrown when a gradient tensor has a different data type than its corresponding input tensor.
/// </summary>
public class GradientTypeException : InvalidOperationException
{
    /// <summary>
    /// Gets the expected data type of the gradient tensor.
    /// </summary>
    public DataType ExpectedDtype { get; }

    /// <summary>
    /// Gets the actual data type of the gradient tensor.
    /// </summary>
    public DataType ActualDtype { get; }

    /// <summary>
    /// Gets the name of the parameter that caused the type mismatch.
    /// </summary>
    public string ParameterName { get; }

    public GradientTypeException(
        DataType expectedDtype,
        DataType actualDtype,
        string parameterName)
        : base($"Gradient dtype {actualDtype} does not match input dtype {expectedDtype} for parameter '{parameterName}'")
    {
        ExpectedDtype = expectedDtype;
        ActualDtype = actualDtype;
        ParameterName = parameterName ?? throw new ArgumentNullException(nameof(parameterName));
    }

    public GradientTypeException(
        DataType expectedDtype,
        DataType actualDtype,
        string parameterName,
        Exception innerException)
        : base($"Gradient dtype {actualDtype} does not match input dtype {expectedDtype} for parameter '{parameterName}'", innerException)
    {
        ExpectedDtype = expectedDtype;
        ActualDtype = actualDtype;
        ParameterName = parameterName ?? throw new ArgumentNullException(nameof(parameterName));
    }
}
