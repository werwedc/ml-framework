using System;
using System.Collections.Generic;

namespace MLFramework.Autograd.Exceptions;

/// <summary>
/// Exception thrown when a gradient tensor contains NaN (Not a Number) values.
/// </summary>
public class GradientNaNException : InvalidOperationException
{
    /// <summary>
    /// Gets the name of the parameter that contains NaN values.
    /// </summary>
    public string ParameterName { get; }

    /// <summary>
    /// Gets the flattened index of the first NaN value in the tensor data.
    /// </summary>
    public int NaNIndex { get; }

    /// <summary>
    /// Gets the multi-dimensional indices of the first NaN value.
    /// </summary>
    public IReadOnlyList<int> NaNLocation { get; }

    public GradientNaNException(
        string parameterName,
        int nanIndex)
        : base($"Gradient for parameter '{parameterName}' contains NaN values (first at index {nanIndex})")
    {
        ParameterName = parameterName ?? throw new ArgumentNullException(nameof(parameterName));
        NaNIndex = nanIndex;
        NaNLocation = new List<int>();
    }

    public GradientNaNException(
        string parameterName,
        int nanIndex,
        IReadOnlyList<int> nanLocation)
        : base($"Gradient for parameter '{parameterName}' contains NaN values at location [{string.Join(", ", nanLocation)}]")
    {
        ParameterName = parameterName ?? throw new ArgumentNullException(nameof(parameterName));
        NaNIndex = nanIndex;
        NaNLocation = nanLocation ?? throw new ArgumentNullException(nameof(nanLocation));
    }

    public GradientNaNException(
        string parameterName,
        int nanIndex,
        Exception innerException)
        : base($"Gradient for parameter '{parameterName}' contains NaN values (first at index {nanIndex})", innerException)
    {
        ParameterName = parameterName ?? throw new ArgumentNullException(nameof(parameterName));
        NaNIndex = nanIndex;
        NaNLocation = new List<int>();
    }
}
