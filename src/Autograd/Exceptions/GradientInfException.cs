using System;
using System.Collections.Generic;

namespace MLFramework.Autograd.Exceptions;

/// <summary>
/// Exception thrown when a gradient tensor contains infinite values.
/// </summary>
public class GradientInfException : InvalidOperationException
{
    /// <summary>
    /// Gets the name of the parameter that contains infinite values.
    /// </summary>
    public string ParameterName { get; }

    /// <summary>
    /// Gets the flattened index of the first infinite value in the tensor data.
    /// </summary>
    public int InfIndex { get; }

    /// <summary>
    /// Gets whether the infinity is positive (true) or negative (false).
    /// </summary>
    public bool IsPositiveInfinity { get; }

    /// <summary>
    /// Gets the multi-dimensional indices of the first infinite value.
    /// </summary>
    public IReadOnlyList<int> InfLocation { get; }

    public GradientInfException(
        string parameterName,
        int infIndex,
        bool isPositiveInfinity)
        : base($"Gradient for parameter '{parameterName}' contains {(isPositiveInfinity ? "positive" : "negative")} infinity values (first at index {infIndex})")
    {
        ParameterName = parameterName ?? throw new ArgumentNullException(nameof(parameterName));
        InfIndex = infIndex;
        IsPositiveInfinity = isPositiveInfinity;
        InfLocation = new List<int>();
    }

    public GradientInfException(
        string parameterName,
        int infIndex,
        bool isPositiveInfinity,
        IReadOnlyList<int> infLocation)
        : base($"Gradient for parameter '{parameterName}' contains {(isPositiveInfinity ? "positive" : "negative")} infinity values at location [{string.Join(", ", infLocation)}]")
    {
        ParameterName = parameterName ?? throw new ArgumentNullException(nameof(parameterName));
        InfIndex = infIndex;
        IsPositiveInfinity = isPositiveInfinity;
        InfLocation = infLocation ?? throw new ArgumentNullException(nameof(infLocation));
    }

    public GradientInfException(
        string parameterName,
        int infIndex,
        bool isPositiveInfinity,
        Exception innerException)
        : base($"Gradient for parameter '{parameterName}' contains {(isPositiveInfinity ? "positive" : "negative")} infinity values (first at index {infIndex})", innerException)
    {
        ParameterName = parameterName ?? throw new ArgumentNullException(nameof(parameterName));
        InfIndex = infIndex;
        IsPositiveInfinity = isPositiveInfinity;
        InfLocation = new List<int>();
    }
}
