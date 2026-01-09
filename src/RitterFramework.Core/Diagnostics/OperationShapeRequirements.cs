namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Defines the shape requirements for an operation type.
/// </summary>
public class OperationShapeRequirements
{
    /// <summary>
    /// Number of input tensors required.
    /// </summary>
    public int InputCount { get; set; }

    /// <summary>
    /// Expected dimension count for each input.
    /// e.g., new[] { 2, 2 } for matrix multiply (2D x 2D)
    /// </summary>
    public int[] ExpectedDimensions { get; set; }

    /// <summary>
    /// Dimension constraints: key=input_index, value=(dimension_index, constraint)
    /// e.g., { [0, 1]: (1, "must_match", [1, 0]) }
    /// Means: input[0].dim[1] must match input[1].dim[0]
    /// </summary>
    public Dictionary<int, Dictionary<int, DimensionConstraint>> DimensionConstraints { get; set; }

    /// <summary>
    /// Human-readable description of shape requirements.
    /// </summary>
    public string Description { get; set; }

    /// <summary>
    /// Format string for error messages.
    /// </summary>
    public string ErrorMessageFormat { get; set; }

    /// <summary>
    /// Optional: Custom validation logic.
    /// </summary>
    public Func<IEnumerable<long[]>, IDictionary<string, object>, ValidationResult> CustomValidator { get; set; }
}
