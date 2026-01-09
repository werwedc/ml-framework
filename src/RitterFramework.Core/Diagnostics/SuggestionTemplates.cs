namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Template strings for generating actionable fix suggestions for shape mismatches.
/// </summary>
public static class SuggestionTemplates
{
    /// <summary>
    /// Template for missing batch dimension.
    /// Args: [0] - Current shape as string
    /// </summary>
    public const string MissingBatchDim = "Input missing batch dimension. Add unsqueeze(0) or reshape to [1, {0}]";

    /// <summary>
    /// Template for channel order mismatch.
    /// Args: [0] - Current format (e.g., "NHWC"), [1] - Expected format (e.g., "NCHW")
    /// </summary>
    public const string ChannelOrderMismatch = "Channel order mismatch. Permute from {0} to {1}";

    /// <summary>
    /// Template for feature size mismatch in linear layer.
    /// Args: [0] - Actual feature count, [1] - Expected feature count
    /// </summary>
    public const string FeatureSizeMismatch = "Previous layer outputs {0} features, but this layer expects {1}. Adjust layer configuration";

    /// <summary>
    /// Template for transpose requirement.
    /// Args: [0] - Current shape description, [1] - Expected shape description
    /// </summary>
    public const string TransposeRequired = "Consider transposing {0} to {1}";

    /// <summary>
    /// Template for concatenation dimension mismatch.
    /// Args: [0] - Axis with mismatch, [1] - Suggested axis
    /// </summary>
    public const string ConcatenationDimensionMismatch = "Cannot concatenate on axis {0} with different sizes. Use axis {1} or reshape inputs";

    /// <summary>
    /// Template for broadcasting failure.
    /// Args: [0] - Shape 1, [1] - Shape 2
    /// </summary>
    public const string BroadcastingFailure = "Cannot broadcast shapes {0} and {1}. Batch sizes must match or be 1";

    /// <summary>
    /// Template for generic shape mismatch.
    /// Args: [0] - Operation type, [1] - Current shape, [2] - Expected shape
    /// </summary>
    public const string GenericShapeMismatch = "Shape mismatch for {0}: {1} vs expected {2}";

    /// <summary>
    /// Template for reshape suggestion.
    /// Args: [0] - Current shape, [1] - Suggested shape
    /// </summary>
    public const string ReshapeSuggestion = "Try reshaping from {0} to {1}";

    /// <summary>
    /// Template for squeeze operation.
    /// Args: [0] - Dimension to squeeze
    /// </summary>
    public const string SqueezeSuggestion = "Consider squeezing dimension {0} to remove size-1 dimension";

    /// <summary>
    /// Template for unsqueeze operation.
    /// Args: [0] - Dimension to unsqueeze
    /// </summary>
    public const string UnsqueezeSuggestion = "Consider unsqueezing at dimension {0} to add size-1 dimension";
}
