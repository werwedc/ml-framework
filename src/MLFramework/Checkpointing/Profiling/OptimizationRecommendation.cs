namespace MLFramework.Checkpointing.Profiling;

/// <summary>
/// Optimization recommendation
/// </summary>
public class OptimizationRecommendation
{
    /// <summary>
    /// Type of recommendation
    /// </summary>
    public RecommendationType Type { get; set; }

    /// <summary>
    /// Priority of the recommendation
    /// </summary>
    public RecommendationPriority Priority { get; set; }

    /// <summary>
    /// Title of the recommendation
    /// </summary>
    public string Title { get; set; } = string.Empty;

    /// <summary>
    /// Detailed description
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Affected layer ID (if applicable)
    /// </summary>
    public string? AffectedLayerId { get; set; }

    /// <summary>
    /// Expected impact
    /// </summary>
    public string ExpectedImpact { get; set; } = string.Empty;

    /// <summary>
    /// Creates a string representation
    /// </summary>
    /// <returns>String representation</returns>
    public override string ToString()
    {
        return $"[{Priority}] {Title}: {Description}";
    }
}
