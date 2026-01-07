namespace MLFramework.Fusion;

/// <summary>
/// Represents a matched fusion pattern
/// </summary>
public record FusionPatternMatch
{
    /// <summary>
    /// Gets the pattern definition that was matched
    /// </summary>
    public required FusionPatternDefinition Pattern { get; init; }

    /// <summary>
    /// Gets the operations that were matched
    /// </summary>
    public required IReadOnlyList<Operation> MatchedOperations { get; init; }

    /// <summary>
    /// Gets the match score (higher = better match)
    /// </summary>
    public required int MatchScore { get; init; }
}
