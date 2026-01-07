namespace MLFramework.Fusion;

/// <summary>
/// Defines a fusion pattern
/// </summary>
public record FusionPatternDefinition
{
    /// <summary>
    /// Gets the pattern name
    /// </summary>
    public required string Name { get; init; }

    /// <summary>
    /// Gets the sequence of operation types that define this pattern
    /// </summary>
    public required IReadOnlyList<string> OpTypeSequence { get; init; }

    /// <summary>
    /// Gets the matching strategy delegate
    /// </summary>
    public required PatternMatchDelegate MatchStrategy { get; init; }

    /// <summary>
    /// Gets or sets the fusion strategy
    /// </summary>
    public FusionStrategy Strategy { get; init; } = FusionStrategy.Merge;

    /// <summary>
    /// Gets or sets the priority (higher = evaluated first)
    /// </summary>
    public int Priority { get; init; } = 0;
}
