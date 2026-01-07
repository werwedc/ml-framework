namespace MLFramework.Fusion;

/// <summary>
/// Statistics for fusion operations
/// </summary>
public record FusionStatistics
{
    public required int TotalOperations { get; init; }
    public required int FusedOperations { get; init; }
    public required int FusedGroups { get; init; }
    public required double FusionPercentage { get; init; }
    public required int RejectedFusions { get; init; }
    public required double AverageOperationsPerFusedGroup { get; init; }
    public required IReadOnlyDictionary<FusionPatternType, int> PatternCounts { get; init; }
    public required IReadOnlyDictionary<string, int> RejectionReasons { get; init; }
    public required DateTime StartTime { get; init; }
    public required DateTime EndTime { get; init; }
}
