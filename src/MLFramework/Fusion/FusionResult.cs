namespace MLFramework.Fusion;

/// <summary>
/// Result of applying fusion transformations to a graph
/// </summary>
public record FusionResult
{
    public required ComputationalGraph FusedGraph { get; init; }
    public required IReadOnlyList<FusedOperation> FusedOperations { get; init; }
    public required int OriginalOpCount { get; init; }
    public required int FusedOpCount { get; init; }
    public required IReadOnlyList<FusionRejected> RejectedFusions { get; init; }
}

/// <summary>
/// Information about a fusion that was rejected
/// </summary>
public record FusionRejected
{
    public required IReadOnlyList<Operation> Operations { get; init; }
    public required string RejectionReason { get; init; }
}

/// <summary>
/// Result of validating a fusion transformation
/// </summary>
public record FusionValidationResult
{
    public required bool IsValid { get; init; }
    public required IReadOnlyList<string> Errors { get; init; }
    public required IReadOnlyList<string> Warnings { get; init; }
}

/// <summary>
/// Options for controlling fusion behavior
/// </summary>
public record FusionOptions
{
    public bool EnableFusion { get; init; } = true;
    public int MaxFusionOps { get; init; } = 10;
    public int MinBenefitScore { get; init; } = 50;
    public bool EnableBatchNormFolding { get; init; } = true;
    public bool EnableConvActivationFusion { get; init; } = true;
    public FusionAggressiveness Aggressiveness { get; init; } = FusionAggressiveness.Medium;
}

/// <summary>
/// Aggressiveness level for fusion
/// </summary>
public enum FusionAggressiveness
{
    Conservative,
    Medium,
    Aggressive
}
