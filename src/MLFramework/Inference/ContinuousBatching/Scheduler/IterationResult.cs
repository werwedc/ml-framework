namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Result of a single scheduler iteration.
/// </summary>
public record class IterationResult(
    int IterationNumber,
    int RequestCount,
    int TokensGenerated,
    int RequestsCompleted,
    TimeSpan ProcessingTime,
    long MemoryBytesUsed
);
