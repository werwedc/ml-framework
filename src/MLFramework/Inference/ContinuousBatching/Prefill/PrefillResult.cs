namespace MLFramework.Inference.ContinuousBatching.Prefill;

/// <summary>
/// Result of single request prefill.
/// </summary>
public record class PrefillResult(
    RequestId RequestId,
    bool Success,
    int[] PromptTokens,
    float[] InitialLogits,
    int ProcessedTokens,
    long MemoryBytesUsed,
    TimeSpan ProcessingTime,
    string? ErrorMessage
);
