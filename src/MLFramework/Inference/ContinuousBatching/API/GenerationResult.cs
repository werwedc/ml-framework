namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Result of a generation request.
/// </summary>
public record class GenerationResult(
    RequestId RequestId,
    string GeneratedText,
    int TokensGenerated,
    CompletionReason Reason,
    TimeSpan ProcessingTime,
    TimeSpan QueueTime,
    Dictionary<string, object>? Metadata
);
