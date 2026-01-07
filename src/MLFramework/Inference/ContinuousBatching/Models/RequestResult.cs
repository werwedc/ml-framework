namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Structured result of a completed request.
/// </summary>
public record class RequestResult(
    RequestId RequestId,
    string GeneratedText,
    int TokensGenerated,
    CompletionReason Reason,
    TimeSpan ProcessingTime
);
