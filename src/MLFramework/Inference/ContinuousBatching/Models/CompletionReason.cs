namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Enumerate reasons why a request completed.
/// </summary>
public enum CompletionReason
{
    /// <summary>
    /// End-of-sequence token generated.
    /// </summary>
    EosTokenReached,

    /// <summary>
    /// Max generation limit reached.
    /// </summary>
    MaxTokensReached,

    /// <summary>
    /// Request cancelled by client.
    /// </summary>
    Cancelled,

    /// <summary>
    /// Custom length condition met.
    /// </summary>
    LengthReached,

    /// <summary>
    /// Stop string condition met.
    /// </summary>
    StopString,

    /// <summary>
    /// Request timed out.
    /// </summary>
    Timeout
}
