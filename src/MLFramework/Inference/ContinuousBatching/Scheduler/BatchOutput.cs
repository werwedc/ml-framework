namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Output from a model forward pass on a batch.
/// </summary>
public record class BatchOutput(
    Dictionary<RequestId, int> GeneratedTokens,
    Dictionary<RequestId, float[]> Logits,
    bool[] IsEosReached
);
