namespace MlFramework.Inference.PagedAttention.Phases;

/// <summary>
/// Phase of a sequence in the inference pipeline.
/// </summary>
public enum SequencePhase
{
    /// <summary>
    /// Currently processing prompt tokens.
    /// </summary>
    Prefill,

    /// <summary>
    /// Currently generating tokens one at a time.
    /// </summary>
    Decode
}
