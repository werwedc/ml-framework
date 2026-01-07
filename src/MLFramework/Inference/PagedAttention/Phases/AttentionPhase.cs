namespace MlFramework.Inference.PagedAttention.Phases;

/// <summary>
/// Execution phase for attention computation.
/// </summary>
public enum AttentionPhase
{
    /// <summary>
    /// Prefill phase: processing prompt tokens in parallel.
    /// </summary>
    Prefill,

    /// <summary>
    /// Decode phase: processing generated tokens one at a time.
    /// </summary>
    Decode
}
