namespace MlFramework.Inference.PagedAttention.Sharing;

/// <summary>
/// State tracking for speculative decoding.
/// </summary>
public class SpeculativeState
{
    /// <summary>
    /// Base sequence length before speculation.
    /// </summary>
    public int BaseSequenceLength { get; set; }

    /// <summary>
    /// IDs of blocks allocated for speculation.
    /// </summary>
    public List<int> SpeculativeBlockIds { get; set; } = new List<int>();

    /// <summary>
    /// Number of tokens that have been verified.
    /// </summary>
    public int VerifiedCount { get; set; }
}
