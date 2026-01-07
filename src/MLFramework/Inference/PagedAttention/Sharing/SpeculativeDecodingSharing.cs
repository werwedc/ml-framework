namespace MlFramework.Inference.PagedAttention.Sharing;

/// <summary>
/// Implements block sharing for speculative decoding.
/// Speculated tokens share blocks until verification.
/// </summary>
public class SpeculativeDecodingSharing
{
    private readonly BlockShareManager _shareManager;
    private readonly int _speculationLength;
    private readonly Dictionary<int, SpeculativeState> _speculativeStates;

    public SpeculativeDecodingSharing(
        BlockShareManager shareManager,
        int speculationLength = 4)
    {
        _shareManager = shareManager;
        _speculationLength = speculationLength;
        _speculativeStates = new Dictionary<int, SpeculativeState>();
    }

    /// <summary>
    /// Allocate speculative blocks for a sequence.
    /// </summary>
    /// <param name="sequenceId">ID of the sequence</param>
    /// <param name="baseSequenceLength">Length before speculation</param>
    /// <param name="speculativeBlockIds">List of speculative block IDs</param>
    public void AllocateSpeculativeBlocks(
        int sequenceId,
        int baseSequenceLength,
        List<int> speculativeBlockIds)
    {
        lock (_shareManager)
        {
            // Track speculative state for this sequence
            _speculativeStates[sequenceId] = new SpeculativeState
            {
                BaseSequenceLength = baseSequenceLength,
                SpeculativeBlockIds = speculativeBlockIds,
                VerifiedCount = 0
            };

            // Mark these blocks as speculative (not yet verified)
            // They will be shared between main sequence and speculator
            // This is a simplified implementation
        }
    }

    /// <summary>
    /// Verify speculated tokens and update block sharing.
    /// </summary>
    /// <param name="sequenceId">ID of the sequence</param>
    /// <param name="verifiedTokens">Number of tokens that were verified</param>
    /// <returns>List of blocks to keep (verified) and free (rejected)</returns>
    public (List<int> keptBlocks, List<int> freedBlocks) VerifySpeculation(
        int sequenceId,
        int verifiedTokens)
    {
        var keptBlocks = new List<int>();
        var freedBlocks = new List<int>();

        lock (_shareManager)
        {
            if (!_speculativeStates.TryGetValue(sequenceId, out var state))
            {
                return (keptBlocks, freedBlocks);
            }

            state.VerifiedCount = verifiedTokens;

            // Keep blocks for verified tokens
            // Free blocks for rejected tokens
            // This is a simplified implementation

            // For simplicity, assume first verifiedTokens blocks are kept
            for (int i = 0; i < state.SpeculativeBlockIds.Count; i++)
            {
                if (i < verifiedTokens)
                {
                    keptBlocks.Add(state.SpeculativeBlockIds[i]);
                }
                else
                {
                    freedBlocks.Add(state.SpeculativeBlockIds[i]);
                }
            }
        }

        return (keptBlocks, freedBlocks);
    }

    /// <summary>
    /// Reject all speculative blocks.
    /// </summary>
    /// <param name="sequenceId">ID of the sequence</param>
    /// <returns>List of blocks to free</returns>
    public List<int> RejectAllSpeculation(int sequenceId)
    {
        var freedBlocks = new List<int>();

        lock (_shareManager)
        {
            if (_speculativeStates.TryGetValue(sequenceId, out var state))
            {
                freedBlocks.AddRange(state.SpeculativeBlockIds);
                _speculativeStates.Remove(sequenceId);
            }
        }

        return freedBlocks;
    }

    /// <summary>
    /// Get the speculative state for a sequence.
    /// </summary>
    public SpeculativeState? GetSpeculativeState(int sequenceId)
    {
        return _speculativeStates.TryGetValue(sequenceId, out var state) ? state : null;
    }
}
