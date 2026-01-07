using MlFramework.Inference.PagedAttention.Kernels;

namespace MlFramework.Inference.PagedAttention.Phases;

/// <summary>
/// Manages transitions between prefill and decode phases.
/// </summary>
public class PhaseManager
{
    private readonly PrefillOrchestrator _prefillOrchestrator;
    private readonly DecodeOrchestrator _decodeOrchestrator;
    private readonly Dictionary<int, SequencePhase> _sequencePhases;

    public PhaseManager(
        PrefillOrchestrator prefillOrchestrator,
        DecodeOrchestrator decodeOrchestrator)
    {
        _prefillOrchestrator = prefillOrchestrator;
        _decodeOrchestrator = decodeOrchestrator;
        _sequencePhases = new Dictionary<int, SequencePhase>();
    }

    /// <summary>
    /// Get the current phase for a sequence.
    /// </summary>
    public SequencePhase GetPhase(int sequenceId)
    {
        return _sequencePhases.TryGetValue(sequenceId, out var phase)
            ? phase
            : SequencePhase.Prefill;
    }

    /// <summary>
    /// Set the phase for a sequence.
    /// </summary>
    public void SetPhase(int sequenceId, SequencePhase phase)
    {
        _sequencePhases[sequenceId] = phase;
    }

    /// <summary>
    /// Transition a sequence from prefill to decode.
    /// </summary>
    public void TransitionToDecode(int sequenceId)
    {
        if (_sequencePhases.ContainsKey(sequenceId))
        {
            _sequencePhases[sequenceId] = SequencePhase.Decode;
        }
    }

    /// <summary>
    /// Check if a sequence is in prefill phase.
    /// </summary>
    public bool IsPrefilling(int sequenceId)
    {
        return GetPhase(sequenceId) == SequencePhase.Prefill;
    }

    /// <summary>
    /// Check if a sequence is in decode phase.
    /// </summary>
    public bool IsDecoding(int sequenceId)
    {
        return GetPhase(sequenceId) == SequencePhase.Decode;
    }

    /// <summary>
    /// Remove a sequence from phase tracking.
    /// </summary>
    public void RemoveSequence(int sequenceId)
    {
        _sequencePhases.Remove(sequenceId);
    }

    /// <summary>
    /// Get the prefill orchestrator.
    /// </summary>
    public PrefillOrchestrator PrefillOrchestrator => _prefillOrchestrator;

    /// <summary>
    /// Get the decode orchestrator.
    /// </summary>
    public DecodeOrchestrator DecodeOrchestrator => _decodeOrchestrator;
}
