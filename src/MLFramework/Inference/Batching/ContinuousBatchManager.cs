using MlFramework.Inference.PagedAttention;
using MlFramework.Inference.PagedAttention.Phases;
using MlFramework.Inference.Scheduling;

namespace MlFramework.Inference.Batching;

/// <summary>
/// Manages continuous batching for PagedAttention inference.
/// Dynamically adds/removes sequences from batches to maximize throughput.
/// </summary>
public class ContinuousBatchManager
{
    private readonly KVCacheBlockManager _blockManager;
    private readonly BlockTable _blockTable;
    private readonly PhaseManager _phaseManager;
    private readonly PagedAttentionScheduler _scheduler;
    private readonly List<int> _activeSequences;
    private readonly int _maxBatchSize;
    private readonly object _lock = new object();

    public ContinuousBatchManager(
        KVCacheBlockManager blockManager,
        BlockTable blockTable,
        PhaseManager phaseManager,
        PagedAttentionScheduler scheduler,
        int maxBatchSize = 32)
    {
        _blockManager = blockManager;
        _blockTable = blockTable;
        _phaseManager = phaseManager;
        _scheduler = scheduler;
        _maxBatchSize = maxBatchSize;
        _activeSequences = new List<int>();
    }

    /// <summary>
    /// Try to add a new sequence to the current batch.
    /// </summary>
    /// <param name="sequenceId">ID of the sequence to add</param>
    /// <returns>True if added, false if batch is full or insufficient memory</returns>
    public bool TryAddSequence(int sequenceId)
    {
        lock (_lock)
        {
            // Check if batch is full
            if (_activeSequences.Count >= _maxBatchSize)
            {
                return false;
            }

            // Check if sequence is already active
            if (_activeSequences.Contains(sequenceId))
            {
                return true;
            }

            // Check if we have memory for this sequence
            var stats = _blockManager.GetStats();
            if (stats.FreeBlocks == 0)
            {
                return false;
            }

            _activeSequences.Add(sequenceId);
            return true;
        }
    }

    /// <summary>
    /// Remove a sequence from the batch and free its resources.
    /// </summary>
    public void RemoveSequence(int sequenceId)
    {
        lock (_lock)
        {
            _activeSequences.Remove(sequenceId);

            // Free blocks and update block table
            _blockManager.FreeSequenceBlocks(sequenceId);
            _blockTable.RemoveSequence(sequenceId);
            _phaseManager.RemoveSequence(sequenceId);
        }
    }

    /// <summary>
    /// Get the current batch of active sequences.
    /// </summary>
    public List<int> GetCurrentBatch()
    {
        lock (_lock)
        {
            return new List<int>(_activeSequences);
        }
    }

    /// <summary>
    /// Get sequences in prefill phase.
    /// </summary>
    public List<int> GetPrefillSequences()
    {
        lock (_lock)
        {
            return _activeSequences
                .Where(id => _phaseManager.IsPrefilling(id))
                .ToList();
        }
    }

    /// <summary>
    /// Get sequences in decode phase.
    /// </summary>
    public List<int> GetDecodeSequences()
    {
        lock (_lock)
        {
            return _activeSequences
                .Where(id => _phaseManager.IsDecoding(id))
                .ToList();
        }
    }

    /// <summary>
    /// Mark a sequence as ready for decode phase.
    /// </summary>
    public void TransitionToDecode(int sequenceId)
    {
        lock (_lock)
        {
            if (_activeSequences.Contains(sequenceId))
            {
                _phaseManager.TransitionToDecode(sequenceId);
            }
        }
    }

    /// <summary>
    /// Check if we can accept more sequences.
    /// </summary>
    public bool CanAcceptMoreSequences()
    {
        lock (_lock)
        {
            if (_activeSequences.Count >= _maxBatchSize)
            {
                return false;
            }

            return _scheduler.ShouldApplyBackpressure() == false;
        }
    }

    /// <summary>
    /// Get batch statistics.
    /// </summary>
    public BatchStats GetStats()
    {
        lock (_lock)
        {
            var prefillSeqs = GetPrefillSequences();
            var decodeSeqs = GetDecodeSequences();

            return new BatchStats
            {
                TotalSequences = _activeSequences.Count,
                PrefillSequences = prefillSeqs.Count,
                DecodeSequences = decodeSeqs.Count,
                MaxBatchSize = _maxBatchSize,
                Utilization = (double)_activeSequences.Count / _maxBatchSize
            };
        }
    }

    /// <summary>
    /// Clear all sequences from the batch.
    /// </summary>
    public void Clear()
    {
        lock (_lock)
        {
            foreach (var seqId in new List<int>(_activeSequences))
            {
                RemoveSequence(seqId);
            }
        }
    }
}

/// <summary>
/// Statistics about the current batch.
/// </summary>
public class BatchStats
{
    public int TotalSequences { get; set; }
    public int PrefillSequences { get; set; }
    public int DecodeSequences { get; set; }
    public int MaxBatchSize { get; set; }
    public double Utilization { get; set; }

    public override string ToString()
    {
        return $"BatchStats: " +
               $"Total={TotalSequences}/{MaxBatchSize}, " +
               $"Prefill={PrefillSequences}, " +
               $"Decode={DecodeSequences}, " +
               $"Utilization={Utilization:P0}";
    }
}
