# Spec: Continuous Batching Support

## Overview
Implement continuous batching (also known as iteration-level scheduling) for PagedAttention. This allows new sequences to join existing batches dynamically and sequences to finish and free their blocks, maximizing GPU utilization.

## Target Directory
`src/MlFramework/Inference/Batching/`

## Classes to Implement

### ContinuousBatchManager
```csharp
using MlFramework.Inference.PagedAttention;
using MlFramework.Inference.PagedAttention.Phases;
using MlFramework.Tensor;

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
```

### IterationScheduler
```csharp
using MlFramework.Inference.PagedAttention.Phases;
using MlFramework.Inference.Scheduling;

namespace MlFramework.Inference.Batching;

/// <summary>
/// Orchestrates iteration-level scheduling for continuous batching.
/// Manages the scheduling loop where new sequences can join each iteration.
/// </summary>
public class IterationScheduler
{
    private readonly ContinuousBatchManager _batchManager;
    private readonly MemoryAwareScheduler _memoryScheduler;
    private readonly PrefillOrchestrator _prefillOrchestrator;
    private readonly DecodeOrchestrator _decodeOrchestrator;
    private readonly Action<BatchStats> _iterationCallback;

    public IterationScheduler(
        ContinuousBatchManager batchManager,
        MemoryAwareScheduler memoryScheduler,
        PrefillOrchestrator prefillOrchestrator,
        DecodeOrchestrator decodeOrchestrator,
        Action<BatchStats>? iterationCallback = null)
    {
        _batchManager = batchManager;
        _memoryScheduler = memoryScheduler;
        _prefillOrchestrator = prefillOrchestrator;
        _decodeOrchestrator = decodeOrchestrator;
        _iterationCallback = iterationCallback ?? (stats => { });
    }

    /// <summary>
    /// Run one iteration of the scheduling loop.
    /// </summary>
    public void RunIteration(Dictionary<int, Tensor> queries)
    {
        // Step 1: Try to admit new requests from the queue
        TryAdmitNewRequests();

        // Step 2: Process current batch
        var batchStats = _batchManager.GetStats();

        if (batchStats.TotalSequences == 0)
        {
            return;
        }

        // Step 3: Process prefill sequences (if any)
        var prefillSeqs = _batchManager.GetPrefillSequences();
        if (prefillSeqs.Count > 0)
        {
            ProcessPrefill(prefillSeqs, queries);
        }

        // Step 4: Process decode sequences
        var decodeSeqs = _batchManager.GetDecodeSequences();
        if (decodeSeqs.Count > 0)
        {
            ProcessDecode(decodeSeqs, queries);
        }

        // Step 5: Call iteration callback
        _iterationCallback(batchStats);
    }

    /// <summary>
    /// Try to admit new requests from the memory scheduler queue.
    /// </summary>
    private void TryAdmitNewRequests()
    {
        while (_batchManager.CanAcceptMoreSequences())
        {
            var request = _memoryScheduler.TryAdmitNextRequest();
            if (request == null)
            {
                break;
            }

            _batchManager.TryAddSequence(request.RequestId);
        }
    }

    /// <summary>
    /// Process sequences in prefill phase.
    /// </summary>
    private void ProcessPrefill(List<int> sequenceIds, Dictionary<int, Tensor> queries)
    {
        // Group queries by sequence
        var prefillQueries = queries
            .Where(kvp => sequenceIds.Contains(kvp.Key))
            .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);

        // Process prefill in batch
        var results = _prefillOrchestrator.BatchPrefill(
            prefillQueries,
            CreateRanges(sequenceIds),
            GetKernel()
        );

        // Transition sequences to decode
        foreach (var seqId in sequenceIds)
        {
            _batchManager.TransitionToDecode(seqId);
        }
    }

    /// <summary>
    /// Process sequences in decode phase.
    /// </summary>
    private void ProcessDecode(List<int> sequenceIds, Dictionary<int, Tensor> queries)
    {
        // Group queries by sequence
        var decodeQueries = queries
            .Where(kvp => sequenceIds.Contains(kvp.Key))
            .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);

        // Get current lengths
        var lengths = CreateLengthMap(sequenceIds);

        // Process decode in batch
        var results = _decodeOrchestrator.BatchDecode(
            decodeQueries,
            lengths,
            GetKernel()
        );
    }

    private Dictionary<int, (int start, int end)> CreateRanges(List<int> sequenceIds)
    {
        // Create token ranges for each sequence
        var ranges = new Dictionary<int, (int, int)>();
        foreach (var seqId in sequenceIds)
        {
            // TODO: Get actual ranges from block table
            ranges[seqId] = (0, 100);
        }
        return ranges;
    }

    private Dictionary<int, int> CreateLengthMap(List<int> sequenceIds)
    {
        // Get current lengths for each sequence
        var lengths = new Dictionary<int, int>();
        foreach (var seqId in sequenceIds)
        {
            // TODO: Get actual lengths from block table
            lengths[seqId] = 100;
        }
        return lengths;
    }

    private IPagedAttentionKernel GetKernel()
    {
        // Return appropriate kernel based on device
        // This is a placeholder
        return new StandardPagedAttentionKernel(128);
    }
}

/// <summary>
/// Configuration for iteration scheduling.
/// </summary>
public class IterationSchedulerConfig
{
    /// <summary>
    /// Maximum batch size.
    /// </summary>
    public int MaxBatchSize { get; set; } = 32;

    /// <summary>
    /// Minimum sequences before starting decode phase.
    /// </summary>
    public int MinDecodeSequences { get; set; } = 1;

    /// <summary>
    /// Whether to prioritize prefill sequences.
    /// </summary>
    public bool PrioritizePrefill { get; set; } = true;

    /// <summary>
    /// Maximum number of prefill sequences per iteration.
    /// </summary>
    public int MaxPrefillPerIteration { get; set; } = 4;
}
```

## Requirements
1. **Dynamic Batching**: Support adding/removing sequences at iteration boundaries
2. **Phase Awareness**: Handle both prefill and decode sequences in the same batch
3. **Memory Efficiency**: Free blocks as sequences complete
4. **Priority Handling**: Optional prioritization of prefill vs decode
5. **Thread Safety**: Support concurrent access to batch state
6. **Scheduling**: Efficient iteration-level scheduling

## Testing Requirements
1. Unit tests for adding sequences to batch
2. Unit tests for removing sequences from batch
3. Unit tests for batch statistics
4. Unit tests for iteration scheduling
5. Unit tests for phase transitions
6. Integration tests with scheduler
7. Concurrent access tests

## Estimated Time
60 minutes

## Dependencies
- spec_kvcache_block_manager.md
- spec_block_table.md
- spec_prefill_decode_separation.md
- spec_scheduler_integration.md

## Success Criteria
- Dynamic sequence addition/removal works correctly
- Efficient iteration scheduling
- Proper memory management (blocks freed when sequences finish)
- Accurate batch statistics
- Thread-safe operations
- Smooth phase transitions
