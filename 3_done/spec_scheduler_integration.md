# Spec: Scheduler Integration

## Overview
Implement the scheduler integration for PagedAttention, enabling the serving scheduler to query available memory and apply backpressure when blocks are exhausted. This ensures efficient resource management in production serving workloads.

## Target Directory
`src/MlFramework/Inference/Scheduling/`

## Classes to Implement

### PagedAttentionScheduler
```csharp
using MlFramework.Inference.PagedAttention;

namespace MlFramework.Inference.Scheduling;

/// <summary>
/// Scheduler integration for PagedAttention KV cache management.
/// Provides memory-aware scheduling decisions and backpressure mechanisms.
/// </summary>
public class PagedAttentionScheduler
{
    private readonly KVCacheBlockManager _blockManager;
    private readonly int _maxSequencesInFlight;
    private readonly int _reservedBlocks;

    public PagedAttentionScheduler(
        KVCacheBlockManager blockManager,
        int maxSequencesInFlight = 100,
        double reservedBlockRatio = 0.1)
    {
        _blockManager = blockManager;
        _maxSequencesInFlight = maxSequencesInFlight;
        _reservedBlocks = (int)(blockManager.TotalBlocks * reservedBlockRatio);
    }

    /// <summary>
    /// Check if a new request can be admitted based on memory availability.
    /// </summary>
    /// <param name="estimatedMaxTokens">Estimated maximum tokens for the request</param>
    /// <param name="currentSequenceCount">Current number of active sequences</param>
    /// <returns>Admission decision with details</returns>
    public AdmissionDecision CanAdmitRequest(
        int estimatedMaxTokens,
        int currentSequenceCount)
    {
        var stats = _blockManager.GetStats();

        // Check if we have too many sequences in flight
        if (currentSequenceCount >= _maxSequencesInFlight)
        {
            return AdmissionDecision.Rejected(
                "Maximum sequences in flight exceeded"
            );
        }

        // Estimate blocks needed for this request
        int blocksNeeded = EstimateBlocksNeeded(estimatedMaxTokens);

        // Check if we have enough free blocks
        int availableBlocks = stats.FreeBlocks - _reservedBlocks;

        if (availableBlocks < blocksNeeded)
        {
            return AdmissionDecision.Rejected(
                $"Insufficient memory: need {blocksNeeded} blocks, have {availableBlocks}"
            );
        }

        return AdmissionDecision.Accepted(
            availableBlocks - blocksNeeded
        );
    }

    /// <summary>
    /// Get the number of available blocks for new requests.
    /// </summary>
    public int GetAvailableBlocks()
    {
        var stats = _blockManager.GetStats();
        return Math.Max(0, stats.FreeBlocks - _reservedBlocks);
    }

    /// <summary>
    /// Estimate the number of blocks needed for a request.
    /// </summary>
    private int EstimateBlocksNeeded(int estimatedTokens)
    {
        int blockSize = _blockManager.BlockSize;
        return (estimatedTokens + blockSize - 1) / blockSize;
    }

    /// <summary>
    /// Get memory pressure status (0 = low, 1 = high).
    /// </summary>
    public double GetMemoryPressure()
    {
        var stats = _blockManager.GetStats();
        int availableBlocks = Math.Max(0, stats.FreeBlocks - _reservedBlocks);
        return 1.0 - (availableBlocks / (double)_blockManager.TotalBlocks);
    }

    /// <summary>
    /// Check if memory pressure is high enough to trigger backpressure.
    /// </summary>
    public bool ShouldApplyBackpressure(double threshold = 0.8)
    {
        return GetMemoryPressure() >= threshold;
    }

    /// <summary>
    /// Get recommended maximum sequence length for new requests.
    /// </summary>
    public int GetRecommendedMaxSequenceLength()
    {
        int availableBlocks = GetAvailableBlocks();
        return availableBlocks * _blockManager.BlockSize;
    }

    /// <summary>
    /// Get scheduler statistics.
    /// </summary>
    public SchedulerStats GetStats()
    {
        var blockStats = _blockManager.GetStats();

        return new SchedulerStats
        {
            TotalBlocks = blockStats.TotalBlocks,
            FreeBlocks = blockStats.FreeBlocks,
            AllocatedBlocks = blockStats.AllocatedBlocks,
            AvailableBlocks = GetAvailableBlocks(),
            ReservedBlocks = _reservedBlocks,
            ActiveSequences = blockStats.ActiveSequences,
            MemoryPressure = GetMemoryPressure(),
            MaxSequencesInFlight = _maxSequencesInFlight
        };
    }
}

/// <summary>
/// Result of a request admission decision.
/// </summary>
public class AdmissionDecision
{
    /// <summary>
    /// Whether the request was admitted.
    /// </summary>
    public bool Accepted { get; private set; }

    /// <summary>
    /// Reason for rejection (if rejected).
    /// </summary>
    public string? RejectionReason { get; private set; }

    /// <summary>
    /// Number of blocks available after admission (if accepted).
    /// </summary>
    public int RemainingBlocks { get; private set; }

    private AdmissionDecision(bool accepted, string? reason, int remainingBlocks)
    {
        Accepted = accepted;
        RejectionReason = reason;
        RemainingBlocks = remainingBlocks;
    }

    public static AdmissionDecision Accepted(int remainingBlocks)
    {
        return new AdmissionDecision(true, null, remainingBlocks);
    }

    public static AdmissionDecision Rejected(string reason)
    {
        return new AdmissionDecision(false, reason, 0);
    }
}

/// <summary>
/// Statistics about the scheduler state.
/// </summary>
public class SchedulerStats
{
    public int TotalBlocks { get; set; }
    public int FreeBlocks { get; set; }
    public int AllocatedBlocks { get; set; }
    public int AvailableBlocks { get; set; }
    public int ReservedBlocks { get; set; }
    public int ActiveSequences { get; set; }
    public double MemoryPressure { get; set; }
    public int MaxSequencesInFlight { get; set; }

    public override string ToString()
    {
        return $"SchedulerStats: " +
               $"Available={AvailableBlocks}/{TotalBlocks}, " +
               $"Reserved={ReservedBlocks}, " +
               $"Pressure={MemoryPressure:P0}, " +
               $"ActiveSeqs={ActiveSequences}";
    }
}
```

### MemoryAwareScheduler
```csharp
namespace MlFramework.Inference.Scheduling;

/// <summary>
/// Memory-aware request queue with backpressure support.
/// Manages pending requests and applies backpressure when memory is constrained.
/// </summary>
public class MemoryAwareScheduler
{
    private readonly PagedAttentionScheduler _pagedScheduler;
    private readonly Queue<PendingRequest> _requestQueue;
    private readonly int _maxQueueSize;
    private readonly object _lock = new object();

    public MemoryAwareScheduler(
        PagedAttentionScheduler pagedScheduler,
        int maxQueueSize = 1000)
    {
        _pagedScheduler = pagedScheduler;
        _maxQueueSize = maxQueueSize;
        _requestQueue = new Queue<PendingRequest>();
    }

    /// <summary>
    /// Add a request to the queue.
    /// </summary>
    /// <returns>True if queued, false if queue is full</returns>
    public bool EnqueueRequest(PendingRequest request)
    {
        lock (_lock)
        {
            if (_requestQueue.Count >= _maxQueueSize)
            {
                return false;
            }

            _requestQueue.Enqueue(request);
            return true;
        }
    }

    /// <summary>
    /// Try to admit the next request from the queue.
    /// </summary>
    /// <returns>Admitted request or null if no request can be admitted</returns>
    public PendingRequest? TryAdmitNextRequest()
    {
        lock (_lock)
        {
            if (_requestQueue.Count == 0)
            {
                return null;
            }

            // Peek at the next request
            var request = _requestQueue.Peek();

            // Check if we can admit it
            var decision = _pagedScheduler.CanAdmitRequest(
                request.EstimatedMaxTokens,
                _pagedScheduler.GetStats().ActiveSequences + 1
            );

            if (decision.Accepted)
            {
                _requestQueue.Dequeue();
                return request;
            }

            return null;
        }
    }

    /// <summary>
    /// Get the number of queued requests.
    /// </summary>
    public int GetQueueSize()
    {
        lock (_lock)
        {
            return _requestQueue.Count;
        }
    }

    /// <summary>
    /// Get all queued requests (for inspection/debugging).
    /// </summary>
    public List<PendingRequest> GetQueuedRequests()
    {
        lock (_lock)
        {
            return _requestQueue.ToList();
        }
    }

    /// <summary>
    /// Check if backpressure should be applied.
    /// </summary>
    public bool ShouldApplyBackpressure(double threshold = 0.8)
    {
        return _pagedScheduler.ShouldApplyBackpressure(threshold);
    }

    /// <summary>
    /// Clear the request queue.
    /// </summary>
    public void ClearQueue()
    {
        lock (_lock)
        {
            _requestQueue.Clear();
        }
    }
}

/// <summary>
/// A pending request waiting to be admitted.
/// </summary>
public class PendingRequest
{
    /// <summary>
    /// Unique request ID.
    /// </summary>
    public int RequestId { get; set; }

    /// <summary>
    /// Estimated maximum tokens for this request.
    /// </summary>
    public int EstimatedMaxTokens { get; set; }

    /// <summary>
    /// Timestamp when request was queued.
    /// </summary>
    public DateTime QueuedAt { get; set; }

    /// <summary>
    /// Priority level for this request.
    /// </summary>
    public int Priority { get; set; } = 0;

    /// <summary>
    /// Request metadata.
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();
}
```

## Requirements
1. **Memory Awareness**: Accurate tracking of available blocks
2. **Admission Control**: Proper request admission/rejection logic
3. **Backpressure**: Apply backpressure when memory is constrained
4. **Queue Management**: Efficient request queue operations
5. **Thread Safety**: Support concurrent access from multiple threads
6. **Reserved Blocks**: Maintain a safety margin of reserved blocks

## Testing Requirements
1. Unit tests for request admission logic
2. Unit tests for block estimation
3. Unit tests for memory pressure calculation
4. Unit tests for backpressure application
5. Unit tests for request queue operations
6. Concurrent access tests (multiple threads enqueueing/dequeueing)
7. Integration tests with block manager

## Estimated Time
45-60 minutes

## Dependencies
- spec_kvcache_block_manager.md

## Success Criteria
- Accurate admission decisions
- Proper backpressure mechanism
- Efficient queue management
- Thread-safe operations
- Correct statistics reporting
