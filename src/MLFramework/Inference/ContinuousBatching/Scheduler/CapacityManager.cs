namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Responsible for tracking and managing GPU memory and compute constraints
/// for continuous batching. The manager ensures that batches stay within available
/// resources and provides estimates for resource allocation decisions.
/// </summary>
public class CapacityManager
{
    private readonly CapacityConstraints _constraints;
    private readonly IGPUResourceManager _gpuManager;
    private readonly Dictionary<RequestId, ResourceAllocation> _allocations;
    private readonly object _lock;
    private long _totalAllocatedBytes;
    private int _totalAllocatedSlots;

    /// <summary>
    /// Initializes a new instance of the CapacityManager.
    /// </summary>
    /// <param name="constraints">Capacity constraints</param>
    /// <param name="gpuManager">GPU resource manager</param>
    public CapacityManager(
        CapacityConstraints constraints,
        IGPUResourceManager gpuManager)
    {
        _constraints = constraints;
        _gpuManager = gpuManager;
        _allocations = new Dictionary<RequestId, ResourceAllocation>();
        _lock = new object();
        _totalAllocatedBytes = 0;
        _totalAllocatedSlots = 0;
    }

    /// <summary>
    /// Tries to allocate resources for a request.
    /// </summary>
    /// <param name="request">The request to allocate resources for</param>
    /// <param name="allocation">The allocated resources if successful</param>
    /// <returns>True if allocation was successful, false otherwise</returns>
    public bool TryAllocate(Request request, out ResourceAllocation allocation)
    {
        allocation = null;

        lock (_lock)
        {
            // Estimate required memory
            long estimatedMemory = EstimateRequiredMemory(request);
            int requiredSlots = 1; // One slot per request

            // Check capacity limits
            if (_totalAllocatedBytes + estimatedMemory > _constraints.EffectiveMemoryBytes)
                return false;

            if (_totalAllocatedSlots + requiredSlots > _constraints.MaxConcurrentRequests)
                return false;

            // Allocate resources
            allocation = new ResourceAllocation(
                request.Id,
                estimatedMemory,
                requiredSlots,
                DateTime.UtcNow
            );

            _allocations[request.Id] = allocation;
            _totalAllocatedBytes += estimatedMemory;
            _totalAllocatedSlots += requiredSlots;

            return true;
        }
    }

    /// <summary>
    /// Releases resources for a completed request.
    /// </summary>
    /// <param name="requestId">The ID of the request to release resources for</param>
    public void Release(RequestId requestId)
    {
        lock (_lock)
        {
            if (!_allocations.TryGetValue(requestId, out var allocation))
                return;

            _totalAllocatedBytes -= allocation.AllocatedMemoryBytes;
            _totalAllocatedSlots -= allocation.AllocatedSlots;
            _allocations.Remove(requestId);
        }
    }

    /// <summary>
    /// Checks if a batch fits within the available capacity.
    /// </summary>
    /// <param name="requestCount">Number of requests in the batch</param>
    /// <param name="estimatedMemoryBytes">Estimated memory needed for the batch</param>
    /// <returns>True if the batch fits, false otherwise</returns>
    public bool CanFitBatch(int requestCount, long estimatedMemoryBytes)
    {
        lock (_lock)
        {
            // Check if batch fits within capacity
            bool fitsSize = _totalAllocatedSlots + requestCount <= _constraints.MaxConcurrentRequests;
            bool fitsMemory = _totalAllocatedBytes + estimatedMemoryBytes <= _constraints.EffectiveMemoryBytes;

            return fitsSize && fitsMemory;
        }
    }

    /// <summary>
    /// Gets the current capacity utilization.
    /// </summary>
    /// <returns>Current utilization statistics</returns>
    public CapacityUtilization GetUtilization()
    {
        lock (_lock)
        {
            double slotUtilization = _constraints.MaxConcurrentRequests > 0
                ? (double)_totalAllocatedSlots / _constraints.MaxConcurrentRequests * 100.0
                : 0.0;

            double memoryUtilization = _constraints.EffectiveMemoryBytes > 0
                ? (double)_totalAllocatedBytes / _constraints.EffectiveMemoryBytes * 100.0
                : 0.0;

            return new CapacityUtilization(
                slotUtilization,
                memoryUtilization,
                _allocations.Count,
                _totalAllocatedBytes,
                _constraints.EffectiveMemoryBytes - _totalAllocatedBytes
            );
        }
    }

    /// <summary>
    /// Gets the available capacity.
    /// </summary>
    /// <returns>Available slots and memory in bytes</returns>
    public (int AvailableSlots, long AvailableMemoryBytes) GetAvailableCapacity()
    {
        lock (_lock)
        {
            int availableSlots = _constraints.MaxConcurrentRequests - _totalAllocatedSlots;
            long availableMemory = _constraints.EffectiveMemoryBytes - _totalAllocatedBytes;

            return (availableSlots, availableMemory);
        }
    }

    /// <summary>
    /// Resets all allocations (for recovery).
    /// </summary>
    public void Reset()
    {
        lock (_lock)
        {
            _allocations.Clear();
            _totalAllocatedBytes = 0;
            _totalAllocatedSlots = 0;
        }
    }

    /// <summary>
    /// Updates resource usage after token generation.
    /// </summary>
    /// <param name="requestId">The ID of the request to update</param>
    /// <param name="additionalTokens">Number of additional tokens generated</param>
    public void UpdateUsage(RequestId requestId, int additionalTokens)
    {
        lock (_lock)
        {
            if (!_allocations.TryGetValue(requestId, out var allocation))
                return;

            // Adjust memory estimate based on actual token generation
            const int bytesPerToken = 2;
            const int kvMultiplier = 2;

            long additionalMemory = additionalTokens * bytesPerToken * kvMultiplier;

            // Update allocation (create new record)
            _allocations[requestId] = allocation with
            {
                AllocatedMemoryBytes = allocation.AllocatedMemoryBytes + additionalMemory
            };

            _totalAllocatedBytes += additionalMemory;
        }
    }

    /// <summary>
    /// Estimates the required memory for a request.
    /// </summary>
    /// <param name="request">The request to estimate memory for</param>
    /// <returns>Estimated memory in bytes</returns>
    private long EstimateRequiredMemory(Request request)
    {
        // Estimate memory for:
        // 1. Prompt tokens (prefill cache)
        // 2. Max generation tokens (KV cache)
        // 3. Activation memory

        const int bytesPerToken = 2; // FP16
        const int kvMultiplier = 2; // Key + Value
        const double activationMultiplier = 1.5; // Estimate

        // Estimate prompt token count (conservative: 4 chars per token)
        int estimatedPromptTokens = (request.Prompt.Length / 4) + 10; // +10 buffer

        int totalTokens = estimatedPromptTokens + request.MaxTokens;
        long kvCacheMemory = totalTokens * bytesPerToken * kvMultiplier;
        long activationMemory = (long)(kvCacheMemory * (activationMultiplier / _constraints.MaxBatchSize));

        return kvCacheMemory + activationMemory;
    }
}
