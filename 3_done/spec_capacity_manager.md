# Spec: Capacity Manager for Continuous Batching

## Overview
Implement the capacity manager responsible for tracking and managing GPU memory and compute constraints for continuous batching. The manager ensures that batches stay within available resources and provides estimates for resource allocation decisions.

## Class: CapacityManager
```csharp
public class CapacityManager
{
    private readonly CapacityConstraints _constraints;
    private readonly IGPUResourceManager _gpuManager;
    private readonly Dictionary<RequestId, ResourceAllocation> _allocations;
    private readonly object _lock;
    private long _totalAllocatedBytes;
    private int _totalAllocatedSlots;

    public CapacityManager(CapacityConstraints constraints,
                           IGPUResourceManager gpuManager)
    {
        _constraints = constraints;
        _gpuManager = gpuManager;
        _allocations = new Dictionary<RequestId, ResourceAllocation>();
        _lock = new object();
        _totalAllocatedBytes = 0;
        _totalAllocatedSlots = 0;
    }

    // Try to allocate resources for a request
    public bool TryAllocate(Request request, out ResourceAllocation allocation);

    // Release resources for a completed request
    public void Release(RequestId requestId);

    // Check if batch fits within capacity
    public bool CanFitBatch(int requestCount, long estimatedMemoryBytes);

    // Get current capacity utilization
    public CapacityUtilization GetUtilization();

    // Get available capacity
    public (int AvailableSlots, long AvailableMemoryBytes) GetAvailableCapacity();

    // Reset all allocations (for recovery)
    public void Reset();

    // Update resource usage after token generation
    public void UpdateUsage(RequestId requestId, int additionalTokens);
}
```

---

## Class: CapacityConstraints
```csharp
public record class CapacityConstraints(
    int MaxBatchSize,               // Maximum requests per batch
    long MaxMemoryBytes,            // Maximum memory in bytes
    long MemoryPerSlotBytes,        // Estimated memory per request slot
    int MaxConcurrentRequests,      // Maximum concurrent requests (across batches)
    double MemoryBufferRatio        // Safety buffer for memory (0.0-1.0)
)
{
    public static readonly CapacityConstraints Default = new(
        MaxBatchSize: 32,
        MaxMemoryBytes: 16L * 1024 * 1024 * 1024, // 16GB
        MemoryPerSlotBytes: 512L * 1024 * 1024,   // 512MB per request
        MaxConcurrentRequests: 64,
        MemoryBufferRatio: 0.1  // 10% buffer
    );

    public long EffectiveMemoryBytes =>
        (long)(_constraints.MaxMemoryBytes * (1.0 - _constraints.MemoryBufferRatio));
}
```

**Purpose**: Define capacity limits and safety buffers.

---

## Class: ResourceAllocation
```csharp
public record class ResourceAllocation(
    RequestId RequestId,
    long AllocatedMemoryBytes,
    int AllocatedSlots,
    DateTime AllocatedTime
)
{
    public TimeSpan Age => DateTime.UtcNow - AllocatedTime;
}
```

**Purpose**: Track resource allocation per request.

---

## Class: CapacityUtilization
```csharp
public record class CapacityUtilization(
    double SlotUtilization,         // Percentage of slots used
    double MemoryUtilization,      // Percentage of memory used
    int ActiveRequestCount,
    long TotalMemoryUsedBytes,
    long AvailableMemoryBytes
)
{
    public double AverageUtilization =>
        (SlotUtilization + MemoryUtilization) / 2.0;
}
```

**Purpose**: Report capacity utilization statistics.

---

## Interface: IGPUResourceManager
```csharp
public interface IGPUResourceManager
{
    // Get total GPU memory available
    long GetTotalMemoryBytes();

    // Get current GPU memory usage
    long GetCurrentMemoryUsageBytes();

    // Get GPU utilization percentage (0-100)
    double GetUtilization();

    // Check if GPU is available
    bool IsAvailable();
}
```

**Purpose**: Abstraction for GPU resource monitoring.

---

## Implementation Details

### TryAllocate
```csharp
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
```

**Requirements**:
- Estimate memory requirements
- Check capacity limits
- Atomically allocate
- Return allocation details

---

### EstimateRequiredMemory
```csharp
private long EstimateRequiredMemory(Request request)
{
    // Estimate memory for:
    // 1. Prompt tokens (prefill cache)
    // 2. Max generation tokens (KV cache)
    // 3. Activation memory

    const int bytesPerToken = 2; // FP16
    const int kvMultiplier = 2; // Key + Value
    const int activationMultiplier = 1.5; // Estimate

    // Estimate prompt token count (conservative: 4 chars per token)
    int estimatedPromptTokens = (request.Prompt.Length / 4) + 10; // +10 buffer

    int totalTokens = estimatedPromptTokens + request.MaxTokens;
    long kvCacheMemory = totalTokens * bytesPerToken * kvMultiplier;
    long activationMemory = kvCacheMemory * (activationMultiplier / _constraints.MaxBatchSize);

    return kvCacheMemory + activationMemory;
}
```

**Requirements**:
- Conservative memory estimates
- Account for KV cache
- Account for activations
- Include buffer for uncertainty

---

### Release
```csharp
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
```

**Requirements**:
- Remove allocation from tracking
- Update totals
- Handle missing allocations gracefully

---

### CanFitBatch
```csharp
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
```

**Requirements**:
- Check both size and memory constraints
- Use effective memory (with buffer)
- Thread-safe

---

### GetUtilization
```csharp
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
```

**Requirements**:
- Calculate utilization percentages
- Handle division by zero
- Return accurate statistics

---

### GetAvailableCapacity
```csharp
public (int AvailableSlots, long AvailableMemoryBytes) GetAvailableCapacity()
{
    lock (_lock)
    {
        int availableSlots = _constraints.MaxConcurrentRequests - _totalAllocatedSlots;
        long availableMemory = _constraints.EffectiveMemoryBytes - _totalAllocatedBytes;

        return (availableSlots, availableMemory);
    }
}
```

**Requirements**:
- Return available resources
- Thread-safe

---

### Reset
```csharp
public void Reset()
{
    lock (_lock)
    {
        _allocations.Clear();
        _totalAllocatedBytes = 0;
        _totalAllocatedSlots = 0;
    }
}
```

**Requirements**:
- Clear all allocations
- Reset counters
- Thread-safe

---

### UpdateUsage
```csharp
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
```

**Requirements**:
- Update based on actual token generation
- Handle missing allocations
- Update totals

---

## Files to Create

### Implementation
- `src/MLFramework/Inference/ContinuousBatching/Scheduler/CapacityManager.cs`
- `src/MLFramework/Inference/ContinuousBatching/Scheduler/CapacityConstraints.cs`
- `src/MLFramework/Inference/ContinuousBatching/Scheduler/ResourceAllocation.cs`
- `src/MLFramework/Inference/ContinuousBatching/Scheduler/CapacityUtilization.cs`

### Tests
- `tests/MLFramework.Tests/Inference/ContinuousBatching/CapacityManagerTests.cs`

---

## Dependencies
- `spec_continuous_batching_models.md` (Request, RequestId)

---

## Testing Requirements

### Unit Tests (with Mock IGPUResourceManager)
1. **Basic Allocation**:
   - Allocate resources successfully
   - Reject when capacity exceeded
   - Return allocation details

2. **Resource Release**:
   - Release allocated resources
   - Update capacity counters
   - Handle missing allocations

3. **Capacity Checks**:
   - CanFitBatch returns correct result
   - GetAvailableCapacity returns accurate values
   - Handle edge cases (zero capacity)

4. **Utilization Tracking**:
   - GetUtilization returns accurate percentages
   - Handle division by zero
   - Reflect current allocations

5. **Memory Estimation**:
   - EstimateRequiredMemory produces reasonable values
   - Different prompts produce different estimates
   - Include safety buffers

6. **Usage Updates**:
   - UpdateUsage adjusts memory allocation
   - Track actual token generation
   - Update totals correctly

7. **Reset Functionality**:
   - Reset clears all allocations
   - Reset zeroes all counters
   - Thread-safe reset

8. **Thread Safety**:
   - Concurrent allocation attempts
   - Concurrent release operations
   - Race conditions handling

---

## Success Criteria
- [ ] All public methods implemented
- [ ] Capacity constraints enforced
- [ ] Memory estimation reasonable
- [ ] Thread-safe operations
- [ ] Unit tests cover all scenarios
- [ ] Resource tracking accurate
- [ ] Buffer mechanism works correctly
