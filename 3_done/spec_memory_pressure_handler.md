# Spec: Memory Pressure Handler

## Purpose
Implement LRU eviction of least-recently-used model versions when memory pressure is detected, preventing out-of-memory errors.

## Technical Requirements

### Core Functionality
- Monitor memory usage of loaded models
- Detect memory pressure (configurable threshold)
- Evict LRU versions when pressure detected
- Protect pinned versions from eviction
- Track access frequency for eviction decisions
- Log eviction events for debugging

### Data Structures
```csharp
public class ModelMemoryInfo
{
    public string ModelName { get; }
    public string Version { get; }
    public long MemoryBytes { get; }
    public DateTime LastAccessTime { get; }
    public int AccessCount { get; set; }
    public bool IsPinned { get; set; }
    public long Weight { get; } // For LRU + LFU hybrid
}

public interface IMemoryPressureHandler
{
    void TrackModelLoad(string modelName, string version, long memoryBytes);
    void TrackModelAccess(string modelName, string version);
    void PinModel(string modelName, string version);
    void UnpinModel(string modelName, string version);
    long GetTotalMemoryUsage();
    void SetMemoryThreshold(long thresholdBytes);
    Task EvictIfNeededAsync(long requiredBytes = 0);
    IEnumerable<ModelMemoryInfo> GetLoadedModelsInfo();
    void UntrackModel(string modelName, string version);
}

public class EvictionResult
{
    public List<string> EvictedVersions { get; }
    public long BytesFreed { get; }
    public bool Sufficient { get; }
}
```

### Eviction Algorithm
1. Check current memory usage vs threshold
2. If over threshold or need space:
   - Sort unpinned models by (LastAccessTime, AccessCount)
   - Evict least recently/frequently used models
   - Continue until threshold met or no more to evict
3. Log eviction decisions

## Dependencies
- `spec_model_loader.md` (to unload models)
- `spec_reference_counting.md` (to check if safe to evict)

## Testing Requirements
- Track model load, verify memory usage calculated
- Track model access, verify LastAccessTime updated
- Pin model, verify it's not evicted
- Evict unpinned model, verify it's removed from tracking
- Set memory threshold, verify pressure detected
- EvictIfNeeded frees required memory
- Don't evict models with active references
- LRU eviction order is correct
- Concurrent access tracking (100 threads)
- Performance test: Eviction decision < 10ms

## Success Criteria
- [ ] Accurately tracks memory usage
- [ ] Evicts LRU models under pressure
- [ ] Respects pinned models
- [ ] Doesn't evict models with active references
- [ ] Memory usage stays under threshold
- [ ] Thread-safe under high concurrency
- [ ] Eviction decision < 10ms

## Implementation Notes
- Use `ConcurrentDictionary` for thread-safe model tracking
- Use `SortedDictionary` or priority queue for LRU ordering
- Query actual memory size from models if available
- Estimate memory size if not available (e.g., based on parameters)
- Add GPU memory tracking support (optional)
- Implement hybrid LRU+LFU for better eviction decisions
- Add memory pressure monitoring timer (optional)
- Consider memory pressure callbacks from OS (optional)

## Performance Targets
- TrackModelAccess: < 0.01ms
- GetTotalMemoryUsage: < 1ms
- EvictIfNeededAsync: < 10ms decision time
- Support 100+ loaded models
- Track 10,000+ accesses per second

## Memory Estimation
If model doesn't provide actual memory size:
- Estimate based on parameter count (4 bytes per float)
- Add overhead for activation buffers
- Consider batch size and input dimensions
- Allow manual memory hint configuration

## Edge Cases
- Eviction needed but no unpinned models
- All models pinned
- Evicting model with recent spike in access
- Memory threshold set too low
- Concurrent evictions for same models
