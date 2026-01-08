# Spec: Reference Counting for Safe Unloading

## Purpose
Implement reference counting to track active inference requests, ensuring models are not unloaded while requests are still processing.

## Technical Requirements

### Core Functionality
- Track active inference requests per model version
- Increment reference when request starts
- Decrement reference when request completes
- Prevent unloading while references > 0
- Wait for references to reach zero before unloading
- Thread-safe reference updates

### Data Structures
```csharp
public interface IReferenceTracker
{
    void AcquireReference(string modelName, string version, string requestId);
    void ReleaseReference(string modelName, string version, string requestId);
    int GetReferenceCount(string modelName, string version);
    bool HasReferences(string modelName, string version);
    Task WaitForZeroReferencesAsync(string modelName, string version, TimeSpan timeout, CancellationToken ct = default);
    Dictionary<string, int> GetAllReferenceCounts();
}

public class ReferenceLeakException : Exception
{
    public string ModelName { get; }
    public string Version { get; }
}

public class RequestTracker : IDisposable
{
    private readonly IReferenceTracker _tracker;
    private readonly string _modelName;
    private readonly string _version;
    private readonly string _requestId;

    public RequestTracker(IReferenceTracker tracker, string modelName, string version, string requestId);
    public void Dispose();
}
```

### Reference Lifecycle
1. When inference request starts: AcquireReference()
2. Execute inference
3. When inference completes: ReleaseReference()
4. Can unload model only when GetReferenceCount() == 0

## Dependencies
- None (standalone utility)
- Used by `spec_model_hotswapper.md` (to ensure safe unloading)

## Testing Requirements
- Acquire reference, verify count increases
- Release reference, verify count decreases
- Multiple concurrent requests, verify count is accurate
- WaitForZeroReferences returns immediately when count == 0
- WaitForZeroReferences blocks until count reaches 0
- WaitForZeroReferences times out if count > 0
- Using RequestTracker with using statement auto-releases
- Release reference that doesn't exist (should throw or log)
- Acquire/release under high concurrency (1000 threads)
- Test reference leak detection (optional)

## Success Criteria
- [ ] References increment/decrement correctly
- [ ] Thread-safe under high concurrency (1000+ concurrent requests)
- [ ] WaitForZeroReferences blocks correctly
- [ ] WaitForZeroReferences respects timeout
- [ ] RequestTracker properly auto-releases on disposal
- [ ] No race conditions in reference updates
- [ ] Performance: Acquire/Release < 0.01ms each

## Implementation Notes
- Use `ConcurrentDictionary` for thread-safe storage
- Use `Interlocked.Increment/Decrement` for atomic updates
- Consider using `CountdownEvent` or `AsyncManualResetEvent` for WaitForZeroReferences
- Add logging for reference tracking (for debugging leaks)
- Consider adding reference timeout detection (optional)
- Track request IDs for debugging
- Add metrics for average reference count (optional)

## Performance Targets
- AcquireReference: < 0.01ms
- ReleaseReference: < 0.01ms
- GetReferenceCount: < 0.01ms
- Support 100,000+ reference updates per second
- WaitForZeroReferences wake-up: < 1ms

## Edge Cases
- Release reference that was never acquired
- Multiple threads waiting for zero references
- Model unloaded while waiting for zero references
- Request cancellation holding a reference
- Reference leaks (never released)
