# Spec: Model Hot-Swapper Core

## Purpose
Implement zero-downtime version swapping, allowing new models to be deployed without disrupting active inference requests.

## Technical Requirements

### Core Functionality
- Swap from one version to another without dropping requests
- Load new version while old version remains active
- Gradually transition traffic from old to new version
- Maintain both versions during transition
- Track swap state (pending, in-progress, completed)

### Data Structures
```csharp
public enum SwapState
{
    NotStarted,
    LoadingNewVersion,
    Transitioning,
    OldVersionDraining,
    Completed,
    Failed
}

public class SwapOperation
{
    public string ModelName { get; }
    public string FromVersion { get; }
    public string ToVersion { get; }
    public SwapState State { get; set; }
    public DateTime StartTime { get; }
    public DateTime? EndTime { get; set; }
    public string ErrorMessage { get; set; }
}

public interface IModelHotSwapper
{
    Task<SwapOperation> SwapVersionAsync(string modelName, string fromVersion, string toVersion);
    SwapOperation GetSwapStatus(string operationId);
    void WaitForDrainage(string modelName, string version, TimeSpan timeout);
    bool IsVersionActive(string modelName, string version);
    Task RollbackAsync(string operationId);
}
```

### Swap Algorithm
1. Validate fromVersion is active and toVersion exists
2. Load toVersion (async, in background)
3. Mark swap as "transitioning"
4. Update router to route new requests to toVersion
5. Wait for old version to drain (no more active requests)
6. Unload fromVersion
7. Mark swap as "completed"

## Dependencies
- `spec_model_loader.md` (to load/unload models)
- `spec_version_router_core.md` (to update routing)

## Testing Requirements
- Swap from v1.0 to v1.1, verify both versions active during transition
- Swap with non-existent target version (should throw)
- Swap with source not active (should throw)
- WaitForDrainage completes when no active requests
- WaitForDrainage times out if requests still active
- Rollback reverts to source version
- GetSwapStatus returns correct state throughout swap
- Concurrent swaps for different models (should work)
- Concurrent swaps for same model (should throw or queue)
- Performance test: Swap completes in < 100ms (excluding model load time)

## Success Criteria
- [ ] Zero dropped requests during swap
- [ ] Both versions active during transition
- [ ] Old version drains before unloading
- [ ] Swap state tracked correctly
- [ ] Rollback functionality works
- [ ] Thread-safe swap operations
- [ ] Swap completes in < 100ms (routing update only)

## Implementation Notes
- Use async/await for non-blocking operations
- Implement proper state machine for swap lifecycle
- Add detailed logging at each swap stage
- Use cancellation tokens to abort swaps if needed
- Consider adding swap queue for same model (optional)
- Add swap timeout handling
- Persist swap state for recovery (optional future)

## Performance Targets
- Swap routing update: < 10ms
- Drain time: configurable, default 30s
- Total swap time: < 100ms + model load time
- Support 10+ concurrent swaps for different models

## Edge Cases
- Model load failure during swap
- Router update failure
- Old version never drains (timeout)
- Rollback during swap
- Multiple swaps queued for same model
