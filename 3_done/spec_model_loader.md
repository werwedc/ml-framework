# Spec: Model Loader Interface

## Purpose
Define the interface and basic implementation for loading and unloading model artifacts, supporting both synchronous and asynchronous operations.

## Technical Requirements

### Core Functionality
- Load model from file path with version tracking
- Support synchronous and asynchronous loading
- Unload model with cleanup
- Track loaded models in memory
- Support cancellation during async loads

### Interface Design
```csharp
public interface IModel
{
    string Name { get; }
    string Version { get; }
    DateTime LoadTime { get; }
    bool IsActive { get; set; }
    Task<InferenceResult> InferAsync(InferenceInput input);
}

public interface IModelLoader
{
    IModel Load(string modelPath, string version);
    Task<IModel> LoadAsync(string modelPath, string version, CancellationToken ct = default);
    void Unload(IModel model);
    bool IsLoaded(string name, string version);
    IEnumerable<IModel> GetLoadedModels();
}

public abstract class BaseModel : IModel
{
    public string Name { get; protected set; }
    public string Version { get; protected set; }
    public DateTime LoadTime { get; protected set; }
    public bool IsActive { get; set; } = true;
    public abstract Task<InferenceResult> InferAsync(InferenceInput input);
    public abstract void Dispose();
}
```

### Error Handling
- Throw `FileNotFoundException` if model path doesn't exist
- Throw `InvalidOperationException` if model is already loaded
- Throw `OperationCanceledException` on cancellation
- Validate version format on load

## Dependencies
- `spec_model_version_registry.md` (shares versioning concepts)

## Testing Requirements
- Load valid model, verify it's tracked
- Attempt to load non-existent file (should throw)
- Attempt duplicate load (should throw)
- Async load with cancellation token, verify cancellation works
- Unload model, verify it's removed from tracking
- Query loaded models, verify correct list
- Test with mock model implementation

## Success Criteria
- [ ] Models load successfully from valid paths
- [ ] Duplicate loads are prevented
- [ ] Unloading properly cleans up resources
- [ ] Cancellation token properly stops async loads
- [ ] IsLoaded returns correct state
- [ ] All loaded models are tracked

## Implementation Notes
- Use `ConcurrentDictionary` to track loaded models
- Implement proper disposal pattern in BaseModel
- Consider adding progress reporting for async loads (optional)
- Add logging for load/unload operations
- Mock model should be created for testing

## Performance Targets
- Synchronous load: < 100ms (excluding model init time)
- Async load overhead: < 10ms (excluding model init time)
