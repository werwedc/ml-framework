# Spec: Model Version Registry

## Purpose
Create a centralized registry for tracking all deployed model versions with metadata support and semantic versioning.

## Technical Requirements

### Core Functionality
- Register models with unique name + version pairs (e.g., "resnet50" + "v1.0.0")
- Support semantic versioning format (MAJOR.MINOR.PATCH)
- Prevent duplicate registrations
- Allow model unregistration
- Query available versions for a model name
- Check if specific version exists

### Thread Safety
- All operations must be thread-safe for concurrent access
- Use appropriate synchronization (ConcurrentDictionary, ReaderWriterLockSlim, etc.)
- Support high-frequency queries (1000+ RPS)

### Data Structures
```csharp
public class ModelMetadata
{
    public string Version { get; set; }
    public DateTime TrainingDate { get; set; }
    public Dictionary<string, object> Hyperparameters { get; set; }
    public Dictionary<string, float> PerformanceMetrics { get; set; }
    public string ArtifactPath { get; set; }
}

public interface IModelRegistry
{
    void RegisterModel(string name, string version, ModelMetadata metadata);
    void UnregisterModel(string name, string version);
    bool HasVersion(string name, string version);
    IEnumerable<string> GetVersions(string name);
    ModelMetadata GetMetadata(string name, string version);
    IEnumerable<string> GetAllModelNames();
}
```

## Dependencies
- None (foundational component)

## Testing Requirements
- Register single model, verify it's queryable
- Register multiple versions of same model
- Prevent duplicate registration (should throw)
- Unregister existing model, verify it's removed
- Query non-existent model returns empty list
- Concurrent registration test (10 threads)
- Query performance test (1000+ queries under 10ms)

## Success Criteria
- [ ] Registry can store 100+ model versions
- [ ] All operations are thread-safe
- [ ] No memory leaks after register/unregister cycles
- [ ] Performance: 1000 queries < 10ms
- [ ] Unit tests cover all operations

## Implementation Notes
- Use `ConcurrentDictionary<string, ConcurrentDictionary<string, ModelMetadata>>` for thread-safe nested storage
- Consider adding validation for semantic version format
- Add event notifications for registration changes (optional future enhancement)
