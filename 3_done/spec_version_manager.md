# Spec: Model Version Manager Implementation

## Overview
Implement IModelVersionManager for loading/unloading model versions, managing memory resources, and ensuring version isolation.

## Tasks

### 1. Create IModelVersionManager Interface
**File:** `src/ModelVersioning/IModelVersionManager.cs`

```csharp
public interface IModelVersionManager
{
    void LoadVersion(string modelId, string version, string modelPath);
    void UnloadVersion(string modelId, string version);
    bool IsVersionLoaded(string modelId, string version);
    IEnumerable<string> GetLoadedVersions(string modelId);
    void WarmUpVersion(string modelId, string version, IEnumerable<object> warmupData);
    VersionLoadInfo GetLoadInfo(string modelId, string version);
}
```

### 2. Create VersionLoadInfo Class
**File:** `src/ModelVersioning/VersionLoadInfo.cs`

```csharp
public class VersionLoadInfo
{
    public string ModelId { get; set; }
    public string Version { get; set; }
    public bool IsLoaded { get; set; }
    public DateTime LoadTime { get; set; }
    public long MemoryUsageBytes { get; set; }
    public int RequestCount { get; set; }
    public string Status { get; set; }
}
```

### 3. Implement ModelVersionManager Class
**File:** `src/ModelVersioning/ModelVersionManager.cs`

```csharp
public class ModelVersionManager : IModelVersionManager
{
    private readonly Dictionary<string, LoadedModel> _loadedModels;
    private readonly IModelRegistry _registry;
    private readonly object _loadLock;

    public ModelVersionManager(IModelRegistry registry)
    {
        _registry = registry;
        _loadedModels = new Dictionary<string, LoadedModel>();
        _loadLock = new object();
    }

    public void LoadVersion(string modelId, string version, string modelPath)
    {
        // Load model from path
        // Track memory usage
        // Initialize model state
    }

    public void UnloadVersion(string modelId, string version)
    {
        // Unload model
        // Free memory
    }

    public bool IsVersionLoaded(string modelId, string version)
    {
        // Check if loaded
    }

    public IEnumerable<string> GetLoadedVersions(string modelId)
    {
        // Return loaded versions
    }

    public void WarmUpVersion(string modelId, string version, IEnumerable<object> warmupData)
    {
        // Run warmup inference
        // Initialize model state
    }

    public VersionLoadInfo GetLoadInfo(string modelId, string version)
    {
        // Return load information
    }
}
```

### 4. Create LoadedModel Internal Class
**File:** `src/ModelVersioning/LoadedModel.cs` (internal to ModelVersionManager)

```csharp
internal class LoadedModel
{
    public string ModelId { get; set; }
    public string Version { get; set; }
    public object ModelInstance { get; set; } // Generic model wrapper
    public DateTime LoadTime { get; set; }
    public long MemoryUsageBytes { get; set; }
    public int RequestCount { get; set; }
    public bool IsWarmingUp { get; set; }
}
```

### 5. Implement LoadVersion Logic
- Validate model exists in registry
- Load model from path (placeholder for actual model loading)
- Track memory usage (estimate or measure)
- Initialize request counter
- Store in loaded models dictionary
- Thread-safe loading

### 6. Implement UnloadVersion Logic
- Check if version is loaded
- Decrement request count
- Dispose model instance
- Remove from loaded models
- Thread-safe unloading

### 7. Implement WarmUpVersion Logic
- Check if version is loaded
- Set warming up flag
- Run inference on warmup data
- Clear warming up flag
- Track warmup completion

### 8. Implement Version Isolation
- Each version maintains separate state
- No shared state between versions
- Separate request counters per version

## Validation
- Model must exist in registry before loading
- Cannot load same version twice
- Cannot unload non-existent version
- Warmup requires loaded version
- Thread-safe operations

## Testing
**File:** `tests/ModelVersioning/ModelVersionManagerTests.cs`

Create unit tests for:
1. LoadVersion successfully loads model
2. LoadVersion throws for non-existent model
3. UnloadVersion successfully unloads model
4. UnloadVersion throws for non-loaded version
5. IsVersionLoaded returns correct status
6. GetLoadedVersions returns loaded versions
7. WarmUpVersion runs warmup data
8. WarmUpVersion throws for non-loaded version
9. GetLoadInfo returns correct information
10. Concurrent load operations
11. Concurrent unload operations
12. Request count tracking

## Dependencies
- Spec: spec_model_registry.md
- Spec: spec_model_data_models.md
