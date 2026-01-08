# Spec: Model Hot Swapper Implementation

## Overview
Implement IModelHotSwapper for zero-downtime version swapping, graceful draining, and instant rollbacks.

## Tasks

### 1. Create IModelHotSwapper Interface
**File:** `src/ModelVersioning/IModelHotSwapper.cs`

```csharp
public interface IModelHotSwapper
{
    Task<SwapResult> SwapVersion(string modelId, string fromVersion, string toVersion);
    Task<RollbackResult> RollbackVersion(string modelId, string targetVersion);
    HealthCheckResult CheckVersionHealth(string modelId, string version);
    Task<bool> DrainVersion(string modelId, string version, TimeSpan timeout);
    SwapStatus GetSwapStatus(string modelId);
}
```

### 2. Create SwapResult Class
**File:** `src/ModelVersioning/SwapResult.cs`

```csharp
public class SwapResult
{
    public bool Success { get; set; }
    public string Message { get; set; }
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public int RequestsDrained { get; set; }
    public int RequestsRemaining { get; set; }
}
```

### 3. Create RollbackResult Class
**File:** `src/ModelVersioning/RollbackResult.cs`

```csharp
public class RollbackResult
{
    public bool Success { get; set; }
    public string Message { get; set; }
    public DateTime RollbackTime { get; set; }
    public string PreviousVersion { get; set; }
    public string NewVersion { get; set; }
}
```

### 4. Create SwapStatus Class
**File:** `src/ModelVersioning/SwapStatus.cs`

```csharp
public class SwapStatus
{
    public string ModelId { get; set; }
    public string CurrentVersion { get; set; }
    public string TargetVersion { get; set; }
    public SwapState State { get; set; }
    public DateTime StartTime { get; set; }
    public int PendingRequests { get; set; }
}
```

### 5. Create SwapState Enum
**File:** `src/ModelVersioning/SwapState.cs`

```csharp
public enum SwapState
{
    Idle,
    Draining,
    Swapping,
    Completed,
    Failed
}
```

### 6. Implement ModelHotSwapper Class
**File:** `src/ModelVersioning/ModelHotSwapper.cs`

```csharp
public class ModelHotSwapper : IModelHotSwapper
{
    private readonly IModelVersionManager _versionManager;
    private readonly IVersionRouter _router;
    private readonly Dictionary<string, SwapStatus> _swapStatuses;
    private readonly object _swapLock;

    public ModelHotSwapper(
        IModelVersionManager versionManager,
        IVersionRouter router)
    {
        _versionManager = versionManager;
        _router = router;
        _swapStatuses = new Dictionary<string, SwapStatus>();
        _swapLock = new object();
    }

    public async Task<SwapResult> SwapVersion(string modelId, string fromVersion, string toVersion)
    {
        // 1. Check health of target version
        // 2. Load target version if not loaded
        // 3. Drain requests from source version
        // 4. Update routing policy
        // 5. Verify complete switch
        // 6. Unload source version (optional)
    }

    public async Task<RollbackResult> RollbackVersion(string modelId, string targetVersion)
    {
        // 1. Check health of target version
        // 2. Load target version if needed
        // 3. Update routing policy immediately
        // 4. Drain from current version
    }

    public HealthCheckResult CheckVersionHealth(string modelId, string version)
    {
        // Check if version is loaded
        // Run inference test
        // Validate response
        // Check memory usage
    }

    public async Task<bool> DrainVersion(string modelId, string version, TimeSpan timeout)
    {
        // Wait for in-flight requests to complete
        // Monitor request count
        // Return success if drained within timeout
    }

    public SwapStatus GetSwapStatus(string modelId)
    {
        // Return current swap status
    }
}
```

### 7. Implement Health Check Logic
- Check if version is loaded
- Run sample inference (placeholder)
- Validate inference result format
- Check memory usage within limits
- Return HealthCheckResult with diagnostics

### 8. Implement Drain Logic
- Track in-flight requests
- Wait for request count to reach 0
- Timeout if draining takes too long
- Return number of drained requests

### 9. Implement Swap Workflow
1. Validate both versions exist
2. Check health of target version
3. Load target version if needed
4. Set swap status to Draining
5. Drain requests from source version
6. Update routing policy to redirect traffic
7. Set swap status to Completed
8. Return SwapResult with metrics

### 10. Implement Rollback Workflow
1. Validate target version exists
2. Check health of target version
3. Update routing policy immediately
4. Drain from current version
5. Return RollbackResult

## Validation
- Source version must exist
- Target version must exist
- Target version must pass health check
- Cannot swap to same version
- Rollback target must be previously served version

## Testing
**File:** `tests/ModelVersioning/ModelHotSwapperTests.cs`

Create unit tests for:
1. SwapVersion with healthy target version
2. SwapVersion throws when target unhealthy
3. SwapVersion drains requests correctly
4. SwapVersion updates routing policy
5. RollbackVersion reverts to previous version
6. RollbackVersion is immediate
7. CheckVersionHealth passes for good version
8. CheckVersionHealth fails for bad version
9. DrainVersion completes successfully
10. DrainVersion times out
11. GetSwapStatus returns correct state
12. Concurrent swap attempts (should fail)
13. Swap with unloaded target version
14. Rollback to production version

## Dependencies
- Spec: spec_version_manager.md
- Spec: spec_version_router.md
- Spec: spec_model_data_models.md
