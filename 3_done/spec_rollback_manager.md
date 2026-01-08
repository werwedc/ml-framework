# Spec: Rollback Manager

## Purpose
Manage rollback history and provide instant rollback capabilities to revert to previous model versions.

## Technical Requirements

### Core Functionality
- Record successful deployments with metadata
- Maintain history of deployed versions
- Rollback to any previous version instantly
- Track rollback reasons and timestamps
- Limit history size (e.g., last 10 deployments)
- Automatic rollback on error rate threshold

### Data Structures
```csharp
public class DeploymentRecord
{
    public string DeploymentId { get; }
    public string ModelName { get; }
    public string FromVersion { get; }
    public string ToVersion { get; }
    public DateTime DeploymentTime { get; }
    public string DeployedBy { get; }
    public DeploymentStatus Status { get; }
    public string Reason { get; set; }
    public Dictionary<string, float> PreDeploymentMetrics { get; set; }
    public Dictionary<string, float> PostDeploymentMetrics { get; set; }
}

public enum DeploymentStatus
{
    Success,
    RolledBack,
    Failed
}

public interface IRollbackManager
{
    string RecordDeployment(string modelName, string fromVersion, string toVersion, string deployedBy);
    Task RollbackAsync(string deploymentId, string reason, string initiatedBy);
    Task RollbackToVersionAsync(string modelName, string version, string reason, string initiatedBy);
    IEnumerable<DeploymentRecord> GetDeploymentHistory(string modelName, int limit = 10);
    DeploymentRecord GetDeployment(string deploymentId);
    bool CanRollback(string deploymentId);
    void SetAutoRollbackThreshold(string modelName, float errorRateThreshold, TimeSpan observationWindow);
    void MonitorErrorRate(string modelName, string version, float currentErrorRate);
}

public class RollbackResult
{
    public bool Success { get; }
    public string PreviousDeploymentId { get; }
    public string CurrentDeploymentId { get; }
    public DateTime RollbackTime { get; }
    public string Message { get; }
}
```

### Rollback Algorithm
1. Validate deployment can be rolled back (has previous version)
2. Load previous version
3. Update router to route to previous version
4. Wait for current version to drain
5. Unload current version
6. Mark deployment as "RolledBack"
7. Create new deployment record for rollback

## Dependencies
- `spec_model_hotswapper.md` (uses swap logic)
- `spec_version_router_core.md` (updates routing)

## Testing Requirements
- Record deployment, verify it's in history
- Get deployment history, verify correct order
- Rollback to previous version, verify routing updated
- Attempt rollback with no previous version (should throw)
- Rollback to version already rolled back (should throw)
- Auto-rollback trigger when error rate exceeds threshold
- Monitor error rate below threshold, no auto-rollback
- Multiple rollbacks for same model
- Get deployment by ID
- Limit history size test (keep only last N)
- Performance test: Rollback completes in < 100ms

## Success Criteria
- [ ] Deploys recorded with full metadata
- [ ] History limited to configured size
- [ ] Rollback updates routing correctly
- [ ] Auto-rollback triggers on error threshold
- [ ] Rollback completes in < 100ms (routing only)
- [ ] Can't rollback first deployment (no previous)
- [ ] Thread-safe rollback operations
- [ ] Rollback history preserved across restarts (optional)

## Implementation Notes
- Store deployment records in memory (or persistent storage)
- Use GUIDs for deployment IDs
- Implement LRU for history size limiting
- Add detailed logging for rollback operations
- Consider adding deployment validation hooks (optional)
- Add notification hooks for rollback events (optional)
- Persist deployment records to file/database (optional)

## Performance Targets
- RecordDeployment: < 1ms
- RollbackAsync: < 100ms (excluding model load)
- GetDeploymentHistory: < 10ms
- MonitorErrorRate: < 0.1ms

## Edge Cases
- Rollback during ongoing swap
- Multiple error rate monitors for same version
- Deployment history full when recording new deployment
- Rollback to non-existent version
- Auto-rollback triggered while manual rollback in progress
