# Spec: Integration Tests

## Overview
Create comprehensive integration tests that verify the entire model versioning and hot-swapping system works end-to-end.

## Tasks

### 1. Create Test Setup Infrastructure
**File:** `tests/ModelVersioning/IntegrationTestFixture.cs`

```csharp
public class IntegrationTestFixture : IDisposable
{
    public IModelRegistry Registry { get; }
    public IModelVersionManager VersionManager { get; }
    public IVersionRouter Router { get; }
    public IVersionMonitor Monitor { get; }
    public IModelHotSwapper HotSwapper { get; }

    public IntegrationTestFixture()
    {
        // Setup all services with DI
        // Initialize with test data
    }

    public void Dispose()
    {
        // Cleanup
    }
}
```

### 2. Create Full Workflow Test
**File:** `tests/ModelVersioning/IntegrationTests.cs`

```csharp
public class IntegrationTests : IClassFixture<IntegrationTestFixture>
{
    private readonly IntegrationTestFixture _fixture;

    public IntegrationTests(IntegrationTestFixture fixture)
    {
        _fixture = fixture;
    }

    [Fact]
    public async Task FullWorkflow_RegisterRouteSwapRollback()
    {
        // 1. Register two model versions
        // 2. Set up routing policy (90/10 split)
        // 3. Record metrics for both versions
        // 4. Compare version performance
        // 5. Hot-swap to better version
        // 6. Verify routing changed
        // 7. Rollback if needed
        // 8. Verify rollback succeeded
    }
}
```

### 3. Test Complete A/B Testing Scenario
**File:** `tests/ModelVersioning/ABTestingIntegrationTests.cs`

```csharp
public class ABTestingIntegrationTests : IClassFixture<IntegrationTestFixture>
{
    [Fact]
    public async Task ABTesting_CompareAndPromote()
    {
        // 1. Register v1.0.0 and v2.0.0
        // 2. Set up 90/10 A/B test
        // 3. Simulate traffic (100 requests)
        // 4. Record metrics for each version
        // 5. Compare results
        // 6. Promote winner to 100%
        // 7. Verify traffic routing
    }
}
```

### 4. Test Canary Deployment Scenario
**File:** `tests/ModelVersioning/CanaryDeploymentTests.cs`

```csharp
public class CanaryDeploymentTests : IClassFixture<IntegrationTestFixture>
{
    [Fact]
    public async Task Canary_GradualRampup()
    {
        // 1. Start with 5% traffic to new version
        // 2. Monitor metrics
        // 3. Gradually increase to 10%, 25%, 50%
        // 4. Check health at each stage
        // 5. Rollback if metrics degrade
        // 6. Complete rollout if metrics good
    }
}
```

### 5. Test Shadow Mode Scenario
**File:** `tests/ModelVersioning/ShadowModeTests.cs`

```csharp
public class ShadowModeTests : IClassFixture<IntegrationTestFixture>
{
    [Fact]
    public void ShadowMode_CompareOutputs()
    {
        // 1. Set up shadow mode with primary and shadow version
        // 2. Send requests
        // 3. Verify only primary responses returned
        // 4. Verify shadow version received traffic
        // 5. Compare metrics between versions
    }
}
```

### 6. Test Emergency Rollback Scenario
**File:** `tests/ModelVersioning/EmergencyRollbackTests.cs`

```csharp
public class EmergencyRollbackTests : IClassFixture<IntegrationTestFixture>
{
    [Fact]
    public async Task EmergencyRollback_OnHighErrorRate()
    {
        // 1. Deploy new version
        // 2. Simulate high error rate on new version
        // 3. Trigger alert
        // 4. Execute rollback
        // 5. Verify rollback completed < 30 seconds
        // 6. Verify traffic restored to old version
    }
}
```

### 7. Test Multi-Version Serving
**File:** `tests/ModelVersioning/MultiVersionServingTests.cs`

```csharp
public class MultiVersionServingTests : IClassFixture<IntegrationTestFixture>
{
    [Fact]
    public void MultipleVersions_ServeConcurrently()
    {
        // 1. Load 3 versions simultaneously
        // 2. Route to all 3 versions
        // 3. Verify no version interference
        // 4. Verify memory management
        // 5. Unload one version
        // 6. Verify others continue serving
    }
}
```

### 8. Test Time-Based Routing
**File:** `tests/ModelVersioning/TimeBasedRoutingTests.cs`

```csharp
public class TimeBasedRoutingTests : IClassFixture<IntegrationTestFixture>
{
    [Fact]
    public void TimeBased_RouteBySchedule()
    {
        // 1. Set up time-based routing rules
        // 2. Verify routing by time of day
        // 3. Verify routing by day of week
        // 4. Test time boundary transitions
    }
}
```

### 9. Test Version Lineage Tracking
**File:** `tests/ModelVersioning/VersionLineageTests.cs`

```csharp
public class VersionLineageTests : IClassFixture<IntegrationTestFixture>
{
    [Fact]
    public void Lineage_TrackParentChild()
    {
        // 1. Register base model
        // 2. Register fine-tuned model with parent
        // 3. Verify lineage tracking
        // 4. Query models by lineage
    }
}
```

### 10. Test Memory Management
**File:** `tests/ModelVersioning/MemoryManagementTests.cs`

```csharp
public class MemoryManagementTests : IClassFixture<IntegrationTestFixture>
{
    [Fact]
    public void Memory_ManageMultipleVersions()
    {
        // 1. Load multiple versions
        // 2. Verify memory tracking
        // 3. Unload versions
        // 4. Verify memory freed
        // 5. Test memory limits
    }
}
```

## Test Data Requirements
- Create mock model files for testing
- Create test metadata with various values
- Create diverse routing scenarios
- Simulate various metric patterns

## Test Coverage Goals
- All integration points between components
- End-to-end workflows
- Error scenarios and recovery
- Performance requirements (< 1 min swap, < 30 sec rollback)
- Concurrent operations

## Dependencies
- All previous specs must be completed first
- xUnit test framework
- Moq for mocking external dependencies
