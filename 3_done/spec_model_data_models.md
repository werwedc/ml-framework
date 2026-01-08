# Spec: Core Data Models for Model Versioning

## Overview
Define the core data structures for model versioning, including model metadata, version information, lifecycle states, and related data transfer objects.

## Tasks

### 1. Create ModelMetadata Class
**File:** `src/ModelVersioning/ModelMetadata.cs`

```csharp
public class ModelMetadata
{
    public DateTime CreationTimestamp { get; set; }
    public Dictionary<string, object> TrainingParameters { get; set; }
    public PerformanceMetrics Performance { get; set; }
    public string DatasetVersion { get; set; }
    public string ArchitectureHash { get; set; }
    public Dictionary<string, string> CustomMetadata { get; set; }
}
```

### 2. Create PerformanceMetrics Class
**File:** `src/ModelVersioning/PerformanceMetrics.cs`

```csharp
public class PerformanceMetrics
{
    public float Accuracy { get; set; }
    public float LatencyMs { get; set; }
    public float Throughput { get; set; }
    public float MemoryUsageMB { get; set; }
}
```

### 3. Create LifecycleState Enum
**File:** `src/ModelVersioning/LifecycleState.cs`

```csharp
public enum LifecycleState
{
    Draft,
    Staging,
    Production,
    Archived
}
```

### 4. Create ModelInfo Class
**File:** `src/ModelVersioning/ModelInfo.cs`

```csharp
public class ModelInfo
{
    public string ModelId { get; set; }
    public string Name { get; set; }
    public string VersionTag { get; set; }
    public ModelMetadata Metadata { get; set; }
    public LifecycleState State { get; set; }
    public string ParentModelId { get; set; } // For parent-child relationships
}
```

### 5. Create HealthCheckResult Class
**File:** `src/ModelVersioning/HealthCheckResult.cs`

```csharp
public class HealthCheckResult
{
    public bool IsHealthy { get; set; }
    public string Message { get; set; }
    public DateTime CheckTimestamp { get; set; }
    public Dictionary<string, object> Diagnostics { get; set; }
}
```

## Validation Requirements
- All properties should be nullable where appropriate
- Add validation attributes (e.g., [Range] for Accuracy 0-1)
- Implement ToString() for debugging
- Add JSON serialization attributes where needed

## Testing
**File:** `tests/ModelVersioning/DataModelsTests.cs`

Create unit tests for:
1. ModelMetadata creation and property assignment
2. PerformanceMetrics validation (accuracy must be 0-1)
3. ModelInfo state transitions
4. HealthCheckResult success/failure scenarios
5. JSON serialization/deserialization

## Dependencies
- System.Text.Json for serialization
- System.ComponentModel.DataAnnotations for validation
