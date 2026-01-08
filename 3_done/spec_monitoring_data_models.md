# Spec: Monitoring Data Models

## Overview
Define data structures for collecting and comparing performance metrics across model versions, including alerts and metric comparisons.

## Tasks

### 1. Create VersionMetrics Class
**File:** `src/ModelVersioning/VersionMetrics.cs`

```csharp
public class VersionMetrics
{
    public string ModelId { get; set; }
    public string Version { get; set; }
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public int TotalRequests { get; set; }
    public double AverageLatencyMs { get; set; }
    public double P50LatencyMs { get; set; }
    public double P95LatencyMs { get; set; }
    public double P99LatencyMs { get; set; }
    public double ErrorRate { get; set; }
    public double Throughput { get; set; }
    public double MemoryUsageMB { get; set; }
}
```

### 2. Create MetricComparison Class
**File:** `src/ModelVersioning/MetricComparison.cs`

```csharp
public class MetricComparison
{
    public VersionMetrics Version1 { get; set; }
    public VersionMetrics Version2 { get; set; }
    public MetricDelta LatencyDelta { get; set; }
    public MetricDelta ErrorRateDelta { get; set; }
    public MetricDelta ThroughputDelta { get; set; }
    public DateTime ComparisonTime { get; set; }
}
```

### 3. Create MetricDelta Class
**File:** `src/ModelVersioning/MetricDelta.cs`

```csharp
public class MetricDelta
{
    public double AbsoluteDifference { get; set; }
    public double PercentageChange { get; set; }
    public string Direction { get; set; } // "better", "worse", "neutral"
}
```

### 4. Create VersionAlert Class
**File:** `src/ModelVersioning/VersionAlert.cs`

```csharp
public class VersionAlert
{
    public string AlertId { get; set; }
    public string ModelId { get; set; }
    public string Version { get; set; }
    public AlertType Type { get; set; }
    public string Message { get; set; }
    public DateTime Timestamp { get; set; }
    public Dictionary<string, object> Context { get; set; }
    public AlertSeverity Severity { get; set; }
}
```

### 5. Create AlertType Enum
**File:** `src/ModelVersioning/AlertType.cs`

```csharp
public enum AlertType
{
    HighLatency,
    HighErrorRate,
    LowThroughput,
    MemoryExceeded,
    AnomalyDetected
}
```

### 6. Create AlertSeverity Enum
**File:** `src/ModelVersioning/AlertSeverity.cs`

```csharp
public enum AlertSeverity
{
    Info,
    Warning,
    Critical
}
```

### 7. Create MetricSample Class
**File:** `src/ModelVersioning/MetricSample.cs`

```csharp
public class MetricSample
{
    public DateTime Timestamp { get; set; }
    public double LatencyMs { get; set; }
    public bool Success { get; set; }
    public double MemoryUsageMB { get; set; }
}
```

## Validation Requirements
- EndTime must be >= StartTime
- Latency and throughput values must be non-negative
- Error rate must be 0-1
- Alert must have valid type and severity
- Percentage change calculation

## Testing
**File:** `tests/ModelVersioning/MonitoringDataModelsTests.cs`

Create unit tests for:
1. VersionMetrics creation and aggregation
2. MetricComparison calculation (delta and percentage)
3. MetricDelta direction determination (better/worse)
4. VersionAlert creation for each alert type
5. AlertSeverity assignment
6. MetricSample recording
7. JSON serialization/deserialization
8. Invalid time range detection
9. Percentage change calculation edge cases
10. P50/P95/P99 percentile calculations

## Dependencies
- Spec: spec_model_data_models.md
- System.Text.Json
