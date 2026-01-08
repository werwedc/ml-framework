# Spec: Routing Data Models

## Overview
Define data structures for traffic routing policies, routing rules, request contexts, and version selection criteria.

## Tasks

### 1. Create RoutingPolicy Class
**File:** `src/ModelVersioning/RoutingPolicy.cs`

```csharp
public class RoutingPolicy
{
    public string ModelId { get; set; }
    public List<RoutingRule> Rules { get; set; }
    public RoutingMode Mode { get; set; }
    public DateTime EffectiveDate { get; set; }
}
```

### 2. Create RoutingMode Enum
**File:** `src/ModelVersioning/RoutingMode.cs`

```csharp
public enum RoutingMode
{
    Percentage,      // Route based on percentage split
    Shadow,          // Route to multiple, return from primary
    Deterministic,   // Route based on user/segment
    TimeBased        // Route based on schedule
}
```

### 3. Create RoutingRule Class
**File:** `src/ModelVersioning/RoutingRule.cs`

```csharp
public class RoutingRule
{
    public string Version { get; set; }
    public double Percentage { get; set; }
    public string UserIdPattern { get; set; }
    public string Segment { get; set; }
    public string Region { get; set; }
    public TimeRange TimeRange { get; set; }
    public bool IsPrimary { get; set; } // For shadow mode
}
```

### 4. Create TimeRange Class
**File:** `src/ModelVersioning/TimeRange.cs`

```csharp
public class TimeRange
{
    public TimeSpan StartTime { get; set; }
    public TimeSpan EndTime { get; set; }
    public DayOfWeek[] DaysOfWeek { get; set; }
}
```

### 5. Create RequestContext Class
**File:** `src/ModelVersioning/RequestContext.cs`

```csharp
public class RequestContext
{
    public string UserId { get; set; }
    public string Segment { get; set; }
    public string Region { get; set; }
    public DateTime RequestTime { get; set; }
    public Dictionary<string, string> Metadata { get; set; }
}
```

### 6. Create RoutingResult Class
**File:** `src/ModelVersioning/RoutingResult.cs`

```csharp
public class RoutingResult
{
    public string Version { get; set; }
    public bool IsShadow { get; set; }
    public List<string> ShadowVersions { get; set; }
    public string RuleMatched { get; set; }
}
```

## Validation Requirements
- Percentage values must be 0-100
- Total percentages in policy must equal 100 (for Percentage mode)
- TimeRange must have valid start/end times
- Required fields based on RoutingMode

## Testing
**File:** `tests/ModelVersioning/RoutingDataModelsTests.cs`

Create unit tests for:
1. RoutingPolicy creation with multiple rules
2. RoutingRule validation (percentage bounds)
3. TimeRange validation (start < end)
4. RequestContext creation with metadata
5. RoutingResult construction
6. JSON serialization/deserialization
7. Different RoutingMode configurations
8. Percentage validation (sum = 100)
9. Shadow mode rule configuration
10. Time-based rule configuration

## Dependencies
- Spec: spec_model_data_models.md
- System.Text.Json
- System.ComponentModel.DataAnnotations
