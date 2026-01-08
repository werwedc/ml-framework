# Spec: Version Router Implementation

## Overview
Implement IVersionRouter for flexible request routing based on user segments, percentages, metadata, and time-based schedules.

## Tasks

### 1. Create IVersionRouter Interface
**File:** `src/ModelVersioning/IVersionRouter.cs`

```csharp
public interface IVersionRouter
{
    void SetRoutingPolicy(RoutingPolicy policy);
    RoutingResult RouteRequest(RequestContext context);
    void UpdatePolicy(RoutingPolicy newPolicy);
    RoutingPolicy GetCurrentPolicy();
}
```

### 2. Implement VersionRouter Class
**File:** `src/ModelVersioning/VersionRouter.cs`

```csharp
public class VersionRouter : IVersionRouter
{
    private RoutingPolicy _currentPolicy;
    private readonly object _policyLock;
    private readonly Random _hashGenerator;

    public VersionRouter()
    {
        _policyLock = new object();
        _hashGenerator = new Random();
    }

    public void SetRoutingPolicy(RoutingPolicy policy)
    {
        // Validate policy
        // Set new policy atomically
    }

    public RoutingResult RouteRequest(RequestContext context)
    {
        // Route based on policy mode
    }

    public void UpdatePolicy(RoutingPolicy newPolicy)
    {
        // Validate and update
    }

    public RoutingPolicy GetCurrentPolicy()
    {
        // Return current policy
    }
}
```

### 3. Implement Routing Logic by Mode

#### Percentage Mode
- Use deterministic hashing on user ID or random selection
- Match against percentage boundaries
- Return version based on hash value

#### Shadow Mode
- Route to primary version for response
- Route to shadow versions for comparison
- Mark shadow versions in result

#### Deterministic Mode
- Match user ID pattern (regex)
- Match segment
- Match region
- Return matching version

#### Time-Based Mode
- Check request time against TimeRange
- Check day of week
- Return matching version

### 4. Implement Policy Validation
- Total percentage must equal 100 (Percentage mode)
- At least one rule required
- All versions must exist (registry check optional)
- Time ranges must be valid
- Primary version required for Shadow mode

## Validation
- Policy validation before setting
- Atomic policy updates (thread-safe)
- Hash consistency for same user

## Testing
**File:** `tests/ModelVersioning/VersionRouterTests.cs`

Create unit tests for:
1. Percentage-based routing (90/10 split)
2. Shadow mode routing
3. Deterministic routing by user segment
4. Deterministic routing by region
5. Time-based routing
6. Policy validation (percentage sum = 100)
7. Policy validation (invalid time range)
8. Concurrent policy updates
9. Atomic policy switching
10. Consistent routing for same user
11. Policy update without losing in-flight requests

## Dependencies
- Spec: spec_model_data_models.md
- Spec: spec_routing_data_models.md
- Thread synchronization
