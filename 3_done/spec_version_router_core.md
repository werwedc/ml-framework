# Spec: Version Router Core

## Purpose
Implement core routing logic to direct inference requests to specific model versions based on routing context.

## Technical Requirements

### Core Functionality
- Route requests to specific model version
- Default to latest version if not specified
- Validate requested version exists
- Handle routing errors gracefully
- Support routing context for decision making

### Data Structures
```csharp
public class RoutingContext
{
    public string PreferredVersion { get; set; }
    public Dictionary<string, string> Headers { get; set; }
    public string ExperimentId { get; set; }
    public string UserId { get; set; }
}

public interface IVersionRouter
{
    IModel GetModel(string modelName, RoutingContext context);
    IModel GetModel(string modelName, string version);
    void SetDefaultVersion(string modelName, string version);
    string GetDefaultVersion(string modelName);
}

public class RoutingException : Exception
{
    public string ModelName { get; }
    public string RequestedVersion { get; }
}
```

### Routing Logic
1. Check if context specifies PreferredVersion
2. If yes, validate version exists and return
3. If no, use default version for model
4. If default not set, use latest version (by semantic version)
5. If no versions exist, throw RoutingException

## Dependencies
- `spec_model_version_registry.md` (to query available versions)
- `spec_model_loader.md` (to retrieve loaded models)

## Testing Requirements
- Route with explicit version, verify correct model returned
- Route without version, verify default version used
- Route without version and no default, verify latest used
- Route to non-existent version, throw RoutingException
- Route to model with no versions, throw RoutingException
- Set default version, verify it's used
- Test routing context with headers
- Performance test: 1000+ routes under 1ms

## Success Criteria
- [ ] Routes to correct model version based on context
- [ ] Falls back to default/latest version appropriately
- [ ] Throws meaningful exceptions for invalid requests
- [ ] Routing performance < 1ms per request
- [ ] Thread-safe under concurrent routing

## Implementation Notes
- Use dependency injection to access ModelRegistry and ModelLoader
- Implement semantic version comparison for "latest" logic
- Cache default versions to avoid repeated queries
- Add detailed logging for routing decisions
- Consider adding routing metrics (optional future enhancement)

## Performance Targets
- GetModel latency: < 1ms (for cached routes)
- Support 10,000+ routing decisions per second
