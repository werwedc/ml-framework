# Spec: Header-Based Routing

## Purpose
Enable routing decisions based on HTTP headers, supporting fine-grained control over model version selection.

## Technical Requirements

### Core Functionality
- Define routing rules based on header patterns
- Support exact match, prefix match, regex match
- Combine with traffic splitting for hybrid routing
- Priority-based rule evaluation
- Rule validation and management

### Data Structures
```csharp
public enum MatchType
{
    Exact,
    Prefix,
    Regex,
    Contains
}

public class RoutingRule
{
    public string HeaderName { get; set; }
    public string HeaderValue { get; set; }
    public MatchType MatchType { get; set; }
    public string TargetVersion { get; set; }
    public int Priority { get; set; }
    public string Description { get; set; }
}

public interface IHeaderRouter
{
    void RegisterRoutingRule(string modelName, RoutingRule rule);
    void UnregisterRoutingRule(string modelName, string ruleId);
    string RouteByHeaders(string modelName, RoutingContext context);
    IEnumerable<RoutingRule> GetRules(string modelName);
    void ClearRules(string modelName);
}

public interface IEnhancedVersionRouter : IVersionRouter
{
    void RegisterRoutingRule(string modelName, RoutingRule rule);
    void SetTrafficSplit(string modelName, Dictionary<string, float> versionPercentages);
}
```

### Routing Logic
1. Check if context.Headers contains rule header
2. Evaluate rules by priority (highest first)
3. Apply match type (exact, prefix, regex)
4. If rule matches, return target version
5. If no rules match, fall back to traffic splitting or default

## Dependencies
- `spec_version_router_core.md` (extends IVersionRouter)
- `spec_traffic_splitting.md` (used as fallback)

## Testing Requirements
- Register exact match rule, verify it routes matching requests
- Register prefix match rule (X-Feature: beta), verify it routes prefixes
- Register regex rule, verify pattern matching works
- Register multiple rules with priorities, verify correct order
- Rule with non-existent version (should throw)
- Rule with invalid regex (should throw)
- Same request with matching higher priority rule takes precedence
- Clear rules, verify they're removed
- Concurrent rule registration test
- Performance test: 1000+ header-based routes per ms

## Success Criteria
- [ ] Rules evaluate in priority order
- [ ] All match types (exact, prefix, regex) work correctly
- [ ] Falls back to traffic splitting if no rules match
- [ ] Validates target version exists
- [ ] Validates regex patterns
- [ ] Thread-safe rule management
- [ ] Routing performance < 1ms per request
- [ ] Can handle 100+ routing rules

## Implementation Notes
- Use `System.Text.RegularExpressions` for regex matching
- Cache compiled regex for performance
- Use `ConcurrentDictionary` for thread-safe rule storage
- Add rule validation on registration
- Consider adding rule versioning (optional)
- Log rule matches for debugging

## Performance Targets
- RouteByHeaders latency: < 1ms
- RegisterRoutingRule: < 1ms
- Support 1000+ rules per model
- Evaluate 10+ rules per request in < 1ms

## Edge Cases
- Headers not present in request
- Multiple rules matching same header
- Regex exceptions during evaluation
- Rule updates during active routing
