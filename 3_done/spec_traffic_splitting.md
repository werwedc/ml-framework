# Spec: Traffic Splitting Engine

## Purpose
Implement percentage-based traffic distribution between multiple model versions for A/B testing and canary deployments.

## Technical Requirements

### Core Functionality
- Configure traffic splits as percentage allocations
- Route requests based on weighted random selection
- Validate percentages sum to 100%
- Support dynamic updates to traffic splits
- Thread-safe traffic distribution

### Data Structures
```csharp
public interface ITrafficSplitter
{
    void SetTrafficSplit(string modelName, Dictionary<string, float> versionPercentages);
    string SelectVersion(string modelName, string requestId);
    Dictionary<string, float> GetTrafficSplit(string modelName);
    void ClearTrafficSplit(string modelName);
    float GetVersionAllocation(string modelName, string version);
}

public class TrafficSplitConfig
{
    public Dictionary<string, float> VersionPercentages { get; set; }
    public DateTime LastUpdated { get; set; }
    public string UpdatedBy { get; set; }
}
```

### Selection Algorithm
- Generate consistent hash or random value from requestId
- Map hash value to percentage ranges
- Example: v1.0=70%, v1.1=30%
  - Hash [0, 0.70) -> v1.0
  - Hash [0.70, 1.0) -> v1.1

## Dependencies
- `spec_model_version_registry.md` (to validate versions exist)

## Testing Requirements
- Set traffic split 70/30, verify distribution over 1000 requests (~700/300)
- Attempt to set split that doesn't sum to 1.0 (should throw)
- Set split for non-existent model (should throw)
- Set split with negative percentage (should throw)
- Update split from 70/30 to 50/50, verify new distribution
- Same request ID produces same version (deterministic routing)
- Clear split, verify it's removed
- Concurrent updates test (10 threads setting splits)
- Performance test: 1000+ selections per ms

## Success Criteria
- [ ] Traffic distribution matches configured percentages within 2% error
- [ ] Same request ID consistently routes to same version
- [ ] Validates percentages sum to 1.0
- [ ] Thread-safe under high concurrency
- [ ] Selection performance < 0.1ms per request
- [ ] Can handle 10+ version splits

## Implementation Notes
- Use deterministic hashing (e.g., xxHash, MurmurHash) on request ID
- Use thread-safe data structures for config storage
- Add validation for percentage ranges (0-1.0)
- Consider using atomic snapshots for split updates
- Add logging for split configuration changes
- Cache computed percentage ranges for faster lookups

## Performance Targets
- SelectVersion latency: < 0.1ms
- Support 100,000+ selections per second
- SetTrafficSplit: < 1ms (even with 10+ versions)

## Edge Cases
- Single version split (100%)
- Many versions split (10+)
- Very small percentages (< 1%)
- Split updates during active routing
