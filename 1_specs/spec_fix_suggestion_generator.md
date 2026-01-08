# Spec: Suggested Fix Generator

## Overview
Implement pattern-based suggestion system to provide actionable fixes for common shape mismatch scenarios.

## Requirements

### Class: FixSuggestionGenerator
- Location: `src/Diagnostics/FixSuggestionGenerator.cs`

```csharp
public class FixSuggestionGenerator
{
    public List<string> GenerateSuggestions(
        OperationType operationType,
        long[][] inputShapes,
        long[][] expectedShapes)
}
```

### Common Pattern Detection

**Pattern 1: Missing Batch Dimension**
```csharp
// Input: [784], Expected: [*, 784]
// Suggestion: "Input missing batch dimension. Add unsqueeze(0) or reshape to [1, 784]"
```

**Pattern 2: Channel Order Mismatch**
```csharp
// Input: [32, 224, 224, 3], Expected: [32, 3, 224, 224]
// Suggestion: "Channel order mismatch. Permute from NHWC to NCHW or vice versa"
```

**Pattern 3: Feature Size Mismatch in Linear Layer**
```csharp
// Input: [32, 256], Weight: [128, 10]
// Suggestion: "Previous layer outputs 256 features, but this layer expects 128. Adjust layer configuration"
```

**Pattern 4: Transpose Required**
```csharp
// Input: [10, 32], Weight: [10, 64]
// Suggestion: "Consider transposing input to [32, 10] or weight to [64, 10]"
```

**Pattern 5: Concatenation Dimension Mismatch**
```csharp
// Input1: [32, 128], Input2: [32, 256], Axis: 0
// Suggestion: "Cannot concatenate on axis 0 with different sizes. Use axis 1 or reshape inputs"
```

**Pattern 6: Broadcasting Failure**
```csharp
// Input1: [32, 10], Input2: [20, 10]
// Suggestion: "Cannot broadcast shapes [32, 10] and [20, 10]. Batch sizes must match or be 1"
```

### Suggestion Templates

Create a suggestion template system in `src/Diagnostics/SuggestionTemplates.cs`:

```csharp
public static class SuggestionTemplates
{
    public const string MissingBatchDim = "Input missing batch dimension. Add unsqueeze(0) or reshape to [1, {0}]";
    public const string ChannelOrderMismatch = "Channel order mismatch. Permute from {0} to {1}";
    public const string FeatureSizeMismatch = "Previous layer outputs {0} features, but this layer expects {1}. Adjust layer configuration";
    public const string TransposeRequired = "Consider transposing {0} to {1}";
    // ... more templates
}
```

### Pattern Matching Logic
```csharp
private List<string> DetectPatterns(
    OperationType operationType,
    long[][] inputShapes,
    long[][] expectedShapes)
{
    var suggestions = new List<string>();

    // Check for missing batch dimension
    if (inputShapes[0].Length + 1 == expectedShapes[0].Length)
    {
        suggestions.Add(string.Format(SuggestionTemplates.MissingBatchDim,
            string.Join(", ", inputShapes[0])));
    }

    // Check for channel order mismatch
    if (IsChannelOrderMismatch(inputShapes[0], expectedShapes[0]))
    {
        suggestions.Add(string.Format(SuggestionTemplates.ChannelOrderMismatch,
            "NHWC", "NCHW"));
    }

    // ... more pattern checks

    return suggestions;
}
```

## Tests
- Create `tests/Diagnostics/FixSuggestionGeneratorTests.cs`
- Test each pattern detection with various inputs
- Test suggestion templates formatting
- Test edge cases (no pattern match, multiple patterns)
- Test with different operation types

## Success Criteria
- [ ] FixSuggestionGenerator class implemented
- [ ] Pattern detection for 6+ common scenarios
- [ ] Suggestion templates for reusability
- [ ] Actionable and accurate suggestions
- [ ] Unit tests pass for all patterns
