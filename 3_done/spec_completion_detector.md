# Spec: Completion Detector for Continuous Batching

## Overview
Implement the completion detection logic that determines when a request has finished generating. The detector supports multiple completion conditions (EOS, max tokens, stop strings, length) and provides detailed completion reasons for monitoring and logging.

## Class: CompletionDetector
```csharp
public class CompletionDetector
{
    private readonly CompletionConfiguration _config;
    private readonly ITokenizer _tokenizer;

    public CompletionDetector(CompletionConfiguration config,
                              ITokenizer tokenizer)
    {
        _config = config;
        _tokenizer = tokenizer;
    }

    // Check if request is completed
    public (bool IsCompleted, CompletionReason Reason) CheckCompletion(
        Request request);

    // Check specific completion conditions
    private bool CheckEosToken(Request request);
    private bool CheckMaxTokens(Request request);
    private bool CheckStopString(Request request);
    private bool CheckLengthConstraint(Request request);
    private bool CheckCancellation(Request request);
}
```

---

## Class: CompletionConfiguration
```csharp
public record class CompletionConfiguration(
    int EosTokenId,                    // End-of-sequence token ID
    int DefaultMaxTokens,              // Default max tokens if not specified
    List<string>? StopStrings,         // Strings that stop generation
    int? MaxResponseLength,            // Max response length in characters
    bool EnableEarlyStopping,          // Stop if confidence low
    double EarlyStoppingThreshold      // Confidence threshold
)
{
    public static readonly CompletionConfiguration Default = new(
        EosTokenId: 2,  // Common EOS token ID
        DefaultMaxTokens: 256,
        StopStrings: null,
        MaxResponseLength: null,
        EnableEarlyStopping: false,
        EarlyStoppingThreshold: 0.01
    );
}
```

**Purpose**: Configurable completion detection parameters.

---

## Interface: ITokenizer
```csharp
public interface ITokenizer
{
    int[] Encode(string text);
    string Decode(int[] tokenIds);
    int GetVocabSize();
}
```

**Purpose**: Tokenization interface for string operations.

---

## Implementation Details

### Main Completion Check
```csharp
public (bool IsCompleted, CompletionReason Reason) CheckCompletion(
    Request request)
{
    // Check cancellation first (highest priority)
    if (CheckCancellation(request))
    {
        return (true, CompletionReason.Cancelled);
    }

    // Check max tokens
    if (CheckMaxTokens(request))
    {
        return (true, CompletionReason.MaxTokensReached);
    }

    // Check EOS token
    if (CheckEosToken(request))
    {
        return (true, CompletionReason.EosTokenReached);
    }

    // Check stop strings
    if (CheckStopString(request))
    {
        return (true, CompletionReason.StopString);
    }

    // Check length constraint
    if (CheckLengthConstraint(request))
    {
        return (true, CompletionReason.LengthReached);
    }

    return (false, CompletionReason.EosTokenReached); // Default reason
}
```

**Requirements**:
- Check conditions in priority order
- Return completion status and reason
- Return accurate reason for logging

---

### CheckEosToken
```csharp
private bool CheckEosToken(Request request)
{
    if (request.GeneratedTokenIds.Count == 0)
        return false;

    int lastToken = request.GeneratedTokenIds[^1];
    return lastToken == _config.EosTokenId;
}
```

**Requirements**:
- Only check if tokens have been generated
- Compare last token with configured EOS ID

---

### CheckMaxTokens
```csharp
private bool CheckMaxTokens(Request request)
{
    int maxTokens = request.MaxTokens > 0
        ? request.MaxTokens
        : _config.DefaultMaxTokens;

    return request.GeneratedTokens >= maxTokens;
}
```

**Requirements**:
- Use request's max tokens if specified
- Fall back to default configuration
- Check if count exceeded

---

### CheckStopString
```csharp
private bool CheckStopString(Request request)
{
    if (_config.StopStrings == null || _config.StopStrings.Count == 0)
        return false;

    if (request.GeneratedTokenIds.Count == 0)
        return false;

    // Decode recent tokens to check for stop strings
    string generatedText = _tokenizer.Decode(
        request.GeneratedTokenIds.ToArray()
    );

    foreach (var stopString in _config.StopStrings)
    {
        if (generatedText.Contains(stopString))
        {
            // Remove stop string from generated text
            int stopIndex = generatedText.IndexOf(stopString);
            if (stopIndex >= 0)
            {
                // Truncate at stop string
                // Note: In production, need to update GeneratedTokenIds too
            }
            return true;
        }
    }

    return false;
}
```

**Requirements**:
- Check all configured stop strings
- Decode tokens to text for comparison
- Handle stop string removal (future enhancement)
- Return true if any stop string found

---

### CheckLengthConstraint
```csharp
private bool CheckLengthConstraint(Request request)
{
    if (_config.MaxResponseLength == null)
        return false;

    if (request.GeneratedTokenIds.Count == 0)
        return false;

    string generatedText = _tokenizer.Decode(
        request.GeneratedTokenIds.ToArray()
    );

    return generatedText.Length >= _config.MaxResponseLength.Value;
}
```

**Requirements**:
- Only check if max length configured
- Decode tokens to text
- Compare character count

---

### CheckCancellation
```csharp
private bool CheckCancellation(Request request)
{
    return request.CancellationToken.IsCancellationRequested;
}
```

**Requirements**:
- Check cancellation token state
- Fast path for frequently-called method

---

## Files to Create

### Implementation
- `src/MLFramework/Inference/ContinuousBatching/Scheduler/CompletionDetector.cs`
- `src/MLFramework/Inference/ContinuousBatching/Scheduler/CompletionConfiguration.cs`

### Tests
- `tests/MLFramework.Tests/Inference/ContinuousBatching/CompletionDetectorTests.cs`

---

## Dependencies
- `spec_continuous_batching_models.md` (Request, CompletionReason)

---

## Testing Requirements

### Unit Tests (with Mock ITokenizer)
1. **EOS Token Detection**:
   - Detect when last token is EOS
   - Not complete when last token is not EOS
   - Handle empty generated tokens list

2. **Max Tokens Detection**:
   - Detect when max tokens reached
   - Use request-specific max tokens
   - Fall back to default max tokens
   - Not complete before limit

3. **Stop String Detection**:
   - Detect when stop string appears
   - Handle multiple stop strings
   - Case-sensitive matching
   - Return first matched reason

4. **Length Constraint Detection**:
   - Detect when character limit reached
   - Decode tokens correctly
   - Not complete before limit

5. **Cancellation Detection**:
   - Detect cancelled requests
   - Return correct reason

6. **Priority Order**:
   - Cancellation checked first
   - Max tokens checked before EOS
   - EOS checked before stop strings
   - All conditions checked in order

7. **Edge Cases**:
   - Empty configuration (use defaults)
   - No stop strings configured
   - Max length not configured
   - Zero max tokens

---

## Success Criteria
- [ ] All completion conditions implemented
- [ ] Priority order correct
- [ ] Completion reasons accurate
- [ ] Tokenizer integration tested (with mocks)
- [ ] Unit tests cover all scenarios
- [ ] Performance acceptable (fast path for common checks)
