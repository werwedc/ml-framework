# Spec: Continuous Batching - Core Data Models

## Overview
Define the fundamental data structures and models required for continuous batching implementation. These models will be used across the scheduler, request queue, and execution engine.

## Class: RequestId
```csharp
public record struct RequestId(Guid Id)
{
    public static RequestId New() => new(Guid.NewGuid());
    public static RequestId Empty => new(Guid.Empty);
}
```

**Purpose**: Unique identifier for each inference request.
**Requirements**:
- Lightweight struct to minimize memory overhead
- Easy generation and comparison
- Default empty state for initialization

---

## Class: Request
```csharp
public class Request
{
    public RequestId Id { get; }
    public string Prompt { get; }
    public int MaxTokens { get; }
    public CancellationToken CancellationToken { get; }
    public TaskCompletionSource<string> CompletionSource { get; }
    public DateTime EnqueuedTime { get; }
    public Priority Priority { get; }
    public int GeneratedTokens { get; set; }
    public bool IsCompleted { get; set; }
    public List<int> GeneratedTokenIds { get; }

    public Request(RequestId id, string prompt, int maxTokens, 
                   CancellationToken token, Priority priority = Priority.Normal)
    {
        Id = id;
        Prompt = prompt;
        MaxTokens = maxTokens;
        CancellationToken = token;
        CompletionSource = new TaskCompletionSource<string>();
        EnqueuedTime = DateTime.UtcNow;
        Priority = priority;
        GeneratedTokens = 0;
        IsCompleted = false;
        GeneratedTokenIds = new List<int>();
    }
}
```

**Properties**:
- `Id`: Unique identifier
- `Prompt`: Input text for generation
- `MaxTokens`: Maximum tokens to generate
- `CancellationToken`: For cancellation support
- `CompletionSource`: Task for async completion
- `EnqueuedTime`: Tracking for priority/fairness
- `Priority`: Priority level for scheduling
- `GeneratedTokens`: Count of tokens generated so far
- `IsCompleted`: Completion flag
- `GeneratedTokenIds`: Accumulated token IDs

**Requirements**:
- Thread-safe property access (where needed)
- Support cancellation via CancellationToken
- Track generation progress

---

## Enum: Priority
```csharp
public enum Priority
{
    Low = 0,
    Normal = 1,
    High = 2,
    Urgent = 3
}
```

**Purpose**: Define request priority levels for scheduler.

---

## Class: Batch
```csharp
public class Batch
{
    public int BatchId { get; }
    public List<Request> Requests { get; }
    public DateTime CreatedTime { get; }
    public int Size => Requests.Count;
    public long EstimatedMemoryBytes { get; set; }

    public Batch(int batchId)
    {
        BatchId = batchId;
        Requests = new List<Request>();
        CreatedTime = DateTime.UtcNow;
        EstimatedMemoryBytes = 0;
    }

    public void AddRequest(Request request) { /* implementation */ }
    public void RemoveRequest(RequestId requestId) { /* implementation */ }
    public bool Contains(RequestId requestId) { /* implementation */ }
    public Request? GetRequest(RequestId requestId) { /* implementation */ }
}
```

**Properties**:
- `BatchId`: Sequential identifier for the batch
- `Requests`: List of active requests in batch
- `CreatedTime`: For tracking batch lifetime
- `Size`: Current number of requests
- `EstimatedMemoryBytes`: Memory estimate for capacity management

**Methods**:
- `AddRequest`: Add a request to batch
- `RemoveRequest`: Remove a request by ID
- `Contains`: Check if request is in batch
- `GetRequest`: Retrieve request by ID

**Requirements**:
- Efficient add/remove operations
- Track memory usage for capacity management

---

## Class: CompletionReason
```csharp
public enum CompletionReason
{
    EosTokenReached,      // End-of-sequence token generated
    MaxTokensReached,     // Max generation limit reached
    Cancelled,            // Request cancelled by client
    LengthReached,        // Custom length condition met
    StopString,           // Stop string condition met
    Timeout               // Request timed out
}
```

**Purpose**: Enumerate reasons why a request completed.

---

## Class: RequestResult
```csharp
public record class RequestResult(
    RequestId RequestId,
    string GeneratedText,
    int TokensGenerated,
    CompletionReason Reason,
    TimeSpan ProcessingTime
);
```

**Purpose**: Structured result of a completed request.
**Properties**:
- `RequestId`: Associated request ID
- `GeneratedText`: Full generated text
- `TokensGenerated`: Number of tokens generated
- `Reason`: Completion reason
- `ProcessingTime`: Total processing duration

---

## Class: BatchStats
```csharp
public record class BatchStats(
    int BatchId,
    int RequestCount,
    long MemoryBytesUsed,
    double UtilizationPercentage,
    TimeSpan ProcessingTime
);
```

**Purpose**: Statistics for monitoring batch performance.

---

## Implementation Notes

### Files to Create
- `src/MLFramework/Inference/ContinuousBatching/Models/RequestId.cs`
- `src/MLFramework/Inference/ContinuousBatching/Models/Request.cs`
- `src/MLFramework/Inference/ContinuousBatching/Models/Priority.cs`
- `src/MLFramework/Inference/ContinuousBatching/Models/Batch.cs`
- `src/MLFramework/Inference/ContinuousBatching/Models/CompletionReason.cs`
- `src/MLFramework/Inference/ContinuousBatching/Models/RequestResult.cs`
- `src/MLFramework/Inference/ContinuousBatching/Models/BatchStats.cs`

### Dependencies
- None (pure data models)

### Testing Requirements
- Unit tests for Request creation and state management
- Unit tests for Batch add/remove operations
- Verify RequestId uniqueness and equality
- Test completion reason enum values

---

## Success Criteria
- [ ] All model classes compile
- [ ] Models support required properties and methods
- [ ] Thread-safety where needed (concurrent access to Request)
- [ ] Unit tests pass for all models
- [ ] Memory overhead is minimal (especially for frequently-used structs)
