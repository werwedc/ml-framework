# Spec: Prefill Handler for Continuous Batching

## Overview
Implement the prefill handler responsible for processing new request prompts before they join the active batch. The prefill phase handles prompt tokenization, initial model forward pass, and KV cache population.

## Class: PrefillHandler
```csharp
public class PrefillHandler
{
    private readonly IModelExecutor _modelExecutor;
    private readonly ITokenizer _tokenizer;
    private readonly ContinuousBatchKVCacheManager _kvCacheManager;
    private readonly PrefillConfiguration _config;
    private readonly ILogger _logger;

    public PrefillHandler(
        IModelExecutor modelExecutor,
        ITokenizer tokenizer,
        ContinuousBatchKVCacheManager kvCacheManager,
        PrefillConfiguration config,
        ILogger logger)
    {
        _modelExecutor = modelExecutor;
        _tokenizer = tokenizer;
        _kvCacheManager = kvCacheManager;
        _config = config;
        _logger = logger;
    }

    // Process prefill for a single request
    public Task<PrefillResult> ProcessPrefillAsync(
        Request request,
        CancellationToken cancellationToken = default);

    // Process prefill for multiple requests (batch prefill)
    public Task<BatchPrefillResult> ProcessBatchPrefillAsync(
        IEnumerable<Request> requests,
        CancellationToken cancellationToken = default);

    // Check if prefill is needed for a request
    public bool NeedsPrefill(Request request);

    // Estimate prefill memory requirement
    public long EstimatePrefillMemory(Request request);
}
```

---

## Class: PrefillConfiguration
```csharp
public record class PrefillConfiguration(
    int MaxPrefillBatchSize,          // Max requests in prefill batch
    long MaxPrefillMemoryBytes,        // Max memory for prefill
    int PrefillChunkSize,              // Chunk size for long prompts
    bool EnablePrefillCaching,         // Cache common prompts
    int PrefillCacheMaxEntries,        // Max entries in prefill cache
    int PrefillCacheTtlSeconds         // TTL for cache entries
)
{
    public static readonly PrefillConfiguration Default = new(
        MaxPrefillBatchSize: 8,
        MaxPrefillMemoryBytes: 4L * 1024 * 1024 * 1024, // 4GB
        PrefillChunkSize: 512,
        EnablePrefillCaching: true,
        PrefillCacheMaxEntries: 100,
        PrefillCacheTtlSeconds: 300
    );
}
```

**Purpose**: Configure prefill behavior.

---

## Class: PrefillResult
```csharp
public record class PrefillResult(
    RequestId RequestId,
    bool Success,
    int[] PromptTokens,
    float[] InitialLogits,
    int ProcessedTokens,
    long MemoryBytesUsed,
    TimeSpan ProcessingTime,
    string? ErrorMessage
);
```

**Purpose**: Result of single request prefill.

---

## Class: BatchPrefillResult
```csharp
public record class BatchPrefillResult(
    int TotalRequests,
    int SuccessfulRequests,
    int FailedRequests,
    List<PrefillResult> RequestResults,
    TimeSpan TotalProcessingTime,
    long TotalMemoryUsed
)
```

**Purpose**: Result of batch prefill operation.

---

## Class: PrefillCacheEntry
```csharp
public class PrefillCacheEntry
{
    public string Prompt { get; }
    public int[] Tokens { get; }
    public float[] Logits { get; }
    public DateTime CreatedTime { get; }
    public DateTime LastAccessedTime { get; }
    public int AccessCount { get; private set; }

    public PrefillCacheEntry(string prompt, int[] tokens, float[] logits)
    {
        Prompt = prompt;
        Tokens = tokens;
        Logits = logits;
        CreatedTime = DateTime.UtcNow;
        LastAccessedTime = DateTime.UtcNow;
        AccessCount = 0;
    }

    public void MarkAccessed()
    {
        LastAccessedTime = DateTime.UtcNow;
        AccessCount++;
    }

    public bool IsExpired(TimeSpan ttl) =>
        DateTime.UtcNow - LastAccessedTime > ttl;
}
```

**Purpose**: Cache entry for prefill results.

---

## Implementation Details

### ProcessPrefillAsync
```csharp
public async Task<PrefillResult> ProcessPrefillAsync(
    Request request,
    CancellationToken cancellationToken = default)
{
    var stopwatch = Stopwatch.StartNew();

    try
    {
        // Check cache first
        if (_config.EnablePrefillCaching)
        {
            var cachedResult = TryGetFromCache(request.Prompt);
            if (cachedResult != null)
            {
                return new PrefillResult(
                    request.Id,
                    Success: true,
                    cachedResult.Tokens,
                    cachedResult.Logits,
                    cachedResult.Tokens.Length,
                    EstimatePrefillMemory(request),
                    stopwatch.Elapsed,
                    ErrorMessage: null
                );
            }
        }

        // Tokenize prompt
        int[] promptTokens = _tokenizer.Encode(request.Prompt);

        if (promptTokens.Length == 0)
        {
            return new PrefillResult(
                request.Id,
                Success: false,
                Array.Empty<int>(),
                Array.Empty<float>(),
                0,
                0,
                stopwatch.Elapsed,
                "Empty prompt"
            );
        }

        // Handle long prompts with chunking
        if (promptTokens.Length > _config.PrefillChunkSize)
        {
            return await ProcessChunkedPrefillAsync(
                request, promptTokens, stopwatch, cancellationToken);
        }

        // Process single chunk
        float[] logits = await ProcessPrefillChunkAsync(
            request, promptTokens, cancellationToken);

        long memoryUsed = EstimatePrefillMemory(request);

        // Cache result if enabled
        if (_config.EnablePrefillCaching)
        {
            AddToCache(request.Prompt, promptTokens, logits);
        }

        return new PrefillResult(
            request.Id,
            Success: true,
            promptTokens,
            logits,
            promptTokens.Length,
            memoryUsed,
            stopwatch.Elapsed,
            ErrorMessage: null
        );
    }
    catch (Exception ex)
    {
        _logger.LogError(ex, "Prefill failed for request {RequestId}", request.Id);
        return new PrefillResult(
            request.Id,
            Success: false,
            Array.Empty<int>(),
            Array.Empty<float>(),
            0,
            0,
            stopwatch.Elapsed,
            ex.Message
        );
    }
}
```

**Requirements**:
- Check cache first
- Tokenize prompt
- Handle long prompts with chunking
- Process prefill forward pass
- Cache results if enabled
- Handle errors gracefully

---

### ProcessChunkedPrefillAsync
```csharp
private async Task<PrefillResult> ProcessChunkedPrefillAsync(
    Request request,
    int[] promptTokens,
    Stopwatch stopwatch,
    CancellationToken cancellationToken)
{
    var allLogits = new List<float>();
    int processedTokens = 0;

    for (int i = 0; i < promptTokens.Length; i += _config.PrefillChunkSize)
    {
        int chunkSize = Math.Min(_config.PrefillChunkSize, promptTokens.Length - i);
        int[] chunk = promptTokens.Skip(i).Take(chunkSize).ToArray();

        float[] chunkLogits = await ProcessPrefillChunkAsync(
            request, chunk, cancellationToken);

        allLogits.AddRange(chunkLogits);
        processedTokens += chunk.Length;

        cancellationToken.ThrowIfCancellationRequested();
    }

    return new PrefillResult(
        request.Id,
        Success: true,
        promptTokens,
        allLogits.ToArray(),
        processedTokens,
        EstimatePrefillMemory(request),
        stopwatch.Elapsed,
        ErrorMessage: null
    );
}
```

**Requirements**:
- Process prompt in chunks
- Aggregate results
- Handle cancellation

---

### ProcessPrefillChunkAsync
```csharp
private async Task<float[]> ProcessPrefillChunkAsync(
    Request request,
    int[] tokens,
    CancellationToken cancellationToken)
{
    // Create temporary batch for prefill
    var prefillBatch = new Batch(-1) // Special prefill batch ID
    {
        Requests = new List<Request> { request }
    };

    // Set request's current tokens for prefill
    // Note: In production, this would be managed more carefully
    request.GeneratedTokenIds.AddRange(tokens);

    // Execute forward pass
    var output = await _modelExecutor.ExecuteBatchAsync(prefillBatch, cancellationToken);

    // Return logits for last token
    if (output.Logits.TryGetValue(request.Id, out var logits))
    {
        return logits;
    }

    return Array.Empty<float>();
}
```

**Requirements**:
- Create temporary batch
- Execute model forward pass
- Return logits

---

### ProcessBatchPrefillAsync
```csharp
public async Task<BatchPrefillResult> ProcessBatchPrefillAsync(
    IEnumerable<Request> requests,
    CancellationToken cancellationToken = default)
{
    var requestList = requests.ToList();
    var results = new List<PrefillResult>();
    var stopwatch = Stopwatch.StartNew();
    long totalMemoryUsed = 0;

    // Process in batches to respect memory limits
    for (int i = 0; i < requestList.Count; i += _config.MaxPrefillBatchSize)
    {
        var batch = requestList
            .Skip(i)
            .Take(_config.MaxPrefillBatchSize)
            .ToList();

        var batchResults = await Task.WhenAll(
            batch.Select(r => ProcessPrefillAsync(r, cancellationToken))
        );

        results.AddRange(batchResults);

        // Estimate memory used
        foreach (var result in batchResults)
        {
            if (result.Success)
            {
                totalMemoryUsed += result.MemoryBytesUsed;
            }
        }

        cancellationToken.ThrowIfCancellationRequested();
    }

    int successful = results.Count(r => r.Success);
    int failed = results.Count(r => !r.Success);

    return new BatchPrefillResult(
        requestList.Count,
        successful,
        failed,
        results,
        stopwatch.Elapsed,
        totalMemoryUsed
    );
}
```

**Requirements**:
- Process multiple requests
- Respect batch size limits
- Respect memory limits
- Aggregate results

---

### NeedsPrefill
```csharp
public bool NeedsPrefill(Request request)
{
    // Request needs prefill if:
    // 1. Has no generated tokens yet
    // 2. Has a prompt that hasn't been processed
    return request.GeneratedTokens == 0 && !string.IsNullOrEmpty(request.Prompt);
}
```

**Requirements**:
- Check if prefill needed
- Fast check for common case

---

### EstimatePrefillMemory
```csharp
public long EstimatePrefillMemory(Request request)
{
    const int bytesPerToken = 2; // FP16
    const int kvMultiplier = 2; // Key + Value

    int estimatedTokens = EstimateTokenCount(request.Prompt);

    // Prompt processing memory + KV cache
    long promptMemory = estimatedTokens * bytesPerToken;
    long kvCacheMemory = estimatedTokens * bytesPerToken * kvMultiplier;

    return promptMemory + kvCacheMemory;
}

private int EstimateTokenCount(string prompt)
{
    // Conservative estimate: ~4 chars per token
    return (prompt.Length / 4) + 10; // +10 buffer
}
```

**Requirements**:
- Conservative memory estimate
- Account for prompt and KV cache

---

## Prefill Cache Methods

```csharp
private readonly ConcurrentDictionary<string, PrefillCacheEntry> _prefillCache;

private PrefillCacheEntry? TryGetFromCache(string prompt)
{
    if (_prefillCache.TryGetValue(prompt, out var entry))
    {
        if (entry.IsExpired(TimeSpan.FromSeconds(_config.PrefillCacheTtlSeconds)))
        {
            _prefillCache.TryRemove(prompt, out _);
            return null;
        }

        entry.MarkAccessed();
        return entry;
    }
    return null;
}

private void AddToCache(string prompt, int[] tokens, float[] logits)
{
    if (_prefillCache.Count >= _config.PrefillCacheMaxEntries)
    {
        EvictOldestEntry();
    }

    var entry = new PrefillCacheEntry(prompt, tokens, logits);
    _prefillCache.TryAdd(prompt, entry);
}

private void EvictOldestEntry()
{
    var oldestEntry = _prefillCache
        .OrderBy(e => e.Value.LastAccessedTime)
        .FirstOrDefault();

    if (oldestEntry.Key != null)
    {
        _prefillCache.TryRemove(oldestEntry.Key, out _);
    }
}
```

---

## Files to Create

### Implementation
- `src/MLFramework/Inference/ContinuousBatching/Prefill/PrefillHandler.cs`
- `src/MLFramework/Inference/ContinuousBatching/Prefill/PrefillConfiguration.cs`
- `src/MLFramework/Inference/ContinuousBatching/Prefill/PrefillResult.cs`
- `src/MLFramework/Inference/ContinuousBatching/Prefill/PrefillCacheEntry.cs`

### Tests
- `tests/MLFramework.Tests/Inference/ContinuousBatching/Prefill/PrefillHandlerTests.cs`

---

## Dependencies
- `spec_continuous_batching_models.md` (Request, RequestId)
- `spec_kvcache_integration.md` (ContinuousBatchKVCacheManager)

---

## Testing Requirements

### Unit Tests (with Mocks)
1. **Basic Prefill**:
   - Process single request successfully
   - Tokenize prompt correctly
   - Return correct results

2. **Chunked Prefill**:
   - Process long prompts in chunks
   - Aggregate chunk results correctly
   - Handle cancellation during chunking

3. **Batch Prefill**:
   - Process multiple requests in batch
   - Respect batch size limits
   - Respect memory limits

4. **Prefill Caching**:
   - Cache prefill results
   - Retrieve from cache on subsequent calls
   - Expire old cache entries
   - Evict when cache full

5. **Memory Estimation**:
   - EstimatePrefillMemory returns reasonable values
   - Different prompt lengths produce different estimates

6. **NeedsPrefill Check**:
   - Return true for new requests
   - Return false for requests with generated tokens
   - Return false for empty prompts

7. **Error Handling**:
   - Handle tokenization errors
   - Handle model execution errors
   - Handle cancellation during prefill

8. **Edge Cases**:
   - Empty prompt
   - Very long prompt
   - Null prompt
   - Invalid tokens

---

## Success Criteria
- [ ] All public methods implemented
- [ ] Prefill processing works correctly
- [ ] Chunking handles long prompts
- [ ] Batch prefill respects limits
- [ ] Caching works and improves performance
- [ ] Memory estimation reasonable
- [ ] Error handling robust
- [ ] Unit tests cover all scenarios
