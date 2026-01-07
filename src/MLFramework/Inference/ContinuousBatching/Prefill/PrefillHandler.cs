using System.Collections.Concurrent;
using System.Diagnostics;
using Microsoft.Extensions.Logging;

namespace MLFramework.Inference.ContinuousBatching.Prefill;

/// <summary>
/// Handler responsible for processing prefill operations for continuous batching.
/// </summary>
public class PrefillHandler
{
    private readonly IModelExecutor _modelExecutor;
    private readonly ITokenizer _tokenizer;
    private readonly IKVCacheManager _kvCacheManager;
    private readonly PrefillConfiguration _config;
    private readonly ILogger _logger;
    private readonly ConcurrentDictionary<string, PrefillCacheEntry> _prefillCache;

    /// <summary>
    /// Creates a new prefill handler.
    /// </summary>
    public PrefillHandler(
        IModelExecutor modelExecutor,
        ITokenizer tokenizer,
        IKVCacheManager kvCacheManager,
        PrefillConfiguration config,
        ILogger logger)
    {
        _modelExecutor = modelExecutor;
        _tokenizer = tokenizer;
        _kvCacheManager = kvCacheManager;
        _config = config;
        _logger = logger;
        _prefillCache = new ConcurrentDictionary<string, PrefillCacheEntry>();
    }

    /// <summary>
    /// Processes prefill for a single request.
    /// </summary>
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

    /// <summary>
    /// Processes prefill for multiple requests (batch prefill).
    /// </summary>
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

    /// <summary>
    /// Checks if prefill is needed for a request.
    /// </summary>
    public bool NeedsPrefill(Request request)
    {
        // Request needs prefill if:
        // 1. Has no generated tokens yet
        // 2. Has a prompt that hasn't been processed
        return request.GeneratedTokens == 0 && !string.IsNullOrEmpty(request.Prompt);
    }

    /// <summary>
    /// Estimates the memory requirement for prefill.
    /// </summary>
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

    /// <summary>
    /// Processes a chunked prefill for long prompts.
    /// </summary>
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

    /// <summary>
    /// Processes a single prefill chunk.
    /// </summary>
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

    /// <summary>
    /// Estimates the token count for a prompt.
    /// </summary>
    private int EstimateTokenCount(string prompt)
    {
        // Conservative estimate: ~4 chars per token
        return (prompt.Length / 4) + 10; // +10 buffer
    }

    /// <summary>
    /// Tries to get a cached prefill result.
    /// </summary>
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

    /// <summary>
    /// Adds a result to the cache.
    /// </summary>
    private void AddToCache(string prompt, int[] tokens, float[] logits)
    {
        if (_prefillCache.Count >= _config.PrefillCacheMaxEntries)
        {
            EvictOldestEntry();
        }

        var entry = new PrefillCacheEntry(prompt, tokens, logits);
        _prefillCache.TryAdd(prompt, entry);
    }

    /// <summary>
    /// Evicts the oldest cache entry.
    /// </summary>
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
}
