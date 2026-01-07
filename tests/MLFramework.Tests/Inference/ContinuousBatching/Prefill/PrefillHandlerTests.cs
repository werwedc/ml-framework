using Microsoft.Extensions.Logging;
using MLFramework.Inference;
using MLFramework.Inference.ContinuousBatching;
using MLFramework.Inference.ContinuousBatching.Prefill;
using Moq;
using Xunit;

namespace MLFramework.Tests.Inference.ContinuousBatching.Prefill;

/// <summary>
/// Unit tests for PrefillHandler.
/// </summary>
public class PrefillHandlerTests
{
    private readonly Mock<IModelExecutor> _mockModelExecutor;
    private readonly Mock<ITokenizer> _mockTokenizer;
    private readonly Mock<IKVCacheManager> _mockKVCacheManager;
    private readonly Mock<ILogger<PrefillHandler>> _mockLogger;
    private readonly PrefillConfiguration _config;

    public PrefillHandlerTests()
    {
        _mockModelExecutor = new Mock<IModelExecutor>();
        _mockTokenizer = new Mock<ITokenizer>();
        _mockKVCacheManager = new Mock<IKVCacheManager>();
        _mockLogger = new Mock<ILogger<PrefillHandler>>();
        _config = PrefillConfiguration.Default;
    }

    private PrefillHandler CreateHandler()
    {
        return new PrefillHandler(
            _mockModelExecutor.Object,
            _mockTokenizer.Object,
            _mockKVCacheManager.Object,
            _config,
            _mockLogger.Object
        );
    }

    [Fact]
    public async Task ProcessPrefillAsync_SuccessfulPrefill_ReturnsCorrectResult()
    {
        // Arrange
        var handler = CreateHandler();
        var request = new Request(
            RequestId.FromString("test-request"),
            "Test prompt",
            maxTokens: 100,
            CancellationToken.None
        );

        var expectedTokens = new[] { 1, 2, 3, 4, 5 };
        var expectedLogits = new float[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f };

        _mockTokenizer.Setup(t => t.Encode(request.Prompt)).Returns(expectedTokens);
        _mockModelExecutor.Setup(e => e.ExecuteBatchAsync(It.IsAny<Batch>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(new ModelOutput(new Dictionary<RequestId, float[]> { { request.Id, expectedLogits } }));

        // Act
        var result = await handler.ProcessPrefillAsync(request);

        // Assert
        Assert.True(result.Success);
        Assert.Equal(request.Id, result.RequestId);
        Assert.Equal(expectedTokens, result.PromptTokens);
        Assert.Equal(expectedLogits, result.InitialLogits);
        Assert.Equal(expectedTokens.Length, result.ProcessedTokens);
        Assert.Null(result.ErrorMessage);
    }

    [Fact]
    public async Task ProcessPrefillAsync_EmptyPrompt_ReturnsFailure()
    {
        // Arrange
        var handler = CreateHandler();
        var request = new Request(
            RequestId.FromString("test-request"),
            "",
            maxTokens: 100,
            CancellationToken.None
        );

        _mockTokenizer.Setup(t => t.Encode(request.Prompt)).Returns(Array.Empty<int>());

        // Act
        var result = await handler.ProcessPrefillAsync(request);

        // Assert
        Assert.False(result.Success);
        Assert.Equal("Empty prompt", result.ErrorMessage);
        Assert.Empty(result.PromptTokens);
        Assert.Empty(result.InitialLogits);
    }

    [Fact]
    public async Task ProcessPrefillAsync_LongPrompt_UsesChunking()
    {
        // Arrange
        var configWithSmallChunk = new PrefillConfiguration(
            MaxPrefillBatchSize: 8,
            MaxPrefillMemoryBytes: 4L * 1024 * 1024 * 1024,
            PrefillChunkSize: 10, // Small chunk size
            EnablePrefillCaching: false,
            PrefillCacheMaxEntries: 100,
            PrefillCacheTtlSeconds: 300
        );

        var handler = new PrefillHandler(
            _mockModelExecutor.Object,
            _mockTokenizer.Object,
            _mockKVCacheManager.Object,
            configWithSmallChunk,
            _mockLogger.Object
        );

        var request = new Request(
            RequestId.FromString("test-request"),
            "Test prompt that is longer than chunk size",
            maxTokens: 100,
            CancellationToken.None
        );

        var expectedTokens = Enumerable.Range(1, 25).ToArray(); // 25 tokens > chunk size of 10
        var chunk1Logits = Enumerable.Range(1, 10).Select(i => (float)i).ToArray();
        var chunk2Logits = Enumerable.Range(11, 10).Select(i => (float)i).ToArray();
        var chunk3Logits = Enumerable.Range(21, 5).Select(i => (float)i).ToArray();

        _mockTokenizer.Setup(t => t.Encode(request.Prompt)).Returns(expectedTokens);

        int callCount = 0;
        _mockModelExecutor.Setup(e => e.ExecuteBatchAsync(It.IsAny<Batch>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(() =>
            {
                callCount++;
                float[] logits = callCount switch
                {
                    1 => chunk1Logits,
                    2 => chunk2Logits,
                    _ => chunk3Logits
                };
                return new ModelOutput(new Dictionary<RequestId, float[]> { { request.Id, logits } });
            });

        // Act
        var result = await handler.ProcessPrefillAsync(request);

        // Assert
        Assert.True(result.Success);
        Assert.Equal(expectedTokens.Length, result.ProcessedTokens);
        Assert.Equal(25, result.InitialLogits.Length); // All chunks combined
        Assert.Equal(3, callCount); // Should be called 3 times for 3 chunks
    }

    [Fact]
    public async Task ProcessBatchPrefillAsync_MultipleRequests_ProcessesAll()
    {
        // Arrange
        var handler = CreateHandler();
        var requests = new[]
        {
            new Request(RequestId.FromString("req1"), "Prompt 1", 100, CancellationToken.None),
            new Request(RequestId.FromString("req2"), "Prompt 2", 100, CancellationToken.None),
            new Request(RequestId.FromString("req3"), "Prompt 3", 100, CancellationToken.None)
        };

        foreach (var request in requests)
        {
            _mockTokenizer.Setup(t => t.Encode(request.Prompt)).Returns(new[] { 1, 2, 3 });
            _mockModelExecutor.Setup(e => e.ExecuteBatchAsync(It.IsAny<Batch>(), It.IsAny<CancellationToken>()))
                .ReturnsAsync(new ModelOutput(new Dictionary<RequestId, float[]> { { request.Id, new[] { 0.1f, 0.2f, 0.3f } } }));
        }

        // Act
        var result = await handler.ProcessBatchPrefillAsync(requests);

        // Assert
        Assert.Equal(3, result.TotalRequests);
        Assert.Equal(3, result.SuccessfulRequests);
        Assert.Equal(0, result.FailedRequests);
        Assert.Equal(3, result.RequestResults.Count);
        Assert.All(result.RequestResults, r => Assert.True(r.Success));
    }

    [Fact]
    public async Task ProcessBatchPrefillAsync_RespectsBatchSizeLimit()
    {
        // Arrange
        var configWithSmallBatch = new PrefillConfiguration(
            MaxPrefillBatchSize: 2, // Small batch size
            MaxPrefillMemoryBytes: 4L * 1024 * 1024 * 1024,
            PrefillChunkSize: 512,
            EnablePrefillCaching: false,
            PrefillCacheMaxEntries: 100,
            PrefillCacheTtlSeconds: 300
        );

        var handler = new PrefillHandler(
            _mockModelExecutor.Object,
            _mockTokenizer.Object,
            _mockKVCacheManager.Object,
            configWithSmallBatch,
            _mockLogger.Object
        );

        var requests = new[]
        {
            new Request(RequestId.FromString("req1"), "Prompt 1", 100, CancellationToken.None),
            new Request(RequestId.FromString("req2"), "Prompt 2", 100, CancellationToken.None),
            new Request(RequestId.FromString("req3"), "Prompt 3", 100, CancellationToken.None),
            new Request(RequestId.FromString("req4"), "Prompt 4", 100, CancellationToken.None),
            new Request(RequestId.FromString("req5"), "Prompt 5", 100, CancellationToken.None)
        };

        foreach (var request in requests)
        {
            _mockTokenizer.Setup(t => t.Encode(request.Prompt)).Returns(new[] { 1, 2, 3 });
            _mockModelExecutor.Setup(e => e.ExecuteBatchAsync(It.IsAny<Batch>(), It.IsAny<CancellationToken>()))
                .ReturnsAsync(new ModelOutput(new Dictionary<RequestId, float[]> { { request.Id, new[] { 0.1f, 0.2f, 0.3f } } }));
        }

        // Act
        var result = await handler.ProcessBatchPrefillAsync(requests);

        // Assert
        Assert.Equal(5, result.TotalRequests);
        Assert.Equal(5, result.SuccessfulRequests);
        // Verify processing was done in batches of 2
        _mockModelExecutor.Verify(
            e => e.ExecuteBatchAsync(It.IsAny<Batch>(), It.IsAny<CancellationToken>()),
            Times.AtLeast(3) // At least 3 batches for 5 requests with batch size 2
        );
    }

    [Fact]
    public async Task ProcessPrefillAsync_CachingEnabled_UsesCache()
    {
        // Arrange
        var configWithCaching = new PrefillConfiguration(
            MaxPrefillBatchSize: 8,
            MaxPrefillMemoryBytes: 4L * 1024 * 1024 * 1024,
            PrefillChunkSize: 512,
            EnablePrefillCaching: true,
            PrefillCacheMaxEntries: 100,
            PrefillCacheTtlSeconds: 300
        );

        var handler = new PrefillHandler(
            _mockModelExecutor.Object,
            _mockTokenizer.Object,
            _mockKVCacheManager.Object,
            configWithCaching,
            _mockLogger.Object
        );

        var request = new Request(
            RequestId.FromString("test-request"),
            "Test prompt",
            maxTokens: 100,
            CancellationToken.None
        );

        var expectedTokens = new[] { 1, 2, 3 };
        var expectedLogits = new[] { 0.1f, 0.2f, 0.3f };

        _mockTokenizer.Setup(t => t.Encode(request.Prompt)).Returns(expectedTokens);
        _mockModelExecutor.Setup(e => e.ExecuteBatchAsync(It.IsAny<Batch>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(new ModelOutput(new Dictionary<RequestId, float[]> { { request.Id, expectedLogits } }));

        // Act - First call
        var result1 = await handler.ProcessPrefillAsync(request);

        // Act - Second call with same prompt
        var result2 = await handler.ProcessPrefillAsync(request);

        // Assert
        Assert.True(result1.Success);
        Assert.True(result2.Success);

        // Tokenizer should be called once (first call)
        _mockTokenizer.Verify(t => t.Encode(request.Prompt), Times.Once);

        // Model executor should be called once (first call)
        _mockModelExecutor.Verify(e => e.ExecuteBatchAsync(It.IsAny<Batch>(), It.IsAny<CancellationToken>()), Times.Once);
    }

    [Fact]
    public async Task ProcessPrefillAsync_CacheEntryExpires_Retokenizes()
    {
        // Arrange
        var configWithShortTtl = new PrefillConfiguration(
            MaxPrefillBatchSize: 8,
            MaxPrefillMemoryBytes: 4L * 1024 * 1024 * 1024,
            PrefillChunkSize: 512,
            EnablePrefillCaching: true,
            PrefillCacheMaxEntries: 100,
            PrefillCacheTtlSeconds: 0 // Immediate expiration
        );

        var handler = new PrefillHandler(
            _mockModelExecutor.Object,
            _mockTokenizer.Object,
            _mockKVCacheManager.Object,
            configWithShortTtl,
            _mockLogger.Object
        );

        var request = new Request(
            RequestId.FromString("test-request"),
            "Test prompt",
            maxTokens: 100,
            CancellationToken.None
        );

        var expectedTokens = new[] { 1, 2, 3 };
        var expectedLogits = new[] { 0.1f, 0.2f, 0.3f };

        _mockTokenizer.Setup(t => t.Encode(request.Prompt)).Returns(expectedTokens);
        _mockModelExecutor.Setup(e => e.ExecuteBatchAsync(It.IsAny<Batch>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(new ModelOutput(new Dictionary<RequestId, float[]> { { request.Id, expectedLogits } }));

        // Act - First call
        var result1 = await handler.ProcessPrefillAsync(request);

        // Wait for TTL to expire (already immediate)
        await Task.Delay(10);

        // Act - Second call - should retokenize
        var result2 = await handler.ProcessPrefillAsync(request);

        // Assert
        Assert.True(result1.Success);
        Assert.True(result2.Success);

        // Tokenizer should be called twice
        _mockTokenizer.Verify(t => t.Encode(request.Prompt), Times.Exactly(2));
    }

    [Fact]
    public void NeedsPrefill_NewRequest_ReturnsTrue()
    {
        // Arrange
        var handler = CreateHandler();
        var request = new Request(
            RequestId.FromString("test-request"),
            "Test prompt",
            maxTokens: 100,
            CancellationToken.None
        );

        // Act
        var result = handler.NeedsPrefill(request);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void NeedsPrefill_RequestWithGeneratedTokens_ReturnsFalse()
    {
        // Arrange
        var handler = CreateHandler();
        var request = new Request(
            RequestId.FromString("test-request"),
            "Test prompt",
            maxTokens: 100,
            CancellationToken.None
        );

        request.GeneratedTokens = 5;

        // Act
        var result = handler.NeedsPrefill(request);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void NeedsPrefill_EmptyPrompt_ReturnsFalse()
    {
        // Arrange
        var handler = CreateHandler();
        var request = new Request(
            RequestId.FromString("test-request"),
            "",
            maxTokens: 100,
            CancellationToken.None
        );

        // Act
        var result = handler.NeedsPrefill(request);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void EstimatePrefillMemory_ReturnsReasonableValue()
    {
        // Arrange
        var handler = CreateHandler();
        var request = new Request(
            RequestId.FromString("test-request"),
            "Test prompt with some text",
            maxTokens: 100,
            CancellationToken.None
        );

        // Act
        var memoryBytes = handler.EstimatePrefillMemory(request);

        // Assert
        Assert.True(memoryBytes > 0);
        // Check it's not ridiculously large
        Assert.True(memoryBytes < 1024 * 1024); // Less than 1MB for a simple prompt
    }

    [Fact]
    public void EstimatePrefillMemory_LongerPrompt_ReturnsHigherEstimate()
    {
        // Arrange
        var handler = CreateHandler();
        var shortRequest = new Request(
            RequestId.FromString("test-request1"),
            "Short",
            maxTokens: 100,
            CancellationToken.None
        );

        var longRequest = new Request(
            RequestId.FromString("test-request2"),
            string.Join(" ", Enumerable.Repeat("word", 100)),
            maxTokens: 100,
            CancellationToken.None
        );

        // Act
        var shortMemory = handler.EstimatePrefillMemory(shortRequest);
        var longMemory = handler.EstimatePrefillMemory(longRequest);

        // Assert
        Assert.True(longMemory > shortMemory);
    }

    [Fact]
    public async Task ProcessPrefillAsync_TokenizerError_ReturnsFailure()
    {
        // Arrange
        var handler = CreateHandler();
        var request = new Request(
            RequestId.FromString("test-request"),
            "Test prompt",
            maxTokens: 100,
            CancellationToken.None
        );

        _mockTokenizer.Setup(t => t.Encode(request.Prompt)).Throws(new Exception("Tokenizer error"));

        // Act
        var result = await handler.ProcessPrefillAsync(request);

        // Assert
        Assert.False(result.Success);
        Assert.Contains("Tokenizer error", result.ErrorMessage);
    }

    [Fact]
    public async Task ProcessPrefillAsync_ModelExecutorError_ReturnsFailure()
    {
        // Arrange
        var handler = CreateHandler();
        var request = new Request(
            RequestId.FromString("test-request"),
            "Test prompt",
            maxTokens: 100,
            CancellationToken.None
        );

        var expectedTokens = new[] { 1, 2, 3 };

        _mockTokenizer.Setup(t => t.Encode(request.Prompt)).Returns(expectedTokens);
        _mockModelExecutor.Setup(e => e.ExecuteBatchAsync(It.IsAny<Batch>(), It.IsAny<CancellationToken>()))
            .Throws(new Exception("Model error"));

        // Act
        var result = await handler.ProcessPrefillAsync(request);

        // Assert
        Assert.False(result.Success);
        Assert.Contains("Model error", result.ErrorMessage);
    }

    [Fact]
    public async Task ProcessPrefillAsync_CancellationDuringChunking_Throws()
    {
        // Arrange
        var configWithSmallChunk = new PrefillConfiguration(
            MaxPrefillBatchSize: 8,
            MaxPrefillMemoryBytes: 4L * 1024 * 1024 * 1024,
            PrefillChunkSize: 5, // Very small chunk size
            EnablePrefillCaching: false,
            PrefillCacheMaxEntries: 100,
            PrefillCacheTtlSeconds: 300
        );

        var handler = new PrefillHandler(
            _mockModelExecutor.Object,
            _mockTokenizer.Object,
            _mockKVCacheManager.Object,
            configWithSmallChunk,
            _mockLogger.Object
        );

        var cts = new CancellationTokenSource();
        var request = new Request(
            RequestId.FromString("test-request"),
            "Test prompt that will require multiple chunks",
            maxTokens: 100,
            cts.Token
        );

        var expectedTokens = Enumerable.Range(1, 20).ToArray();

        _mockTokenizer.Setup(t => t.Encode(request.Prompt)).Returns(expectedTokens);

        int callCount = 0;
        _mockModelExecutor.Setup(e => e.ExecuteBatchAsync(It.IsAny<Batch>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(() =>
            {
                callCount++;
                if (callCount == 2)
                {
                    cts.Cancel(); // Cancel on second chunk
                }
                float[] logits = Enumerable.Repeat(0.1f, 5).ToArray();
                return new ModelOutput(new Dictionary<RequestId, float[]> { { request.Id, logits } });
            });

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            handler.ProcessPrefillAsync(request));
    }

    [Fact]
    public async Task ProcessPrefillAsync_NullRequest_LogsError()
    {
        // Arrange
        var handler = CreateHandler();
        Request? request = null;

        // Act & Assert
        await Assert.ThrowsAsync<NullReferenceException>(() =>
            handler.ProcessPrefillAsync(request!));
    }
}
