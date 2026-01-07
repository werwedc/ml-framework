using System.Collections.Concurrent;
using MLFramework.Inference.ContinuousBatching;
using Xunit;

namespace MLFramework.Tests.Inference.ContinuousBatching;

/// <summary>
/// Tests for BatchManager functionality.
/// </summary>
public class BatchManagerTests
{
    #region Mock Implementations

    private class MockKVCacheManager : IKVCacheManager
    {
        private readonly ConcurrentDictionary<RequestId, long> _allocations = new();
        private long _currentUsage = 0;
        private long _allocationSizeBytes;

        public MockKVCacheManager(long allocationSizeBytes = 1024 * 1024) // 1MB default
        {
            _allocationSizeBytes = allocationSizeBytes;
        }

        public long AllocateCache(RequestId requestId, int maxTokens)
        {
            _allocations[requestId] = _allocationSizeBytes;
            Interlocked.Add(ref _currentUsage, _allocationSizeBytes);
            return _allocationSizeBytes;
        }

        public void ReleaseCache(RequestId requestId)
        {
            if (_allocations.TryRemove(requestId, out var size))
            {
                Interlocked.Add(ref _currentUsage, -size);
            }
        }

        public long GetCurrentUsageBytes()
        {
            return _currentUsage;
        }

        public void Reset()
        {
            _allocations.Clear();
            _currentUsage = 0;
        }
    }

    private class MockRequestQueue : IRequestQueue
    {
        private readonly ConcurrentQueue<Request> _queue = new();
        private int _count = 0;

        public MockRequestQueue(IEnumerable<Request>? initialRequests = null)
        {
            if (initialRequests != null)
            {
                foreach (var request in initialRequests)
                {
                    _queue.Enqueue(request);
                    _count++;
                }
            }
        }

        public List<Request> GetRequests(int maxRequests, long maxMemoryBytes)
        {
            var result = new List<Request>();
            while (result.Count < maxRequests && _queue.TryDequeue(out var request))
            {
                result.Add(request);
                _count--;
            }
            return result;
        }

        public void Enqueue(Request request, Priority priority)
        {
            _queue.Enqueue(request);
            _count++;
        }

        public int Count => _count;
    }

    private class TestBatchManager : BatchManager
    {
        public TestBatchManager(
            IRequestQueue requestQueue,
            IKVCacheManager kvCacheManager,
            BatchConstraints constraints) : base(requestQueue, kvCacheManager, constraints)
        {
        }

        public new Batch PrepareNextIteration() => base.PrepareNextIteration();
    }

    #endregion

    #region Test Factories

    private static Request CreateTestRequest(int maxTokens = 100, int generatedTokens = 0)
    {
        return new Request(
            new RequestId(),
            "test prompt",
            maxTokens,
            CancellationToken.None,
            Priority.Normal
        )
        {
            GeneratedTokens = generatedTokens,
            GeneratedTokenIds = new List<int>()
        };
    }

    private static BatchConstraints CreateTestConstraints(
        int maxBatchSize = 4,
        long maxMemoryBytes = 4L * 1024 * 1024, // 4MB
        int minBatchSize = 1,
        int maxSequenceLength = 1024)
    {
        return new BatchConstraints(maxBatchSize, maxMemoryBytes, minBatchSize, maxSequenceLength);
    }

    #endregion

    #region Basic Operations Tests

    [Fact]
    public void Constructor_WithValidParameters_CreatesManager()
    {
        // Arrange
        var mockQueue = new MockRequestQueue();
        var mockKvCache = new MockKVCacheManager();
        var constraints = CreateTestConstraints();

        // Act
        var manager = new BatchManager(mockQueue, mockKvCache, constraints);

        // Assert
        Assert.NotNull(manager);
        Assert.NotNull(manager.CurrentBatch);
        Assert.Equal(0, manager.ActiveRequestCount);
    }

    [Fact]
    public void Constructor_WithNullRequestQueue_ThrowsArgumentNullException()
    {
        // Arrange
        var mockKvCache = new MockKVCacheManager();
        var constraints = CreateTestConstraints();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new BatchManager(null!, mockKvCache, constraints));
    }

    [Fact]
    public void Constructor_WithNullKVCacheManager_ThrowsArgumentNullException()
    {
        // Arrange
        var mockQueue = new MockRequestQueue();
        var constraints = CreateTestConstraints();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new BatchManager(mockQueue, null!, constraints));
    }

    [Fact]
    public void Constructor_WithNullConstraints_ThrowsArgumentNullException()
    {
        // Arrange
        var mockQueue = new MockRequestQueue();
        var mockKvCache = new MockKVCacheManager();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new BatchManager(mockQueue, mockKvCache, null!));
    }

    [Fact]
    public void PrepareNextIteration_WithEmptyQueue_ReturnsEmptyBatch()
    {
        // Arrange
        var mockQueue = new MockRequestQueue();
        var mockKvCache = new MockKVCacheManager();
        var constraints = CreateTestConstraints();
        var manager = new BatchManager(mockQueue, mockKvCache, constraints);

        // Act
        var batch = manager.PrepareNextIteration();

        // Assert
        Assert.NotNull(batch);
        Assert.Equal(0, batch.Requests.Count);
        Assert.Equal(0, batch.Size);
    }

    [Fact]
    public void PrepareNextIteration_WithQueuedRequests_AddsRequestsToBatch()
    {
        // Arrange
        var requests = new[] { CreateTestRequest(), CreateTestRequest() };
        var mockQueue = new MockRequestQueue(requests);
        var mockKvCache = new MockKVCacheManager();
        var constraints = CreateTestConstraints();
        var manager = new BatchManager(mockQueue, mockKvCache, constraints);

        // Act
        var batch = manager.PrepareNextIteration();

        // Assert
        Assert.Equal(2, batch.Requests.Count);
        Assert.Equal(2, batch.Size);
    }

    [Fact]
    public void HasCapacity_WithEmptyBatch_ReturnsTrue()
    {
        // Arrange
        var mockQueue = new MockRequestQueue();
        var mockKvCache = new MockKVCacheManager();
        var constraints = CreateTestConstraints(maxBatchSize: 4, maxMemoryBytes: 4L * 1024 * 1024);
        var manager = new BatchManager(mockQueue, mockKvCache, constraints);

        // Act
        var hasCapacity = manager.HasCapacity();

        // Assert
        Assert.True(hasCapacity);
    }

    [Fact]
    public void HasCapacity_WithFullBatch_ReturnsFalse()
    {
        // Arrange
        var requests = Enumerable.Range(0, 4).Select(_ => CreateTestRequest()).ToArray();
        var mockQueue = new MockRequestQueue(requests);
        var mockKvCache = new MockKVCacheManager();
        var constraints = CreateTestConstraints(maxBatchSize: 4, maxMemoryBytes: 4L * 1024 * 1024);
        var manager = new BatchManager(mockQueue, mockKvCache, constraints);
        manager.PrepareNextIteration(); // Add all requests to batch

        // Act
        var hasCapacity = manager.HasCapacity();

        // Assert
        Assert.False(hasCapacity);
    }

    [Fact]
    public void GetStats_WithEmptyBatch_ReturnsCorrectStats()
    {
        // Arrange
        var mockQueue = new MockRequestQueue();
        var mockKvCache = new MockKVCacheManager();
        var constraints = CreateTestConstraints(maxBatchSize: 4, maxMemoryBytes: 4L * 1024 * 1024);
        var manager = new BatchManager(mockQueue, mockKvCache, constraints);

        // Act
        var stats = manager.GetStats();

        // Assert
        Assert.NotNull(stats);
        Assert.Equal(0, stats.RequestCount);
        Assert.Equal(0, stats.MemoryBytesUsed);
        Assert.Equal(0.0, stats.UtilizationPercentage);
    }

    [Fact]
    public void GetStats_WithRequests_ReturnsCorrectStats()
    {
        // Arrange
        var requests = new[] { CreateTestRequest(), CreateTestRequest() };
        var mockQueue = new MockRequestQueue(requests);
        var mockKvCache = new MockKVCacheManager(1024 * 1024); // 1MB per request
        var constraints = CreateTestConstraints(maxBatchSize: 4, maxMemoryBytes: 4L * 1024 * 1024);
        var manager = new BatchManager(mockQueue, mockKvCache, constraints);
        manager.PrepareNextIteration();

        // Act
        var stats = manager.GetStats();

        // Assert
        Assert.Equal(2, stats.RequestCount);
        Assert.Equal(2L * 1024 * 1024, stats.MemoryBytesUsed);
        Assert.Equal(50.0, stats.UtilizationPercentage);
    }

    #endregion

    #region Completion Handling Tests

    [Fact]
    public void RemoveCompletedRequests_WithCompletedRequest_RemovesFromBatch()
    {
        // Arrange
        var request = CreateTestRequest(maxTokens: 10, generatedTokens: 10);
        var mockQueue = new MockRequestQueue(new[] { request });
        var mockKvCache = new MockKVCacheManager();
        var constraints = CreateTestConstraints();
        var manager = new BatchManager(mockQueue, mockKvCache, constraints);
        manager.PrepareNextIteration();

        // Act
        manager.RemoveCompletedRequests(new List<RequestId> { request.Id });

        // Assert
        Assert.Equal(0, manager.ActiveRequestCount);
    }

    [Fact]
    public void RemoveCompletedRequests_SetsCompletionTaskResult()
    {
        // Arrange
        var request = CreateTestRequest(maxTokens: 10, generatedTokens: 10);
        var mockQueue = new MockRequestQueue(new[] { request });
        var mockKvCache = new MockKVCacheManager();
        var constraints = CreateTestConstraints();
        var manager = new BatchManager(mockQueue, mockKvCache, constraints);
        manager.PrepareNextIteration();

        // Act
        manager.RemoveCompletedRequests(new List<RequestId> { request.Id });

        // Assert
        Assert.True(request.CompletionSource.Task.IsCompleted);
    }

    [Fact]
    public void RemoveCompletedRequests_ReleasesKVCache()
    {
        // Arrange
        var request = CreateTestRequest();
        var mockQueue = new MockRequestQueue(new[] { request });
        var mockKvCache = new MockKVCacheManager();
        var constraints = CreateTestConstraints();
        var manager = new BatchManager(mockQueue, mockKvCache, constraints);
        manager.PrepareNextIteration();
        Assert.Equal(1024 * 1024, mockKvCache.GetCurrentUsageBytes());

        // Act
        manager.RemoveCompletedRequests(new List<RequestId> { request.Id });

        // Assert
        Assert.Equal(0, mockKvCache.GetCurrentUsageBytes());
    }

    [Fact]
    public void RemoveCompletedRequests_WithNullList_ThrowsArgumentNullException()
    {
        // Arrange
        var mockQueue = new MockRequestQueue();
        var mockKvCache = new MockKVCacheManager();
        var constraints = CreateTestConstraints();
        var manager = new BatchManager(mockQueue, mockKvCache, constraints);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            manager.RemoveCompletedRequests(null!));
    }

    #endregion

    #region Adding New Requests Tests

    [Fact]
    public void AddNewRequests_WithCapacity_AddsRequests()
    {
        // Arrange
        var requests = new[] { CreateTestRequest() };
        var mockQueue = new MockRequestQueue(requests);
        var mockKvCache = new MockKVCacheManager();
        var constraints = CreateTestConstraints(maxBatchSize: 4, maxMemoryBytes: 4L * 1024 * 1024);
        var manager = new BatchManager(mockQueue, mockKvCache, constraints);

        // Act
        manager.AddNewRequests();

        // Assert
        Assert.Equal(1, manager.ActiveRequestCount);
    }

    [Fact]
    public void AddNewRequests_RespectsBatchSizeLimit()
    {
        // Arrange
        var requests = Enumerable.Range(0, 6).Select(_ => CreateTestRequest()).ToArray();
        var mockQueue = new MockRequestQueue(requests);
        var mockKvCache = new MockKVCacheManager();
        var constraints = CreateTestConstraints(maxBatchSize: 4, maxMemoryBytes: 4L * 1024 * 1024);
        var manager = new BatchManager(mockQueue, mockKvCache, constraints);

        // Act
        manager.AddNewRequests();

        // Assert
        Assert.Equal(4, manager.ActiveRequestCount);
    }

    [Fact]
    public void AddNewRequests_RespectsMemoryLimit()
    {
        // Arrange
        var requests = Enumerable.Range(0, 6).Select(_ => CreateTestRequest()).ToArray();
        var mockQueue = new MockRequestQueue(requests);
        var mockKvCache = new MockKVCacheManager(1024 * 1024); // 1MB per request
        var constraints = CreateTestConstraints(maxBatchSize: 10, maxMemoryBytes: 2L * 1024 * 1024); // 2MB total
        var manager = new BatchManager(mockQueue, mockKvCache, constraints);

        // Act
        manager.AddNewRequests();

        // Assert
        Assert.Equal(2, manager.ActiveRequestCount);
    }

    [Fact]
    public void AddNewRequests_WithNoCapacity_DoesNotAddRequests()
    {
        // Arrange
        var requests = Enumerable.Range(0, 4).Select(_ => CreateTestRequest()).ToArray();
        var mockQueue = new MockRequestQueue(requests);
        var mockKvCache = new MockKVCacheManager();
        var constraints = CreateTestConstraints(maxBatchSize: 4, maxMemoryBytes: 4L * 1024 * 1024);
        var manager = new BatchManager(mockQueue, mockKvCache, constraints);
        manager.PrepareNextIteration(); // Fill the batch

        // Act
        manager.AddNewRequests();

        // Assert
        Assert.Equal(4, manager.ActiveRequestCount);
        Assert.Equal(0, mockQueue.Count);
    }

    #endregion

    #region Edge Cases Tests

    [Fact]
    public void PrepareNextIteration_WithEmptyQueueAndCapacity_ReturnsEmptyBatch()
    {
        // Arrange
        var mockQueue = new MockRequestQueue();
        var mockKvCache = new MockKVCacheManager();
        var constraints = CreateTestConstraints(maxBatchSize: 4, maxMemoryBytes: 4L * 1024 * 1024);
        var manager = new BatchManager(mockQueue, mockKvCache, constraints);

        // Act
        var batch = manager.PrepareNextIteration();

        // Assert
        Assert.Equal(0, batch.Size);
    }

    [Fact]
    public void PrepareNextIteration_WithFullBatchAndPendingQueue_DoesNotAddMore()
    {
        // Arrange
        var initialRequests = Enumerable.Range(0, 4).Select(_ => CreateTestRequest()).ToArray();
        var pendingRequests = Enumerable.Range(0, 2).Select(_ => CreateTestRequest()).ToArray();
        var mockQueue = new MockRequestQueue(initialRequests.Concat(pendingRequests));
        var mockKvCache = new MockKVCacheManager();
        var constraints = CreateTestConstraints(maxBatchSize: 4, maxMemoryBytes: 4L * 1024 * 1024);
        var manager = new BatchManager(mockQueue, mockKvCache, constraints);

        // Act
        var batch = manager.PrepareNextIteration();

        // Assert
        Assert.Equal(4, batch.Size);
    }

    [Fact]
    public void PrepareNextIteration_ConcurrentCalls_ThreadSafe()
    {
        // Arrange
        var requests = Enumerable.Range(0, 100).Select(_ => CreateTestRequest()).ToArray();
        var mockQueue = new MockRequestQueue(requests);
        var mockKvCache = new MockKVCacheManager();
        var constraints = CreateTestConstraints(maxBatchSize: 4, maxMemoryBytes: 4L * 1024 * 1024);
        var manager = new BatchManager(mockQueue, mockKvCache, constraints);

        // Act
        var tasks = Enumerable.Range(0, 10).Select(_ =>
            Task.Run(() => manager.PrepareNextIteration())
        ).ToArray();
        Task.WaitAll(tasks);

        // Assert
        // All tasks should complete without throwing exceptions
        Assert.True(tasks.All(t => t.Status == TaskStatus.RanToCompletion));
    }

    #endregion

    #region RefreshBatch Tests

    [Fact]
    public void RefreshBatch_WithActiveRequests_CancelsAll()
    {
        // Arrange
        var requests = Enumerable.Range(0, 4).Select(_ => CreateTestRequest()).ToArray();
        var mockQueue = new MockRequestQueue(requests);
        var mockKvCache = new MockKVCacheManager();
        var constraints = CreateTestConstraints(maxBatchSize: 4, maxMemoryBytes: 4L * 1024 * 1024);
        var manager = new BatchManager(mockQueue, mockKvCache, constraints);
        manager.PrepareNextIteration();

        // Act
        manager.RefreshBatch();

        // Assert
        Assert.Equal(0, manager.ActiveRequestCount);
        Assert.Equal(0, mockKvCache.GetCurrentUsageBytes());
    }

    [Fact]
    public void RefreshBatch_SetsCancellationOnRequests()
    {
        // Arrange
        var request = CreateTestRequest();
        var mockQueue = new MockRequestQueue(new[] { request });
        var mockKvCache = new MockKVCacheManager();
        var constraints = CreateTestConstraints();
        var manager = new BatchManager(mockQueue, mockKvCache, constraints);
        manager.PrepareNextIteration();

        // Act
        manager.RefreshBatch();

        // Assert
        Assert.True(request.CompletionSource.Task.IsCanceled);
    }

    #endregion

    #region Constraints Tests

    [Fact]
    public void HasCapacity_WithMaxBatchSizeZero_ReturnsFalse()
    {
        // Arrange
        var mockQueue = new MockRequestQueue();
        var mockKvCache = new MockKVCacheManager();
        var constraints = CreateTestConstraints(maxBatchSize: 0, maxMemoryBytes: 4L * 1024 * 1024);
        var manager = new BatchManager(mockQueue, mockKvCache, constraints);

        // Act
        var hasCapacity = manager.HasCapacity();

        // Assert
        Assert.False(hasCapacity);
    }

    [Fact]
    public void HasCapacity_WithMemoryLimitZero_ReturnsFalse()
    {
        // Arrange
        var mockQueue = new MockRequestQueue();
        var mockKvCache = new MockKVCacheManager();
        var constraints = CreateTestConstraints(maxBatchSize: 4, maxMemoryBytes: 0);
        var manager = new BatchManager(mockQueue, mockKvCache, constraints);

        // Act
        var hasCapacity = manager.HasCapacity();

        // Assert
        Assert.False(hasCapacity);
    }

    [Fact]
    public void AddNewRequests_WithAllocationFailure_PutsRequestBackInQueue()
    {
        // Arrange
        var request = CreateTestRequest();
        var mockQueue = new MockRequestQueue(new[] { request });
        var mockKvCache = new MockKVCacheManager(4L * 1024 * 1024); // 4MB per request
        var constraints = CreateTestConstraints(maxBatchSize: 4, maxMemoryBytes: 2L * 1024 * 1024); // 2MB total
        var manager = new BatchManager(mockQueue, mockKvCache, constraints);

        // Act
        manager.AddNewRequests();

        // Assert
        Assert.Equal(0, manager.ActiveRequestCount);
        Assert.Equal(1, mockQueue.Count);
    }

    #endregion

    #region EOS Token Detection Tests

    [Fact]
    public void PrepareNextIteration_WithEOSToken_CompletesRequest()
    {
        // Arrange
        var request = CreateTestRequest(maxTokens: 100);
        request.GeneratedTokenIds.Add(0); // EOS token ID
        var mockQueue = new MockRequestQueue(new[] { request });
        var mockKvCache = new MockKVCacheManager();
        var constraints = CreateTestConstraints();
        var manager = new TestBatchManager(mockQueue, mockKvCache, constraints);

        // Manually add request to batch for testing
        manager.CurrentBatch.AddRequest(request);

        // Act
        var batch = manager.PrepareNextIteration();

        // Assert
        // Request should be removed (completed)
        Assert.DoesNotContain(batch.Requests, r => r.Id == request.Id);
        Assert.True(request.CompletionSource.Task.IsCompleted);
    }

    #endregion
}
