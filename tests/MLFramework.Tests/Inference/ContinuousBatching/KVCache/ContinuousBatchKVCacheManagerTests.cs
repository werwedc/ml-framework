using Moq;
using Xunit;

namespace MLFramework.Tests.Inference.ContinuousBatching;

/// <summary>
/// Unit tests for ContinuousBatchKVCacheManager.
/// </summary>
public class ContinuousBatchKVCacheManagerTests
{
    private readonly Mock<IPagedAttentionCache> _mockPagedCache;
    private readonly KVCacheConfiguration _config;
    private readonly ContinuousBatchKVCacheManager _manager;

    public ContinuousBatchKVCacheManagerTests()
    {
        _mockPagedCache = new Mock<IPagedAttentionCache>();
        _config = KVCacheConfiguration.Default;
        _manager = new ContinuousBatchKVCacheManager(_mockPagedCache.Object, _config);
    }

    [Fact]
    public void AllocateCache_WithValidRequest_AllocatesCorrectPages()
    {
        // Arrange
        var requestId = RequestId.New();
        int maxTokens = 128; // Should allocate 8 pages (16 tokens per page)
        int expectedBlocks = 16; // Initial pages per request

        for (int i = 0; i < expectedBlocks; i++)
        {
            _mockPagedCache.Setup(x => x.AllocateBlock()).Returns(i * 10);
        }

        // Act
        long allocatedBytes = _manager.AllocateCache(requestId, maxTokens);

        // Assert
        Assert.Equal(16 * _config.PageSizeBytes, allocatedBytes);
        var allocation = _manager.GetAllocation(requestId);
        Assert.NotNull(allocation);
        Assert.Equal(expectedBlocks, allocation.Value.PageCount);
        Assert.Equal(expectedBlocks * _config.PageSizeTokens, allocation.Value.TotalTokensAllocated);
        _mockPagedCache.Verify(x => x.AllocateBlock(), Times.Exactly(expectedBlocks));
    }

    [Fact]
    public void AllocateCache_WithZeroMaxTokens_AllocatesInitialPages()
    {
        // Arrange
        var requestId = RequestId.New();
        int maxTokens = 0;
        int expectedBlocks = _config.InitialPagesPerRequest;

        for (int i = 0; i < expectedBlocks; i++)
        {
            _mockPagedCache.Setup(x => x.AllocateBlock()).Returns(i * 10);
        }

        // Act
        long allocatedBytes = _manager.AllocateCache(requestId, maxTokens);

        // Assert
        Assert.Equal(expectedBlocks * _config.PageSizeBytes, allocatedBytes);
        _mockPagedCache.Verify(x => x.AllocateBlock(), Times.Exactly(expectedBlocks));
    }

    [Fact]
    public void AllocateCache_WithDuplicateRequestId_ThrowsException()
    {
        // Arrange
        var requestId = RequestId.New();
        _mockPagedCache.Setup(x => x.AllocateBlock()).Returns(100);
        _manager.AllocateCache(requestId, 100);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            _manager.AllocateCache(requestId, 100));
    }

    [Fact]
    public void AllocateCache_WithNegativeMaxTokens_ThrowsException()
    {
        // Arrange
        var requestId = RequestId.New();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            _manager.AllocateCache(requestId, -1));
    }

    [Fact]
    public void AllocateCache_WithAllocationFailure_RollsBackAllocations()
    {
        // Arrange
        var requestId = RequestId.New();
        var callCount = 0;
        _mockPagedCache.Setup(x => x.AllocateBlock())
            .Returns(() =>
            {
                callCount++;
                if (callCount == 3) throw new Exception("Out of memory");
                return callCount * 10;
            });

        // Act & Assert
        Assert.Throws<Exception>(() => _manager.AllocateCache(requestId, 100));

        // Verify rollback - should release the first 2 allocations
        _mockPagedCache.Verify(x => x.ReleaseBlock(10, 1), Times.Once);
        _mockPagedCache.Verify(x => x.ReleaseBlock(20, 1), Times.Once);
    }

    [Fact]
    public void ReleaseCache_WithValidAllocation_ReleasesAllBlocks()
    {
        // Arrange
        var requestId = RequestId.New();
        var blockIndices = new[] { 100, 110, 120, 130 };

        foreach (var blockIndex in blockIndices)
        {
            _mockPagedCache.Setup(x => x.AllocateBlock()).Returns(blockIndex);
        }

        _manager.AllocateCache(requestId, 64);

        // Act
        _manager.ReleaseCache(requestId);

        // Assert
        foreach (var blockIndex in blockIndices)
        {
            _mockPagedCache.Verify(x => x.ReleaseBlock(blockIndex, 1), Times.Once);
        }

        Assert.Null(_manager.GetAllocation(requestId));
    }

    [Fact]
    public void ReleaseCache_WithNonExistentAllocation_DoesNotThrow()
    {
        // Arrange
        var requestId = RequestId.New();

        // Act & Assert - Should not throw
        _manager.ReleaseCache(requestId);
    }

    [Fact]
    public void GetCurrentUsageBytes_WithMultipleAllocations_ReturnsCorrectTotal()
    {
        // Arrange
        var request1 = RequestId.New();
        var request2 = RequestId.New();
        var request3 = RequestId.New();

        _mockPagedCache.Setup(x => x.AllocateBlock()).Returns(100);
        _manager.AllocateCache(request1, 100);

        _mockPagedCache.Setup(x => x.AllocateBlock()).Returns(200);
        _manager.AllocateCache(request2, 100);

        _mockPagedCache.Setup(x => x.AllocateBlock()).Returns(300);
        _manager.AllocateCache(request3, 100);

        // Act
        long totalUsage = _manager.GetCurrentUsageBytes();

        // Assert
        Assert.Equal(3 * 16 * _config.PageSizeBytes, totalUsage);
    }

    [Fact]
    public void GetCurrentUsageBytes_AfterRelease_UpdatesCorrectly()
    {
        // Arrange
        var request1 = RequestId.New();
        var request2 = RequestId.New();

        _mockPagedCache.Setup(x => x.AllocateBlock()).Returns(100);
        _manager.AllocateCache(request1, 100);

        _mockPagedCache.Setup(x => x.AllocateBlock()).Returns(200);
        _manager.AllocateCache(request2, 100);

        _manager.ReleaseCache(request1);

        // Act
        long totalUsage = _manager.GetCurrentUsageBytes();

        // Assert
        Assert.Equal(16 * _config.PageSizeBytes, totalUsage);
    }

    [Fact]
    public void TryExtendCache_WithCapacity_ExtendsSuccessfully()
    {
        // Arrange
        var requestId = RequestId.New();
        for (int i = 0; i < 16; i++)
        {
            _mockPagedCache.SetupSequence(x => x.AllocateBlock()).Returns(i * 10);
        }
        _manager.AllocateCache(requestId, 64);

        _mockPagedCache.Setup(x => x.AllocateBlock()).Returns(200);

        // Act
        bool result = _manager.TryExtendCache(requestId, 32);

        // Assert
        Assert.True(result);
        var allocation = _manager.GetAllocation(requestId);
        Assert.NotNull(allocation);
        Assert.Equal(17, allocation.Value.PageCount);
        Assert.Equal(17 * _config.PageSizeTokens, allocation.Value.TotalTokensAllocated);
    }

    [Fact]
    public void TryExtendCache_AtMaxCapacity_ReturnsFalse()
    {
        // Arrange
        var requestId = RequestId.New();
        var maxConfig = new KVCacheConfiguration(
            PageSizeTokens: 16,
            InitialPagesPerRequest: 2,
            MaxPagesPerRequest: 2,
            CacheBlockSizeBytes: 1024,
            TargetUtilization: 0.85,
            EnableCompaction: true,
            MaxBatchSize: 64
        );

        var manager = new ContinuousBatchKVCacheManager(_mockPagedCache.Object, maxConfig);

        _mockPagedCache.Setup(x => x.AllocateBlock()).Returns(100);
        _mockPagedCache.Setup(x => x.AllocateBlock()).Returns(110);
        manager.AllocateCache(requestId, 32);

        // Act
        bool result = manager.TryExtendCache(requestId, 32);

        // Assert
        Assert.False(result);
        _mockPagedCache.Verify(x => x.AllocateBlock(), Times.Exactly(2));
    }

    [Fact]
    public void TryExtendCache_WithAllocationFailure_RollsBackAndReturnsFalse()
    {
        // Arrange
        var requestId = RequestId.New();
        for (int i = 0; i < 16; i++)
        {
            _mockPagedCache.SetupSequence(x => x.AllocateBlock()).Returns(i * 10);
        }
        _manager.AllocateCache(requestId, 64);

        _mockPagedCache.Setup(x => x.AllocateBlock()).Throws<Exception>("Out of memory");

        // Act
        bool result = _manager.TryExtendCache(requestId, 32);

        // Assert
        Assert.False(result);
        var allocation = _manager.GetAllocation(requestId);
        Assert.Equal(16, allocation.Value.PageCount);
    }

    [Fact]
    public void TryExtendCache_WithNonExistentRequest_ReturnsFalse()
    {
        // Arrange
        var requestId = RequestId.New();

        // Act
        bool result = _manager.TryExtendCache(requestId, 32);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void TryExtendCache_WithZeroAdditionalTokens_ReturnsFalse()
    {
        // Arrange
        var requestId = RequestId.New();
        _mockPagedCache.Setup(x => x.AllocateBlock()).Returns(100);
        _manager.AllocateCache(requestId, 64);

        // Act
        bool result = _manager.TryExtendCache(requestId, 0);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void GetAllocation_WithValidRequest_ReturnsAllocationDetails()
    {
        // Arrange
        var requestId = RequestId.New();
        _mockPagedCache.Setup(x => x.AllocateBlock()).Returns(100);
        _manager.AllocateCache(requestId, 100);

        // Act
        var allocation = _manager.GetAllocation(requestId);

        // Assert
        Assert.NotNull(allocation);
        Assert.Equal(requestId, allocation.Value.RequestId);
        Assert.Equal(16, allocation.Value.PageCount);
        Assert.True(allocation.Value.TotalBytesAllocated > 0);
    }

    [Fact]
    public void GetAllocation_WithNonExistentRequest_ReturnsNull()
    {
        // Arrange
        var requestId = RequestId.New();

        // Act
        var allocation = _manager.GetAllocation(requestId);

        // Assert
        Assert.Null(allocation);
    }

    [Fact]
    public void CompactCache_WhenCompactionDisabled_DoesNotCompact()
    {
        // Arrange
        var disabledConfig = _config with { EnableCompaction = false };
        var manager = new ContinuousBatchKVCacheManager(_mockPagedCache.Object, disabledConfig);

        _mockPagedCache.Setup(x => x.GetTotalBlockCount()).Returns(1000);
        _mockPagedCache.Setup(x => x.GetFreeBlockCount()).Returns(800); // Low utilization

        // Act
        manager.CompactCache();

        // Assert - should not call cache operations
        _mockPagedCache.Verify(x => x.GetTotalBlockCount(), Times.Never);
        _mockPagedCache.Verify(x => x.GetFreeBlockCount(), Times.Never);
    }

    [Fact]
    public void CompactCache_WhenAboveTargetUtilization_DoesNotCompact()
    {
        // Arrange
        _mockPagedCache.Setup(x => x.GetTotalBlockCount()).Returns(1000);
        _mockPagedCache.Setup(x => x.GetFreeBlockCount()).Returns(100); // High utilization (0.9)

        // Act
        _manager.CompactCache();

        // Assert - Should check but not compact
        _mockPagedCache.Verify(x => x.GetTotalBlockCount(), Times.Once);
        _mockPagedCache.Verify(x => x.GetFreeBlockCount(), Times.Once);
    }

    [Fact]
    public void CompactCache_WhenBelowTargetUtilization_ChecksFragmentation()
    {
        // Arrange
        _mockPagedCache.Setup(x => x.GetTotalBlockCount()).Returns(1000);
        _mockPagedCache.Setup(x => x.GetFreeBlockCount()).Returns(500); // Low utilization (0.5)

        // Act
        _manager.CompactCache();

        // Assert
        _mockPagedCache.Verify(x => x.GetTotalBlockCount(), Times.Once);
        _mockPagedCache.Verify(x => x.GetFreeBlockCount(), Times.Once);
    }

    [Fact]
    public void AllocationCount_ReturnsCorrectCount()
    {
        // Arrange
        var request1 = RequestId.New();
        var request2 = RequestId.New();

        _mockPagedCache.Setup(x => x.AllocateBlock()).Returns(100);
        _manager.AllocateCache(request1, 100);

        _mockPagedCache.Setup(x => x.AllocateBlock()).Returns(200);
        _manager.AllocateCache(request2, 100);

        // Act
        int count = _manager.AllocationCount;

        // Assert
        Assert.Equal(2, count);
    }

    [Fact]
    public void ConcurrentAllocations_ThreadSafe()
    {
        // Arrange
        var requests = new List<RequestId>();
        for (int i = 0; i < 10; i++)
        {
            requests.Add(RequestId.New());
        }

        int blockIndex = 0;
        _mockPagedCache.Setup(x => x.AllocateBlock())
            .Returns(() => Interlocked.Increment(ref blockIndex) * 10);

        // Act
        Parallel.ForEach(requests, request =>
        {
            _manager.AllocateCache(request, 100);
        });

        // Assert
        Assert.Equal(10, _manager.AllocationCount);
        _mockPagedCache.Verify(x => x.AllocateBlock(), Times.Exactly(10 * 16)); // 16 pages per request
    }

    [Fact]
    public void ConcurrentAllocationsAndReleases_ThreadSafe()
    {
        // Arrange
        var requests = new List<RequestId>();
        for (int i = 0; i < 20; i++)
        {
            requests.Add(RequestId.New());
        }

        int blockIndex = 0;
        _mockPagedCache.Setup(x => x.AllocateBlock())
            .Returns(() => Interlocked.Increment(ref blockIndex) * 10);

        // Act - Allocate and release concurrently
        Parallel.For(0, 20, i =>
        {
            _manager.AllocateCache(requests[i], 100);
            if (i % 2 == 0)
            {
                _manager.ReleaseCache(requests[i]);
            }
        });

        // Assert - Half should be released
        Assert.Equal(10, _manager.AllocationCount);
    }

    [Fact]
    public void CacheAllocation_AgeProperty_CalculatesCorrectly()
    {
        // Arrange
        var requestId = RequestId.New();
        _mockPagedCache.Setup(x => x.AllocateBlock()).Returns(100);
        _manager.AllocateCache(requestId, 100);
        var allocation = _manager.GetAllocation(requestId);

        // Act
        var age1 = allocation.Value.Age;
        Thread.Sleep(100);
        var age2 = allocation.Value.Age;

        // Assert
        Assert.True(age2 > age1);
        Assert.True(age2 > TimeSpan.FromMilliseconds(100));
    }

    [Fact]
    public void CachePage_EndTokenProperty_CalculatesCorrectly()
    {
        // Arrange
        var page = new CachePage(0, 16, 32, 100);

        // Act
        int endToken = page.EndToken;

        // Assert
        Assert.Equal(48, endToken);
    }
}
