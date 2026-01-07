using MLFramework.Inference.ContinuousBatching;
using Xunit;

namespace MLFramework.Tests.Inference.ContinuousBatching;

/// <summary>
/// Unit tests for CapacityManager.
/// </summary>
public class CapacityManagerTests
{
    private class MockGPUResourceManager : IGPUResourceManager
    {
        public long TotalMemoryBytes { get; set; } = 16L * 1024 * 1024 * 1024;
        public long CurrentMemoryUsageBytes { get; set; } = 0;
        public double Utilization { get; set; } = 0.0;
        public bool Available { get; set; } = true;

        public long GetTotalMemoryBytes() => TotalMemoryBytes;

        public long GetCurrentMemoryUsageBytes() => CurrentMemoryUsageBytes;

        public double GetUtilization() => Utilization;

        public bool IsAvailable() => Available;
    }

    private CapacityManager CreateManager(CapacityConstraints? constraints = null, IGPUResourceManager? gpuManager = null)
    {
        return new CapacityManager(
            constraints ?? CapacityConstraints.Default,
            gpuManager ?? new MockGPUResourceManager()
        );
    }

    private Request CreateRequest(string prompt = "Hello world", int maxTokens = 100)
    {
        return new Request(
            RequestId.New(),
            prompt,
            maxTokens,
            CancellationToken.None
        );
    }

    [Fact]
    public void TryAllocate_SuccessfulAllocation_ReturnsAllocation()
    {
        // Arrange
        var manager = CreateManager();
        var request = CreateRequest("Test prompt", 50);

        // Act
        bool result = manager.TryAllocate(request, out var allocation);

        // Assert
        Assert.True(result);
        Assert.NotNull(allocation);
        Assert.Equal(request.Id, allocation.RequestId);
        Assert.True(allocation.AllocatedMemoryBytes > 0);
        Assert.Equal(1, allocation.AllocatedSlots);
        Assert.True(allocation.Age.TotalSeconds >= 0);
    }

    [Fact]
    public void TryAllocate_ExceedsMemoryCapacity_ReturnsFalse()
    {
        // Arrange
        var constraints = new CapacityConstraints(
            MaxBatchSize: 32,
            MaxMemoryBytes: 1024,
            MemoryPerSlotBytes: 512,
            MaxConcurrentRequests: 10,
            MemoryBufferRatio: 0.1
        );
        var manager = CreateManager(constraints);
        var request = CreateRequest("Very long prompt that requires lots of memory", 1000);

        // Act
        bool result = manager.TryAllocate(request, out var allocation);

        // Assert
        Assert.False(result);
        Assert.Null(allocation);
    }

    [Fact]
    public void TryAllocate_ExceedsConcurrentRequestsCapacity_ReturnsFalse()
    {
        // Arrange
        var constraints = new CapacityConstraints(
            MaxBatchSize: 32,
            MaxMemoryBytes: 16L * 1024 * 1024 * 1024,
            MemoryPerSlotBytes: 512L * 1024 * 1024,
            MaxConcurrentRequests: 1,
            MemoryBufferRatio: 0.1
        );
        var manager = CreateManager(constraints);
        var request1 = CreateRequest("First request", 50);
        var request2 = CreateRequest("Second request", 50);

        // Act
        manager.TryAllocate(request1, out _);
        bool result = manager.TryAllocate(request2, out var allocation);

        // Assert
        Assert.False(result);
        Assert.Null(allocation);
    }

    [Fact]
    public void Release_ValidAllocation_UpdatesCapacityCounters()
    {
        // Arrange
        var manager = CreateManager();
        var request = CreateRequest("Test prompt", 50);
        manager.TryAllocate(request, out var allocation);
        var utilizationBefore = manager.GetUtilization();

        // Act
        manager.Release(request.Id);
        var utilizationAfter = manager.GetUtilization();

        // Assert
        Assert.Equal(1, utilizationBefore.ActiveRequestCount);
        Assert.Equal(0, utilizationAfter.ActiveRequestCount);
        Assert.Equal(0, utilizationAfter.TotalMemoryUsedBytes);
    }

    [Fact]
    public void Release_MissingAllocation_HandlesGracefully()
    {
        // Arrange
        var manager = CreateManager();
        var request = CreateRequest("Test prompt", 50);

        // Act & Assert - Should not throw
        manager.Release(request.Id);
        var utilization = manager.GetUtilization();
        Assert.Equal(0, utilization.ActiveRequestCount);
    }

    [Fact]
    public void Release_AfterAllocation_ReducesTotalAllocatedBytes()
    {
        // Arrange
        var manager = CreateManager();
        var request = CreateRequest("Test prompt", 50);
        manager.TryAllocate(request, out var allocation);

        // Act
        manager.Release(request.Id);
        var availableCapacity = manager.GetAvailableCapacity();

        // Assert
        Assert.Equal(_constraints.EffectiveMemoryBytes, availableCapacity.AvailableMemoryBytes);
    }

    private static readonly CapacityConstraints _constraints = CapacityConstraints.Default;

    [Fact]
    public void CanFitBatch_ValidBatch_ReturnsTrue()
    {
        // Arrange
        var manager = CreateManager();

        // Act
        bool result = manager.CanFitBatch(5, 1024L * 1024 * 1024);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void CanFitBatch_ExceedsMemoryCapacity_ReturnsFalse()
    {
        // Arrange
        var manager = CreateManager();

        // Act
        bool result = manager.CanFitBatch(1, 20L * 1024 * 1024 * 1024);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void CanFitBatch_ExceedsSlotCapacity_ReturnsFalse()
    {
        // Arrange
        var manager = CreateManager();

        // Act
        bool result = manager.CanFitBatch(100, 1024);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void CanFitBatch_WithExistingAllocations_ConsidersExistingUsage()
    {
        // Arrange
        var constraints = new CapacityConstraints(
            MaxBatchSize: 32,
            MaxMemoryBytes: 2L * 1024 * 1024 * 1024,
            MemoryPerSlotBytes: 512L * 1024 * 1024,
            MaxConcurrentRequests: 5,
            MemoryBufferRatio: 0.1
        );
        var manager = CreateManager(constraints);
        var request1 = CreateRequest("First request", 1000);
        manager.TryAllocate(request1, out _);

        // Act
        bool result = manager.CanFitBatch(4, 1L * 1024 * 1024 * 1024);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void GetUtilization_NoAllocations_ReturnsZeroUtilization()
    {
        // Arrange
        var manager = CreateManager();

        // Act
        var utilization = manager.GetUtilization();

        // Assert
        Assert.Equal(0, utilization.ActiveRequestCount);
        Assert.Equal(0.0, utilization.SlotUtilization);
        Assert.Equal(0.0, utilization.MemoryUtilization);
        Assert.Equal(0, utilization.TotalMemoryUsedBytes);
        Assert.Equal(_constraints.EffectiveMemoryBytes, utilization.AvailableMemoryBytes);
    }

    [Fact]
    public void GetUtilization_WithAllocations_ReturnsAccurateStatistics()
    {
        // Arrange
        var manager = CreateManager();
        var request1 = CreateRequest("First prompt", 50);
        var request2 = CreateRequest("Second prompt", 75);

        // Act
        manager.TryAllocate(request1, out _);
        manager.TryAllocate(request2, out _);
        var utilization = manager.GetUtilization();

        // Assert
        Assert.Equal(2, utilization.ActiveRequestCount);
        Assert.True(utilization.SlotUtilization > 0);
        Assert.True(utilization.MemoryUtilization > 0);
        Assert.True(utilization.TotalMemoryUsedBytes > 0);
    }

    [Fact]
    public void GetUtilization_AverageUtilization_CalculatesCorrectly()
    {
        // Arrange
        var manager = CreateManager();
        var request = CreateRequest("Test prompt", 50);
        manager.TryAllocate(request, out _);

        // Act
        var utilization = manager.GetUtilization();

        // Assert
        Assert.Equal(
            (utilization.SlotUtilization + utilization.MemoryUtilization) / 2.0,
            utilization.AverageUtilization
        );
    }

    [Fact]
    public void GetUtilization_DivisionByZero_HandlesSafely()
    {
        // Arrange
        var constraints = new CapacityConstraints(
            MaxBatchSize: 0,
            MaxMemoryBytes: 0,
            MemoryPerSlotBytes: 0,
            MaxConcurrentRequests: 0,
            MemoryBufferRatio: 0.1
        );
        var manager = CreateManager(constraints);

        // Act & Assert - Should not throw
        var utilization = manager.GetUtilization();
        Assert.Equal(0.0, utilization.SlotUtilization);
        Assert.Equal(0.0, utilization.MemoryUtilization);
    }

    [Fact]
    public void GetAvailableCapacity_NoAllocations_ReturnsFullCapacity()
    {
        // Arrange
        var manager = CreateManager();

        // Act
        var capacity = manager.GetAvailableCapacity();

        // Assert
        Assert.Equal(_constraints.MaxConcurrentRequests, capacity.AvailableSlots);
        Assert.Equal(_constraints.EffectiveMemoryBytes, capacity.AvailableMemoryBytes);
    }

    [Fact]
    public void GetAvailableCapacity_WithAllocations_ReturnsRemainingCapacity()
    {
        // Arrange
        var manager = CreateManager();
        var request = CreateRequest("Test prompt", 50);
        manager.TryAllocate(request, out var allocation);

        // Act
        var capacity = manager.GetAvailableCapacity();

        // Assert
        Assert.Equal(_constraints.MaxConcurrentRequests - 1, capacity.AvailableSlots);
        Assert.True(capacity.AvailableMemoryBytes < _constraints.EffectiveMemoryBytes);
    }

    [Fact]
    public void Reset_ClearsAllAllocations()
    {
        // Arrange
        var manager = CreateManager();
        var request1 = CreateRequest("First request", 50);
        var request2 = CreateRequest("Second request", 75);
        manager.TryAllocate(request1, out _);
        manager.TryAllocate(request2, out _);

        // Act
        manager.Reset();
        var utilization = manager.GetUtilization();

        // Assert
        Assert.Equal(0, utilization.ActiveRequestCount);
        Assert.Equal(0, utilization.TotalMemoryUsedBytes);
    }

    [Fact]
    public void Reset_ZeroesAllCounters()
    {
        // Arrange
        var manager = CreateManager();
        var request = CreateRequest("Test request", 50);
        manager.TryAllocate(request, out _);

        // Act
        manager.Reset();
        var capacity = manager.GetAvailableCapacity();

        // Assert
        Assert.Equal(_constraints.MaxConcurrentRequests, capacity.AvailableSlots);
        Assert.Equal(_constraints.EffectiveMemoryBytes, capacity.AvailableMemoryBytes);
    }

    [Fact]
    public void UpdateUsage_ValidRequest_AdjustsMemoryAllocation()
    {
        // Arrange
        var manager = CreateManager();
        var request = CreateRequest("Test prompt", 100);
        manager.TryAllocate(request, out var allocation);
        long initialMemory = allocation.AllocatedMemoryBytes;

        // Act
        manager.UpdateUsage(request.Id, 10);
        var utilization = manager.GetUtilization();

        // Assert
        Assert.True(utilization.TotalMemoryUsedBytes > initialMemory);
    }

    [Fact]
    public void UpdateUsage_MissingRequest_HandlesGracefully()
    {
        // Arrange
        var manager = CreateManager();
        var request = CreateRequest("Test prompt", 100);

        // Act & Assert - Should not throw
        manager.UpdateUsage(request.Id, 10);
        var utilization = manager.GetUtilization();
        Assert.Equal(0, utilization.ActiveRequestCount);
    }

    [Fact]
    public void UpdateUsage_UpdatesTotalCorrectly()
    {
        // Arrange
        var manager = CreateManager();
        var request = CreateRequest("Test prompt", 100);
        manager.TryAllocate(request, out _);
        var utilizationBefore = manager.GetUtilization();

        // Act
        manager.UpdateUsage(request.Id, 5);
        var utilizationAfter = manager.GetUtilization();

        // Assert
        Assert.True(utilizationAfter.TotalMemoryUsedBytes > utilizationBefore.TotalMemoryUsedBytes);
    }

    [Fact]
    public void EstimateRequiredMemory_DifferentPrompts_ProducesDifferentEstimates()
    {
        // Arrange
        var manager = CreateManager();
        var shortPromptRequest = CreateRequest("Hi", 100);
        var longPromptRequest = CreateRequest("This is a very long prompt with many words and characters", 100);

        // Act
        manager.TryAllocate(shortPromptRequest, out var shortAlloc);
        manager.Reset();
        manager.TryAllocate(longPromptRequest, out var longAlloc);

        // Assert
        Assert.NotEqual(shortAlloc.AllocatedMemoryBytes, longAlloc.AllocatedMemoryBytes);
    }

    [Fact]
    public void EstimateRequiredMemory_IncludesSafetyBuffers()
    {
        // Arrange
        var manager = CreateManager();
        var request = CreateRequest("Test", 100);

        // Act
        manager.TryAllocate(request, out var allocation);

        // Assert
        // Estimate includes +10 buffer for prompt tokens and activation overhead
        Assert.True(allocation.AllocatedMemoryBytes > 0);
    }

    [Fact]
    public void ThreadSafety_ConcurrentAllocations()
    {
        // Arrange
        var manager = CreateManager();
        int successCount = 0;
        int failureCount = 0;
        int iterations = 50;
        var barrier = new System.Threading.Barrier(iterations);

        // Act
        Parallel.For(0, iterations, i =>
        {
            var request = CreateRequest($"Request {i}", 50);
            barrier.SignalAndWait(); // Wait for all threads to be ready
            if (manager.TryAllocate(request, out _))
            {
                Interlocked.Increment(ref successCount);
            }
            else
            {
                Interlocked.Increment(ref failureCount);
            }
        });

        var utilization = manager.GetUtilization();

        // Assert
        Assert.True(successCount + failureCount == iterations);
        Assert.True(successCount <= _constraints.MaxConcurrentRequests);
        Assert.Equal(successCount, utilization.ActiveRequestCount);
    }

    [Fact]
    public void ThreadSafety_ConcurrentReleases()
    {
        // Arrange
        var manager = CreateManager();
        var requests = new List<Request>();

        // Allocate all possible slots
        for (int i = 0; i < _constraints.MaxConcurrentRequests; i++)
        {
            var request = CreateRequest($"Request {i}", 50);
            requests.Add(request);
            manager.TryAllocate(request, out _);
        }

        var barrier = new System.Threading.Barrier(requests.Count);

        // Act
        Parallel.ForEach(requests, request =>
        {
            barrier.SignalAndWait(); // Wait for all threads to be ready
            manager.Release(request.Id);
        });

        var utilization = manager.GetUtilization();

        // Assert
        Assert.Equal(0, utilization.ActiveRequestCount);
        Assert.Equal(0, utilization.TotalMemoryUsedBytes);
    }

    [Fact]
    public void ThreadSafety_ConcurrentAllocationAndRelease()
    {
        // Arrange
        var manager = CreateManager();
        var requests = new ConcurrentBag<Request>();
        int iterations = 100;

        // Act
        Parallel.For(0, iterations, i =>
        {
            if (i % 2 == 0)
            {
                // Even: Allocate
                var request = CreateRequest($"Request {i}", 50);
                if (manager.TryAllocate(request, out _))
                {
                    requests.Add(request);
                }
            }
            else
            {
                // Odd: Release first request if available
                if (requests.TryTake(out var request))
                {
                    manager.Release(request.Id);
                }
            }
        });

        var utilization = manager.GetUtilization();

        // Assert - Should complete without exceptions
        Assert.True(utilization.ActiveRequestCount <= _constraints.MaxConcurrentRequests);
    }
}
