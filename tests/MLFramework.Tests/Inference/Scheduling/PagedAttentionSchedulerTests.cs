using MlFramework.Inference.PagedAttention;
using MLFramework.Core;
using Xunit;

namespace MLFramework.Tests.Inference.Scheduling;

/// <summary>
/// Tests for PagedAttentionScheduler
/// </summary>
public class PagedAttentionSchedulerTests
{
    private readonly DeviceId _deviceId = DeviceId.CPU;

    [Fact]
    public void Constructor_InitializesWithCorrectParameters()
    {
        var blockManager = new KVCacheBlockManager(1000, 16, 128, 32, 32, _deviceId);
        var scheduler = new PagedAttentionScheduler(
            blockManager,
            maxSequencesInFlight: 50,
            reservedBlockRatio: 0.2
        );

        var stats = scheduler.GetStats();
        Assert.Equal(50, stats.MaxSequencesInFlight);
        Assert.Equal(200, stats.ReservedBlocks); // 20% of 1000
    }

    [Fact]
    public void CanAdmitRequest_AcceptsWhenMemoryAvailable()
    {
        var blockManager = new KVCacheBlockManager(1000, 16, 128, 32, 32, _deviceId);
        var scheduler = new PagedAttentionScheduler(blockManager);

        var decision = scheduler.CanAdmitRequest(
            estimatedMaxTokens: 100,
            currentSequenceCount: 0
        );

        Assert.True(decision.Accepted);
        Assert.Null(decision.RejectionReason);
    }

    [Fact]
    public void CanAdmitRequest_RejectsWhenSequencesLimitExceeded()
    {
        var blockManager = new KVCacheBlockManager(1000, 16, 128, 32, 32, _deviceId);
        var scheduler = new PagedAttentionScheduler(blockManager, maxSequencesInFlight: 10);

        var decision = scheduler.CanAdmitRequest(
            estimatedMaxTokens: 100,
            currentSequenceCount: 10
        );

        Assert.False(decision.Accepted);
        Assert.NotNull(decision.RejectionReason);
        Assert.Contains("Maximum sequences in flight exceeded", decision.RejectionReason);
    }

    [Fact]
    public void CanAdmitRequest_RejectsWhenMemoryInsufficient()
    {
        var blockManager = new KVCacheBlockManager(1000, 16, 128, 32, 32, _deviceId);
        var scheduler = new PagedAttentionScheduler(blockManager);

        // Allocate most blocks, leaving fewer than needed for the request
        for (int i = 0; i < 900; i++)
        {
            blockManager.AllocateBlock(i);
        }

        var decision = scheduler.CanAdmitRequest(
            estimatedMaxTokens: 200, // Needs 13 blocks
            currentSequenceCount: 0
        );

        Assert.False(decision.Accepted);
        Assert.NotNull(decision.RejectionReason);
        Assert.Contains("Insufficient memory", decision.RejectionReason);
    }

    [Fact]
    public void GetAvailableBlocks_ReturnsCorrectCount()
    {
        var blockManager = new KVCacheBlockManager(1000, 16, 128, 32, 32, _deviceId);
        var scheduler = new PagedAttentionScheduler(blockManager, reservedBlockRatio: 0.1);

        var initialAvailable = scheduler.GetAvailableBlocks();
        Assert.Equal(900, initialAvailable); // 1000 total - 100 reserved

        // Allocate some blocks
        blockManager.AllocateBlocks(1, 50);

        var availableAfterAllocation = scheduler.GetAvailableBlocks();
        Assert.Equal(850, availableAfterAllocation); // 900 - 50
    }

    [Fact]
    public void EstimateBlocksNeeded_CalculatesCorrectly()
    {
        var blockManager = new KVCacheBlockManager(1000, 16, 128, 32, 32, _deviceId);
        var scheduler = new PagedAttentionScheduler(blockManager);

        // Test via admission decision
        var decision1 = scheduler.CanAdmitRequest(16, 0); // 1 block
        Assert.True(decision1.Accepted);

        var decision2 = scheduler.CanAdmitRequest(32, 0); // 2 blocks
        Assert.True(decision2.Accepted);

        var decision3 = scheduler.CanAdmitRequest(17, 0); // 2 blocks (rounded up)
        Assert.True(decision3.Accepted);
    }

    [Fact]
    public void GetMemoryPressure_CalculatesCorrectly()
    {
        var blockManager = new KVCacheBlockManager(1000, 16, 128, 32, 32, _deviceId);
        var scheduler = new PagedAttentionScheduler(blockManager, reservedBlockRatio: 0.1);

        var pressure = scheduler.GetMemoryPressure();
        Assert.Equal(0.1, pressure); // 100/1000 reserved

        // Allocate blocks to increase pressure
        blockManager.AllocateBlocks(1, 400);

        var pressureAfterAllocation = scheduler.GetMemoryPressure();
        Assert.True(pressureAfterAllocation > pressure);
    }

    [Fact]
    public void ShouldApplyBackpressure_ReturnsTrueWhenPressureHigh()
    {
        var blockManager = new KVCacheBlockManager(1000, 16, 128, 32, 32, _deviceId);
        var scheduler = new PagedAttentionScheduler(blockManager, reservedBlockRatio: 0.1);

        // Allocate blocks to create high pressure
        blockManager.AllocateBlocks(1, 800);

        var shouldApply = scheduler.ShouldApplyBackpressure(threshold: 0.8);
        Assert.True(shouldApply);
    }

    [Fact]
    public void ShouldApplyBackpressure_ReturnsFalseWhenPressureLow()
    {
        var blockManager = new KVCacheBlockManager(1000, 16, 128, 32, 32, _deviceId);
        var scheduler = new PagedAttentionScheduler(blockManager);

        var shouldApply = scheduler.ShouldApplyBackpressure(threshold: 0.8);
        Assert.False(shouldApply);
    }

    [Fact]
    public void GetRecommendedMaxSequenceLength_ReturnsCorrectValue()
    {
        var blockManager = new KVCacheBlockManager(1000, 16, 128, 32, 32, _deviceId);
        var scheduler = new PagedAttentionScheduler(blockManager, reservedBlockRatio: 0.1);

        var recommended = scheduler.GetRecommendedMaxSequenceLength();
        Assert.Equal(900 * 16, recommended); // 900 blocks * 16 tokens per block
    }

    [Fact]
    public void GetStats_ReturnsCorrectStatistics()
    {
        var blockManager = new KVCacheBlockManager(1000, 16, 128, 32, 32, _deviceId);
        var scheduler = new PagedAttentionScheduler(blockManager, reservedBlockRatio: 0.1);

        var stats = scheduler.GetStats();

        Assert.Equal(1000, stats.TotalBlocks);
        Assert.Equal(1000, stats.FreeBlocks);
        Assert.Equal(0, stats.AllocatedBlocks);
        Assert.Equal(900, stats.AvailableBlocks);
        Assert.Equal(100, stats.ReservedBlocks);
        Assert.Equal(0, stats.ActiveSequences);
        Assert.Equal(0.1, stats.MemoryPressure);
    }

    [Fact]
    public void GetStats_ReflectsBlockAllocations()
    {
        var blockManager = new KVCacheBlockManager(1000, 16, 128, 32, 32, _deviceId);
        var scheduler = new PagedAttentionScheduler(blockManager, reservedBlockRatio: 0.1);
        blockManager.AllocateBlocks(1, 50);

        var stats = scheduler.GetStats();

        Assert.Equal(50, stats.AllocatedBlocks);
        Assert.Equal(950, stats.FreeBlocks);
        Assert.Equal(850, stats.AvailableBlocks);
        Assert.Equal(1, stats.ActiveSequences);
    }
}

/// <summary>
/// Tests for MemoryAwareScheduler
/// </summary>
public class MemoryAwareSchedulerTests
{

    [Fact]
    public void Constructor_InitializesCorrectly()
    {
        var blockManager = new KVCacheBlockManager(1000, 16, 128, 32, 32, _deviceId);
        var pagedScheduler = new PagedAttentionScheduler(blockManager);
        var scheduler = new MemoryAwareScheduler(pagedScheduler);

        Assert.Equal(0, scheduler.GetQueueSize());
    }

    [Fact]
    public void EnqueueRequest_AddsRequestToQueue()
    {
        var blockManager = new KVCacheBlockManager(1000, 16, 128, 32, 32, _deviceId);
        var pagedScheduler = new PagedAttentionScheduler(blockManager);
        var scheduler = new MemoryAwareScheduler(pagedScheduler);

        var request = new PendingRequest
        {
            RequestId = 1,
            EstimatedMaxTokens = 100
        };

        var result = scheduler.EnqueueRequest(request);

        Assert.True(result);
        Assert.Equal(1, scheduler.GetQueueSize());
    }

    [Fact]
    public void EnqueueRequest_RejectsWhenQueueFull()
    {
        var blockManager = new KVCacheBlockManager(1000, 16, 128, 32, 32, _deviceId);
        var pagedScheduler = new PagedAttentionScheduler(blockManager);
        var scheduler = new MemoryAwareScheduler(pagedScheduler, maxQueueSize: 100);

        // Fill the queue
        for (int i = 0; i < 100; i++)
        {
            scheduler.EnqueueRequest(new PendingRequest { RequestId = i, EstimatedMaxTokens = 100 });
        }

        // Try to add one more
        var result = scheduler.EnqueueRequest(new PendingRequest { RequestId = 100, EstimatedMaxTokens = 100 });

        Assert.False(result);
        Assert.Equal(100, scheduler.GetQueueSize());
    }

    [Fact]
    public void TryAdmitNextRequest_AdmitsWhenMemoryAvailable()
    {
        var blockManager = new KVCacheBlockManager(1000, 16, 128, 32, 32, _deviceId);
        var pagedScheduler = new PagedAttentionScheduler(blockManager);
        var scheduler = new MemoryAwareScheduler(pagedScheduler);

        var request = new PendingRequest
        {
            RequestId = 1,
            EstimatedMaxTokens = 100,
            QueuedAt = DateTime.UtcNow
        };

        scheduler.EnqueueRequest(request);

        var admitted = scheduler.TryAdmitNextRequest();

        Assert.NotNull(admitted);
        Assert.Equal(1, admitted.RequestId);
        Assert.Equal(0, scheduler.GetQueueSize());
    }

    [Fact]
    public void TryAdmitNextRequest_DoesNotAdmitWhenMemoryInsufficient()
    {
        var blockManager = new KVCacheBlockManager(1000, 16, 128, 32, 32, _deviceId);
        var pagedScheduler = new PagedAttentionScheduler(blockManager);
        var scheduler = new MemoryAwareScheduler(pagedScheduler);

        // Allocate blocks to create memory pressure
        blockManager.AllocateBlocks(1, 800);

        var request = new PendingRequest
        {
            RequestId = 1,
            EstimatedMaxTokens = 200, // Needs 13 blocks
            QueuedAt = DateTime.UtcNow
        };

        scheduler.EnqueueRequest(request);

        var admitted = scheduler.TryAdmitNextRequest();

        Assert.Null(admitted);
        Assert.Equal(1, scheduler.GetQueueSize());
    }

    [Fact]
    public void TryAdmitNextRequest_ReturnsNullWhenQueueEmpty()
    {
        var blockManager = new KVCacheBlockManager(1000, 16, 128, 32, 32, _deviceId);
        var pagedScheduler = new PagedAttentionScheduler(blockManager);
        var scheduler = new MemoryAwareScheduler(pagedScheduler);

        var admitted = scheduler.TryAdmitNextRequest();
        Assert.Null(admitted);
    }

    [Fact]
    public void GetQueueSize_ReturnsCorrectCount()
    {
        var blockManager = new KVCacheBlockManager(1000, 16, 128, 32, 32, _deviceId);
        var pagedScheduler = new PagedAttentionScheduler(blockManager);
        var scheduler = new MemoryAwareScheduler(pagedScheduler);

        Assert.Equal(0, scheduler.GetQueueSize());

        for (int i = 0; i < 5; i++)
        {
            scheduler.EnqueueRequest(new PendingRequest { RequestId = i, EstimatedMaxTokens = 100 });
        }

        Assert.Equal(5, scheduler.GetQueueSize());
    }

    [Fact]
    public void GetQueuedRequests_ReturnsAllRequests()
    {
        var blockManager = new KVCacheBlockManager(1000, 16, 128, 32, 32, _deviceId);
        var pagedScheduler = new PagedAttentionScheduler(blockManager);
        var scheduler = new MemoryAwareScheduler(pagedScheduler);

        var requests = new List<PendingRequest>();
        for (int i = 0; i < 3; i++)
        {
            var request = new PendingRequest
            {
                RequestId = i,
                EstimatedMaxTokens = 100,
                QueuedAt = DateTime.UtcNow
            };
            requests.Add(request);
            scheduler.EnqueueRequest(request);
        }

        var queued = scheduler.GetQueuedRequests();

        Assert.Equal(3, queued.Count);
        Assert.Equal(0, queued[0].RequestId);
        Assert.Equal(1, queued[1].RequestId);
        Assert.Equal(2, queued[2].RequestId);
    }

    [Fact]
    public void ShouldApplyBackpressure_DelegatesToPagedScheduler()
    {
        var blockManager = new KVCacheBlockManager(1000, 16, 128, 32, 32, _deviceId);
        var pagedScheduler = new PagedAttentionScheduler(blockManager);
        var scheduler = new MemoryAwareScheduler(pagedScheduler);

        // Allocate blocks to create high pressure
        blockManager.AllocateBlocks(1, 800);

        var shouldApply = scheduler.ShouldApplyBackpressure(threshold: 0.8);

        Assert.True(shouldApply);
    }

    [Fact]
    public void ClearQueue_RemovesAllRequests()
    {
        var blockManager = new KVCacheBlockManager(1000, 16, 128, 32, 32, _deviceId);
        var pagedScheduler = new PagedAttentionScheduler(blockManager);
        var scheduler = new MemoryAwareScheduler(pagedScheduler);

        for (int i = 0; i < 5; i++)
        {
            scheduler.EnqueueRequest(new PendingRequest { RequestId = i, EstimatedMaxTokens = 100 });
        }

        Assert.Equal(5, scheduler.GetQueueSize());

        scheduler.ClearQueue();

        Assert.Equal(0, scheduler.GetQueueSize());
    }

    [Fact]
    public void TryAdmitNextRequest_AdmitsInOrder()
    {
        var blockManager = new KVCacheBlockManager(1000, 16, 128, 32, 32, _deviceId);
        var pagedScheduler = new PagedAttentionScheduler(blockManager);
        var scheduler = new MemoryAwareScheduler(pagedScheduler);

        // Add multiple requests
        for (int i = 0; i < 3; i++)
        {
            var request = new PendingRequest
            {
                RequestId = i,
                EstimatedMaxTokens = 50,
                QueuedAt = DateTime.UtcNow
            };
            scheduler.EnqueueRequest(request);
        }

        // Admit them one by one
        for (int i = 0; i < 3; i++)
        {
            var admitted = scheduler.TryAdmitNextRequest();
            Assert.NotNull(admitted);
            Assert.Equal(i, admitted.RequestId);
        }

        // Queue should be empty
        Assert.Equal(0, scheduler.GetQueueSize());
    }
}

public class AdmissionDecisionTests
{
    [Fact]
    public void CreateAccepted_CreatesSuccessfulDecision()
    {
        var decision = AdmissionDecision.CreateAccepted(100);

        Assert.True(decision.Accepted);
        Assert.Null(decision.RejectionReason);
        Assert.Equal(100, decision.RemainingBlocks);
    }

    [Fact]
    public void CreateRejected_CreatesFailedDecision()
    {
        var decision = AdmissionDecision.CreateRejected("Test reason");

        Assert.False(decision.Accepted);
        Assert.NotNull(decision.RejectionReason);
        Assert.Equal("Test reason", decision.RejectionReason);
        Assert.Equal(0, decision.RemainingBlocks);
    }
}
