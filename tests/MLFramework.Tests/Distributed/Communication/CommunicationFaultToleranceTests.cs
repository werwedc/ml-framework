namespace MLFramework.Tests.Distributed.Communication;

using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;
using System;
using System.Threading.Tasks;

[TestClass]
public class CommunicationFaultToleranceTests
{
    #region Exception Tests

    [TestMethod]
    public void TestCommunicationException_BasicConstructor()
    {
        var ex = new CommunicationException("Test error");
        Assert.AreEqual("Test error", ex.Message);
        Assert.IsNull(ex.Rank);
        Assert.IsNull(ex.BackendName);
    }

    [TestMethod]
    public void TestCommunicationException_WithRankAndBackend()
    {
        var ex = new CommunicationException("Test error", 5, "NCCL");
        Assert.AreEqual("Test error", ex.Message);
        Assert.AreEqual(5, ex.Rank);
        Assert.AreEqual("NCCL", ex.BackendName);
    }

    [TestMethod]
    public void TestCommunicationException_WithInnerException()
    {
        var inner = new InvalidOperationException("Inner error");
        var ex = new CommunicationException("Test error", inner);
        Assert.AreEqual("Test error", ex.Message);
        Assert.AreEqual(inner, ex.InnerException);
    }

    [TestMethod]
    public void TestCommunicationException_FullConstructor()
    {
        var inner = new InvalidOperationException("Inner error");
        var ex = new CommunicationException("Test error", inner, 5, "NCCL");
        Assert.AreEqual("Test error", ex.Message);
        Assert.AreEqual(inner, ex.InnerException);
        Assert.AreEqual(5, ex.Rank);
        Assert.AreEqual("NCCL", ex.BackendName);
    }

    [TestMethod]
    public void TestCommunicationTimeoutException_BasicConstructor()
    {
        var ex = new CommunicationTimeoutException("Timeout error", TimeSpan.FromSeconds(10));
        Assert.AreEqual("Timeout error", ex.Message);
        Assert.AreEqual(TimeSpan.FromSeconds(10), ex.TimeoutDuration);
    }

    [TestMethod]
    public void TestCommunicationTimeoutException_WithRankAndBackend()
    {
        var ex = new CommunicationTimeoutException("Timeout error", TimeSpan.FromSeconds(10), 3, "Gloo");
        Assert.AreEqual("Timeout error", ex.Message);
        Assert.AreEqual(TimeSpan.FromSeconds(10), ex.TimeoutDuration);
        Assert.AreEqual(3, ex.Rank);
        Assert.AreEqual("Gloo", ex.BackendName);
    }

    [TestMethod]
    public void TestRankMismatchException_BasicConstructor()
    {
        var ex = new RankMismatchException("Rank mismatch", 0, 1);
        Assert.AreEqual("Rank mismatch", ex.Message);
        Assert.AreEqual(0, ex.ExpectedRank);
        Assert.AreEqual(1, ex.ActualRank);
    }

    [TestMethod]
    public void TestRankMismatchException_WithRankAndBackend()
    {
        var ex = new RankMismatchException("Rank mismatch", 0, 1, 2, "NCCL");
        Assert.AreEqual("Rank mismatch", ex.Message);
        Assert.AreEqual(0, ex.ExpectedRank);
        Assert.AreEqual(1, ex.ActualRank);
        Assert.AreEqual(2, ex.Rank);
        Assert.AreEqual("NCCL", ex.BackendName);
    }

    #endregion

    #region Communicator Error Handling Tests

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public void TestCommunicator_InvalidRank_ThrowsException()
    {
        var comm = new MockCommunicator(worldSize: 2, rank: 5);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public void TestCommunicator_NegativeRank_ThrowsException()
    {
        var comm = new MockCommunicator(worldSize: 2, rank: -1);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void TestCommunicator_ZeroWorldSize_ThrowsException()
    {
        var comm = new MockCommunicator(worldSize: 0, rank: 0);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void TestCommunicator_NegativeWorldSize_ThrowsException()
    {
        var comm = new MockCommunicator(worldSize: -1, rank: 0);
    }

    #endregion

    #region Operation Error Handling Tests

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public async Task TestBroadcast_InvalidRootRank_ThrowsException()
    {
        var comm = new MockCommunicator(worldSize: 4, rank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        await comm.BroadcastAsync(tensor, 10);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public async Task TestBroadcast_NegativeRootRank_ThrowsException()
    {
        var comm = new MockCommunicator(worldSize: 4, rank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        await comm.BroadcastAsync(tensor, -1);
    }

    [TestMethod]
    public async Task TestAllReduce_NullTensor_HandlesGracefully()
    {
        var comm = new MockCommunicator(worldSize: 2, rank: 0);

        // This should either throw gracefully or handle null
        try
        {
            await comm.AllReduceAsync(null!, ReduceOperation.Sum);
            Assert.Fail("Expected an exception for null tensor");
        }
        catch (NullReferenceException)
        {
            // Expected
        }

        comm.Dispose();
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public async Task TestAllGather_InvalidDimension_ThrowsException()
    {
        var comm = new MockCommunicator(worldSize: 2, rank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        await comm.AllGatherAsync(tensor, dim: 10);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public async Task TestAllGather_NegativeDimension_ThrowsException()
    {
        var comm = new MockCommunicator(worldSize: 2, rank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        await comm.AllGatherAsync(tensor, dim: -2);
    }

    #endregion

    #region Process Group Error Handling Tests

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void TestProcessGroup_NullCommunicator_ThrowsException()
    {
        var group = new ProcessGroup(null!, new System.Collections.Generic.List<int> { 0, 1 }, myGlobalRank: 0);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void TestProcessGroup_NullRanks_ThrowsException()
    {
        var comm = new MockCommunicator(worldSize: 2, rank: 0);
        var group = new ProcessGroup(comm, null!, myGlobalRank: 0);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void TestProcessGroup_EmptyRanks_ThrowsException()
    {
        var comm = new MockCommunicator(worldSize: 2, rank: 0);
        var group = new ProcessGroup(comm, new System.Collections.Generic.List<int>(), myGlobalRank: 0);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void TestProcessGroup_RankOutOfBounds_ThrowsException()
    {
        var comm = new MockCommunicator(worldSize: 2, rank: 0);
        var group = new ProcessGroup(comm, new System.Collections.Generic.List<int> { 0, 5 }, myGlobalRank: 0);
    }

    [TestMethod]
    public void TestProcessGroup_DuplicateRanks_AreRemoved()
    {
        var comm = new MockCommunicator(worldSize: 4, rank: 0);
        var group = new ProcessGroup(comm, new System.Collections.Generic.List<int> { 0, 1, 1, 2 }, myGlobalRank: 0);

        // Duplicates should be removed
        Assert.AreEqual(3, group.WorldSize);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public void TestProcessGroup_GetGlobalRank_InvalidLocalRank_ThrowsException()
    {
        var comm = new MockCommunicator(worldSize: 4, rank: 0);
        var group = new ProcessGroup(comm, new System.Collections.Generic.List<int> { 0, 1, 2 }, myGlobalRank: 0);
        group.GetGlobalRank(5);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public void TestProcessGroup_GetGlobalRank_NegativeLocalRank_ThrowsException()
    {
        var comm = new MockCommunicator(worldSize: 4, rank: 0);
        var group = new ProcessGroup(comm, new System.Collections.Generic.List<int> { 0, 1, 2 }, myGlobalRank: 0);
        group.GetGlobalRank(-1);
    }

    #endregion

    #edge Case Tests

    [TestMethod]
    public async Task TestSingleRank_Communicator_Works()
    {
        var comm = new MockCommunicator(worldSize: 1, rank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f });

        var result1 = await comm.AllReduceAsync(tensor, ReduceOperation.Sum);
        var result2 = await comm.AllGatherAsync(tensor);
        var result3 = await comm.BroadcastAsync(tensor, 0);

        CollectionAssert.AreEqual(tensor.Data, result1.Data);
        CollectionAssert.AreEqual(tensor.Data, result2.Data);
        CollectionAssert.AreEqual(tensor.Data, result3.Data);

        comm.Dispose();
    }

    [TestMethod]
    public async Task TestLargeWorldSize_Works()
    {
        var comm = new MockCommunicator(worldSize: 100, rank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f });

        var result = await comm.AllReduceAsync(tensor, ReduceOperation.Sum);

        Assert.IsNotNull(result);
        comm.Dispose();
    }

    [TestMethod]
    public async Task TestEmptyTensor_HandlesGracefully()
    {
        var comm = new MockCommunicator(worldSize: 2, rank: 0);
        var tensor = Tensor.FromArray(new float[] { });

        var result = await comm.AllReduceAsync(tensor, ReduceOperation.Sum);

        Assert.IsNotNull(result);
        comm.Dispose();
    }

    [TestMethod]
    public async Task TestLargeTensor_HandlesGracefully()
    {
        var comm = new MockCommunicator(worldSize: 2, rank: 0);
        var size = 1000000;
        var data = new float[size];
        for (int i = 0; i < size; i++)
        {
            data[i] = (float)i;
        }
        var tensor = Tensor.FromArray(data);

        var result = await comm.AllReduceAsync(tensor, ReduceOperation.Sum);

        Assert.IsNotNull(result);
        Assert.AreEqual(size, result.Shape.TotalSize);
        comm.Dispose();
    }

    #endregion

    #region Configuration Tests

    [TestMethod]
    public void TestCommunicationConfig_DefaultValues()
    {
        var config = new CommunicationConfig();
        Assert.AreEqual(300000, config.TimeoutMs);
        Assert.IsFalse(config.EnableLogging);
        Assert.IsTrue(config.UsePinnedMemory);
        Assert.AreEqual(3, config.MaxRetries);
        Assert.AreEqual(100, config.RetryDelayMs);
    }

    [TestMethod]
    public void TestCommunicationConfig_CustomValues()
    {
        var config = new CommunicationConfig
        {
            TimeoutMs = 60000,
            EnableLogging = true,
            UsePinnedMemory = false,
            MaxRetries = 5,
            RetryDelayMs = 200
        };

        Assert.AreEqual(60000, config.TimeoutMs);
        Assert.IsTrue(config.EnableLogging);
        Assert.IsFalse(config.UsePinnedMemory);
        Assert.AreEqual(5, config.MaxRetries);
        Assert.AreEqual(200, config.RetryDelayMs);
    }

    [TestMethod]
    public void TestCommunicationConfig_ZeroTimeout_Handles()
    {
        var config = new CommunicationConfig
        {
            TimeoutMs = 0
        };

        Assert.AreEqual(0, config.TimeoutMs);
    }

    [TestMethod]
    public void TestCommunicationConfig_NegativeRetries_Handles()
    {
        var config = new CommunicationConfig
        {
            MaxRetries = -1
        };

        Assert.AreEqual(-1, config.MaxRetries);
    }

    #endregion

    #region Dispose Tests

    [TestMethod]
    public void TestCommunicator_MultipleDispose_DoesNotThrow()
    {
        var comm = new MockCommunicator(worldSize: 2, rank: 0);
        comm.Dispose();
        comm.Dispose(); // Should not throw
    }

    [TestMethod]
    public void TestProcessGroup_MultipleDispose_DoesNotThrow()
    {
        var comm = new MockCommunicator(worldSize: 2, rank: 0);
        var group = new ProcessGroup(comm, new System.Collections.Generic.List<int> { 0, 1 }, myGlobalRank: 0);
        group.Dispose();
        group.Dispose(); // Should not throw
    }

    [TestMethod]
    public void TestDisposal_Cleanup()
    {
        var comm1 = new MockCommunicator(worldSize: 2, rank: 0);
        var comm2 = new MockCommunicator(worldSize: 2, rank: 1);

        comm1.Dispose();
        comm2.Dispose();

        // Should not throw when disposed
    }

    #endregion

    #region Concurrent Access Tests

    [TestMethod]
    public async Task TestConcurrentOperations_DoNotConflict()
    {
        var comm = new MockCommunicator(worldSize: 2, rank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });

        var task1 = comm.AllReduceAsync(tensor, ReduceOperation.Sum);
        var task2 = comm.AllReduceAsync(tensor, ReduceOperation.Max);
        var task3 = comm.AllReduceAsync(tensor, ReduceOperation.Min);
        var task4 = comm.AllReduceAsync(tensor, ReduceOperation.Avg);

        await Task.WhenAll(task1, task2, task3, task4);

        Assert.IsTrue(task1.IsCompleted);
        Assert.IsTrue(task2.IsCompleted);
        Assert.IsTrue(task3.IsCompleted);
        Assert.IsTrue(task4.IsCompleted);

        comm.Dispose();
    }

    #endregion
}
