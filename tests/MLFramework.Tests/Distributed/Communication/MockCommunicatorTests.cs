namespace MLFramework.Tests.Distributed.Communication;

using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

[TestClass]
public class MockCommunicatorTests
{
    [TestMethod]
    public void Constructor_ValidParameters_CreatesCommunicator()
    {
        var comm = new MockCommunicator(worldSize: 4, rank: 2);

        Assert.AreEqual(4, comm.WorldSize);
        Assert.AreEqual(2, comm.Rank);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void Constructor_ZeroWorldSize_ThrowsException()
    {
        var comm = new MockCommunicator(worldSize: 0, rank: 0);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void Constructor_NegativeWorldSize_ThrowsException()
    {
        var comm = new MockCommunicator(worldSize: -1, rank: 0);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void Constructor_RankOutOfBounds_ThrowsException()
    {
        var comm = new MockCommunicator(worldSize: 2, rank: 5);
    }

    #region All-Reduce Tests

    [TestMethod]
    public async Task AllReduce_Sum_SumsTensors()
    {
        var comm1 = new MockCommunicator(worldSize: 2, rank: 0);
        var comm2 = new MockCommunicator(worldSize: 2, rank: 1);

        var tensor1 = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        var tensor2 = Tensor.FromArray(new float[] { 3.0f, 4.0f });

        var result1 = await comm1.AllReduceAsync(tensor1, ReduceOperation.Sum);
        var result2 = await comm2.AllReduceAsync(tensor2, ReduceOperation.Sum);

        // Both ranks should get the sum: [4.0, 6.0]
        CollectionAssert.AreEqual(new float[] { 4.0f, 6.0f }, result1.Data);
        CollectionAssert.AreEqual(new float[] { 4.0f, 6.0f }, result2.Data);
    }

    [TestMethod]
    public async Task AllReduce_Max_ReturnsMaximum()
    {
        var comm1 = new MockCommunicator(worldSize: 3, rank: 0);
        var comm2 = new MockCommunicator(worldSize: 3, rank: 1);
        var comm3 = new MockCommunicator(worldSize: 3, rank: 2);

        var tensor1 = Tensor.FromArray(new float[] { 1.0f, 5.0f });
        var tensor2 = Tensor.FromArray(new float[] { 3.0f, 2.0f });
        var tensor3 = Tensor.FromArray(new float[] { 2.0f, 8.0f });

        var result1 = await comm1.AllReduceAsync(tensor1, ReduceOperation.Max);
        var result2 = await comm2.AllReduceAsync(tensor2, ReduceOperation.Max);
        var result3 = await comm3.AllReduceAsync(tensor3, ReduceOperation.Max);

        // Max: [3.0, 8.0]
        var expected = new float[] { 3.0f, 8.0f };
        CollectionAssert.AreEqual(expected, result1.Data);
        CollectionAssert.AreEqual(expected, result2.Data);
        CollectionAssert.AreEqual(expected, result3.Data);
    }

    [TestMethod]
    public async Task AllReduce_Min_ReturnsMinimum()
    {
        var comm1 = new MockCommunicator(worldSize: 3, rank: 0);
        var comm2 = new MockCommunicator(worldSize: 3, rank: 1);
        var comm3 = new MockCommunicator(worldSize: 3, rank: 2);

        var tensor1 = Tensor.FromArray(new float[] { 1.0f, 5.0f });
        var tensor2 = Tensor.FromArray(new float[] { 3.0f, 2.0f });
        var tensor3 = Tensor.FromArray(new float[] { 2.0f, 8.0f });

        var result1 = await comm1.AllReduceAsync(tensor1, ReduceOperation.Min);
        var result2 = await comm2.AllReduceAsync(tensor2, ReduceOperation.Min);
        var result3 = await comm3.AllReduceAsync(tensor3, ReduceOperation.Min);

        // Min: [1.0, 2.0]
        var expected = new float[] { 1.0f, 2.0f };
        CollectionAssert.AreEqual(expected, result1.Data);
        CollectionAssert.AreEqual(expected, result2.Data);
        CollectionAssert.AreEqual(expected, result3.Data);
    }

    [TestMethod]
    public async Task AllReduce_Avg_ReturnsAverage()
    {
        var comm1 = new MockCommunicator(worldSize: 2, rank: 0);
        var comm2 = new MockCommunicator(worldSize: 2, rank: 1);

        var tensor1 = Tensor.FromArray(new float[] { 2.0f, 4.0f });
        var tensor2 = Tensor.FromArray(new float[] { 4.0f, 6.0f });

        var result1 = await comm1.AllReduceAsync(tensor1, ReduceOperation.Avg);
        var result2 = await comm2.AllReduceAsync(tensor2, ReduceOperation.Avg);

        // Avg: [3.0, 5.0]
        var expected = new float[] { 3.0f, 5.0f };
        CollectionAssert.AreEqual(expected, result1.Data);
        CollectionAssert.AreEqual(expected, result2.Data);
    }

    [TestMethod]
    public async Task AllReduce_Product_MultipliesTensors()
    {
        var comm1 = new MockCommunicator(worldSize: 2, rank: 0);
        var comm2 = new MockCommunicator(worldSize: 2, rank: 1);

        var tensor1 = Tensor.FromArray(new float[] { 2.0f, 3.0f });
        var tensor2 = Tensor.FromArray(new float[] { 4.0f, 5.0f });

        var result1 = await comm1.AllReduceAsync(tensor1, ReduceOperation.Product);
        var result2 = await comm2.AllReduceAsync(tensor2, ReduceOperation.Product);

        // Product: [8.0, 15.0]
        var expected = new float[] { 8.0f, 15.0f };
        CollectionAssert.AreEqual(expected, result1.Data);
        CollectionAssert.AreEqual(expected, result2.Data);
    }

    [TestMethod]
    public async Task AllReduce_SingleRank_ReturnsSameTensor()
    {
        var comm = new MockCommunicator(worldSize: 1, rank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f });

        var result = await comm.AllReduceAsync(tensor, ReduceOperation.Sum);

        CollectionAssert.AreEqual(tensor.Data, result.Data);
    }

    #endregion

    #region All-Gather Tests

    [TestMethod]
    public async Task AllGather_ConcatenatesTensors()
    {
        var comm1 = new MockCommunicator(worldSize: 2, rank: 0);
        var comm2 = new MockCommunicator(worldSize: 2, rank: 1);

        var tensor1 = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        var tensor2 = Tensor.FromArray(new float[] { 3.0f, 4.0f });

        var result1 = await comm1.AllGatherAsync(tensor1, dim: 0);
        var result2 = await comm2.AllGatherAsync(tensor2, dim: 0);

        // Concatenated: [1.0, 2.0, 3.0, 4.0]
        var expected = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        CollectionAssert.AreEqual(expected, result1.Data);
        CollectionAssert.AreEqual(expected, result2.Data);
    }

    [TestMethod]
    public async Task AllGather_NegativeDimension_WorksCorrectly()
    {
        var comm1 = new MockCommunicator(worldSize: 2, rank: 0);
        var comm2 = new MockCommunicator(worldSize: 2, rank: 1);

        var tensor1 = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        var tensor2 = Tensor.FromArray(new float[] { 3.0f, 4.0f });

        // Use dim=-1 (last dimension)
        var result1 = await comm1.AllGatherAsync(tensor1, dim: -1);
        var result2 = await comm2.AllGatherAsync(tensor2, dim: -1);

        var expected = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        CollectionAssert.AreEqual(expected, result1.Data);
        CollectionAssert.AreEqual(expected, result2.Data);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public async Task AllGather_InvalidDimension_ThrowsException()
    {
        var comm = new MockCommunicator(worldSize: 1, rank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });

        await comm.AllGatherAsync(tensor, dim: 5);
    }

    [TestMethod]
    public async Task AllGather_SingleRank_ReturnsSameTensor()
    {
        var comm = new MockCommunicator(worldSize: 1, rank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });

        var result = await comm.AllGatherAsync(tensor, dim: 0);

        CollectionAssert.AreEqual(tensor.Data, result.Data);
    }

    #endregion

    #region Reduce-Scatter Tests

    [TestMethod]
    public async Task ReduceScatter_ReducesAndScatters()
    {
        var comm1 = new MockCommunicator(worldSize: 2, rank: 0);
        var comm2 = new MockCommunicator(worldSize: 2, rank: 1);

        var tensor1 = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        var tensor2 = Tensor.FromArray(new float[] { 5.0f, 6.0f, 7.0f, 8.0f });

        var result1 = await comm1.ReduceScatterAsync(tensor1, ReduceOperation.Sum);
        var result2 = await comm2.ReduceScatterAsync(tensor2, ReduceOperation.Sum);

        // Sum: [6.0, 8.0, 10.0, 12.0], then scatter
        // Rank 0 gets: [6.0, 8.0]
        // Rank 1 gets: [10.0, 12.0]
        CollectionAssert.AreEqual(new float[] { 6.0f, 8.0f }, result1.Data);
        CollectionAssert.AreEqual(new float[] { 10.0f, 12.0f }, result2.Data);
    }

    [TestMethod]
    [ExpectedException(typeof(InvalidOperationException))]
    public async Task ReduceScatter_NonDivisibleDimension_ThrowsException()
    {
        var comm1 = new MockCommunicator(worldSize: 2, rank: 0);
        var comm2 = new MockCommunicator(worldSize: 2, rank: 1);

        // Tensor with odd number of elements (5)
        var tensor1 = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });
        var tensor2 = Tensor.FromArray(new float[] { 6.0f, 7.0f, 8.0f, 9.0f, 10.0f });

        await comm1.ReduceScatterAsync(tensor1, ReduceOperation.Sum);
    }

    #endregion

    #region Broadcast Tests

    [TestMethod]
    public async Task Broadcast_RootBroadcastsToAll()
    {
        var comm1 = new MockCommunicator(worldSize: 3, rank: 0);
        var comm2 = new MockCommunicator(worldSize: 3, rank: 1);
        var comm3 = new MockCommunicator(worldSize: 3, rank: 2);

        var rootTensor = Tensor.FromArray(new float[] { 10.0f, 20.0f });
        var otherTensor = Tensor.FromArray(new float[] { 0.0f, 0.0f });

        var result1 = await comm1.BroadcastAsync(rootTensor, root: 0);
        var result2 = await comm2.BroadcastAsync(otherTensor, root: 0);
        var result3 = await comm3.BroadcastAsync(otherTensor, root: 0);

        var expected = new float[] { 10.0f, 20.0f };
        CollectionAssert.AreEqual(expected, result1.Data);
        CollectionAssert.AreEqual(expected, result2.Data);
        CollectionAssert.AreEqual(expected, result3.Data);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public async Task Broadcast_InvalidRootRank_ThrowsException()
    {
        var comm = new MockCommunicator(worldSize: 2, rank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });

        await comm.BroadcastAsync(tensor, root: 5);
    }

    #endregion

    #region Barrier Tests

    [TestMethod]
    public async TaskBarrier_CompletesSuccessfully()
    {
        var comm = new MockCommunicator(worldSize: 2, rank: 0);

        await comm.BarrierAsync();

        // Should complete without throwing
    }

    #endregion

    #region Dispose Tests

    [TestMethod]
    public void Dispose_CleansUpResources()
    {
        var comm = new MockCommunicator(worldSize: 2, rank: 0);

        comm.Dispose();

        // Should not throw when disposed
        comm.Dispose();
    }

    #endregion
}

[TestClass]
public class CommunicatorFactoryTests
{
    [TestMethod]
    public void Create_MockBackend_ReturnsMockCommunicator()
    {
        var config = new Dictionary<string, object>
        {
            ["world_size"] = 4,
            ["rank"] = 2
        };

        var comm = CommunicatorFactory.Create("mock", config);

        Assert.IsInstanceOfType(comm, typeof(MockCommunicator));
        Assert.AreEqual(4, comm.WorldSize);
        Assert.AreEqual(2, comm.Rank);
    }

    [TestMethod]
    public void Create_MockBackend_DefaultConfig_ReturnsDefaultValues()
    {
        var comm = CommunicatorFactory.Create("mock");

        Assert.IsInstanceOfType(comm, typeof(MockCommunicator));
        Assert.AreEqual(1, comm.WorldSize);
        Assert.AreEqual(0, comm.Rank);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void Create_UnknownBackend_ThrowsException()
    {
        CommunicatorFactory.Create("unknown_backend");
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void Create_NullBackend_ThrowsException()
    {
        CommunicatorFactory.Create(null!);
    }

    [TestMethod]
    [ExpectedException(typeof(NotImplementedException))]
    public void Create_NCCLBackend_ThrowsNotImplemented()
    {
        var config = new Dictionary<string, object>
        {
            ["world_size"] = 2,
            ["rank"] = 0
        };

        CommunicatorFactory.Create("nccl", config);
    }
}

[TestClass]
public class ProcessGroupTests
{
    [TestMethod]
    public void Constructor_ValidParameters_CreatesProcessGroup()
    {
        var globalComm = new MockCommunicator(worldSize: 4, rank: 2);
        var group = new ProcessGroup(globalComm, new List<int> { 0, 2, 3 }, myGlobalRank: 2);

        Assert.IsTrue(group.InGroup);
        Assert.AreEqual(3, group.WorldSize);
        Assert.AreEqual(1, group.LocalRank);
    }

    [TestMethod]
    public void Constructor_NotInGroup_SetsPropertiesCorrectly()
    {
        var globalComm = new MockCommunicator(worldSize: 4, rank: 1);
        var group = new ProcessGroup(globalComm, new List<int> { 0, 2, 3 }, myGlobalRank: 1);

        Assert.IsFalse(group.InGroup);
        Assert.AreEqual(0, group.WorldSize);
        Assert.AreEqual(-1, group.LocalRank);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void Constructor_NullCommunicator_ThrowsException()
    {
        var group = new ProcessGroup(null!, new List<int> { 0, 1 }, myGlobalRank: 0);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void Constructor_NullRanks_ThrowsException()
    {
        var globalComm = new MockCommunicator(worldSize: 2, rank: 0);
        var group = new ProcessGroup(globalComm, null!, myGlobalRank: 0);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void Constructor_EmptyRanks_ThrowsException()
    {
        var globalComm = new MockCommunicator(worldSize: 2, rank: 0);
        var group = new ProcessGroup(globalComm, new List<int>(), myGlobalRank: 0);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void Constructor_RankOutOfBounds_ThrowsException()
    {
        var globalComm = new MockCommunicator(worldSize: 2, rank: 0);
        var group = new ProcessGroup(globalComm, new List<int> { 0, 5 }, myGlobalRank: 0);
    }

    [TestMethod]
    public void GetGlobalRank_ValidLocalRank_ReturnsCorrectRank()
    {
        var globalComm = new MockCommunicator(worldSize: 4, rank: 2);
        var group = new ProcessGroup(globalComm, new List<int> { 0, 2, 3 }, myGlobalRank: 2);

        Assert.AreEqual(0, group.GetGlobalRank(0));
        Assert.AreEqual(2, group.GetGlobalRank(1));
        Assert.AreEqual(3, group.GetGlobalRank(2));
    }

    [TestMethod]
    public void GetLocalRank_ValidGlobalRank_ReturnsCorrectRank()
    {
        var globalComm = new MockCommunicator(worldSize: 4, rank: 2);
        var group = new ProcessGroup(globalComm, new List<int> { 0, 2, 3 }, myGlobalRank: 2);

        Assert.AreEqual(0, group.GetLocalRank(0));
        Assert.AreEqual(1, group.GetLocalRank(2));
        Assert.AreEqual(2, group.GetLocalRank(3));
    }

    [TestMethod]
    public void GetLocalRank_NotInGroup_ReturnsNegativeOne()
    {
        var globalComm = new MockCommunicator(worldSize: 4, rank: 2);
        var group = new ProcessGroup(globalComm, new List<int> { 0, 2, 3 }, myGlobalRank: 2);

        Assert.AreEqual(-1, group.GetLocalRank(1));
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public void GetGlobalRank_InvalidLocalRank_ThrowsException()
    {
        var globalComm = new MockCommunicator(worldSize: 4, rank: 2);
        var group = new ProcessGroup(globalComm, new List<int> { 0, 2, 3 }, myGlobalRank: 2);

        group.GetGlobalRank(5);
    }

    [TestMethod]
    public void AllReduceAsync_NotInGroup_ReturnsUnchangedTensor()
    {
        var globalComm = new MockCommunicator(worldSize: 4, rank: 1);
        var group = new ProcessGroup(globalComm, new List<int> { 0, 2, 3 }, myGlobalRank: 1);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });

        var task = group.AllReduceAsync(tensor, ReduceOperation.Sum);

        // Should return unchanged tensor
        var result = task.Result;
        CollectionAssert.AreEqual(tensor.Data, result.Data);
    }

    [TestMethod]
    public void AllGatherAsync_NotInGroup_ReturnsUnchangedTensor()
    {
        var globalComm = new MockCommunicator(worldSize: 4, rank: 1);
        var group = new ProcessGroup(globalComm, new List<int> { 0, 2, 3 }, myGlobalRank: 1);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });

        var task = group.AllGatherAsync(tensor, dim: 0);

        // Should return unchanged tensor
        var result = task.Result;
        CollectionAssert.AreEqual(tensor.Data, result.Data);
    }
}
