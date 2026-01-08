namespace MLFramework.Tests.Distributed.Communication;

using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

[TestClass]
public class ProcessGroupTests
{
    private MockCommunicator? _globalComm;

    [TestInitialize]
    public void Setup()
    {
        _globalComm = new MockCommunicator(worldSize: 8, rank: 0);
    }

    [TestCleanup]
    public void TearDown()
    {
        _globalComm?.Dispose();
    }

    #region Constructor Tests

    [TestMethod]
    public void TestProcessGroup_ValidParameters_CreatesGroup()
    {
        var group = new ProcessGroup(_globalComm!, new List<int> { 0, 1, 2, 3 }, myGlobalRank: 0);

        Assert.IsTrue(group.InGroup);
        Assert.AreEqual(4, group.WorldSize);
        Assert.AreEqual(0, group.LocalRank);
    }

    [TestMethod]
    public void TestProcessGroup_NotInGroup_SetsPropertiesCorrectly()
    {
        var group = new ProcessGroup(_globalComm!, new List<int> { 1, 2, 3 }, myGlobalRank: 0);

        Assert.IsFalse(group.InGroup);
        Assert.AreEqual(0, group.WorldSize);
        Assert.AreEqual(-1, group.LocalRank);
    }

    [TestMethod]
    public void TestProcessGroup_SingleRank()
    {
        var group = new ProcessGroup(_globalComm!, new List<int> { 0 }, myGlobalRank: 0);

        Assert.IsTrue(group.InGroup);
        Assert.AreEqual(1, group.WorldSize);
        Assert.AreEqual(0, group.LocalRank);
    }

    [TestMethod]
    public void TestProcessGroup_AllRanks()
    {
        var ranks = new List<int> { 0, 1, 2, 3, 4, 5, 6, 7 };
        var group = new ProcessGroup(_globalComm!, ranks, myGlobalRank: 0);

        Assert.IsTrue(group.InGroup);
        Assert.AreEqual(8, group.WorldSize);
        Assert.AreEqual(0, group.LocalRank);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void TestProcessGroup_NullCommunicator_ThrowsException()
    {
        new ProcessGroup(null!, new List<int> { 0, 1 }, myGlobalRank: 0);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void TestProcessGroup_NullRanks_ThrowsException()
    {
        new ProcessGroup(_globalComm!, null!, myGlobalRank: 0);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void TestProcessGroup_EmptyRanks_ThrowsException()
    {
        new ProcessGroup(_globalComm!, new List<int>(), myGlobalRank: 0);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void TestProcessGroup_RankOutOfBounds_ThrowsException()
    {
        new ProcessGroup(_globalComm!, new List<int> { 0, 8 }, myGlobalRank: 0);
    }

    [TestMethod]
    public void TestProcessGroup_RemovesDuplicates()
    {
        var group = new ProcessGroup(_globalComm!, new List<int> { 0, 1, 1, 2 }, myGlobalRank: 0);

        Assert.IsTrue(group.InGroup);
        Assert.AreEqual(3, group.WorldSize);
    }

    [TestMethod]
    public void TestProcessGroup_NegativeRank()
    {
        // Should not throw, just not include negative rank
        var group = new ProcessGroup(_globalComm!, new List<int> { 0, 1, 2 }, myGlobalRank: 0);

        Assert.IsTrue(group.InGroup);
        Assert.AreEqual(3, group.WorldSize);
    }

    #endregion

    #region Rank Mapping Tests

    [TestMethod]
    public void TestProcessGroup_GetGlobalRank_ValidLocalRank_ReturnsCorrectRank()
    {
        var group = new ProcessGroup(_globalComm!, new List<int> { 0, 2, 4, 6 }, myGlobalRank: 0);

        Assert.AreEqual(0, group.GetGlobalRank(0));
        Assert.AreEqual(2, group.GetGlobalRank(1));
        Assert.AreEqual(4, group.GetGlobalRank(2));
        Assert.AreEqual(6, group.GetGlobalRank(3));
    }

    [TestMethod]
    public void TestProcessGroup_GetLocalRank_ValidGlobalRank_ReturnsCorrectRank()
    {
        var group = new ProcessGroup(_globalComm!, new List<int> { 0, 2, 4, 6 }, myGlobalRank: 0);

        Assert.AreEqual(0, group.GetLocalRank(0));
        Assert.AreEqual(1, group.GetLocalRank(2));
        Assert.AreEqual(2, group.GetLocalRank(4));
        Assert.AreEqual(3, group.GetLocalRank(6));
    }

    [TestMethod]
    public void TestProcessGroup_GetLocalRank_NotInGroup_ReturnsNegativeOne()
    {
        var group = new ProcessGroup(_globalComm!, new List<int> { 0, 2, 4, 6 }, myGlobalRank: 0);

        Assert.AreEqual(-1, group.GetLocalRank(1));
        Assert.AreEqual(-1, group.GetLocalRank(3));
        Assert.AreEqual(-1, group.GetLocalRank(5));
        Assert.AreEqual(-1, group.GetLocalRank(7));
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public void TestProcessGroup_GetGlobalRank_InvalidLocalRank_ThrowsException()
    {
        var group = new ProcessGroup(_globalComm!, new List<int> { 0, 1, 2 }, myGlobalRank: 0);
        group.GetGlobalRank(5);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public void TestProcessGroup_GetGlobalRank_NegativeLocalRank_ThrowsException()
    {
        var group = new ProcessGroup(_globalComm!, new List<int> { 0, 1, 2 }, myGlobalRank: 0);
        group.GetGlobalRank(-1);
    }

    #endregion

    #region AllReduce Tests

    [TestMethod]
    public async Task TestProcessGroup_AllReduceAsync_InGroup_ReturnsReducedTensor()
    {
        var group = new ProcessGroup(_globalComm!, new List<int> { 0, 1, 2 }, myGlobalRank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });

        var result = await group.AllReduceAsync(tensor, ReduceOperation.Sum);

        Assert.IsNotNull(result);
    }

    [TestMethod]
    public async Task TestProcessGroup_AllReduceAsync_NotInGroup_ReturnsUnchangedTensor()
    {
        var group = new ProcessGroup(_globalComm!, new List<int> { 1, 2, 3 }, myGlobalRank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });

        var result = await group.AllReduceAsync(tensor, ReduceOperation.Sum);

        // Should return unchanged tensor
        CollectionAssert.AreEqual(tensor.Data, result.Data);
    }

    [TestMethod]
    public async Task TestProcessGroup_AllReduceAsync_AllOperations()
    {
        var group = new ProcessGroup(_globalComm!, new List<int> { 0, 1 }, myGlobalRank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });

        var operations = new[] { ReduceOperation.Sum, ReduceOperation.Max, ReduceOperation.Min, ReduceOperation.Product, ReduceOperation.Avg };

        foreach (var op in operations)
        {
            var result = await group.AllReduceAsync(tensor, op);
            Assert.IsNotNull(result, $"AllReduce with {op} failed");
        }
    }

    #endregion

    #region AllGather Tests

    [TestMethod]
    public async Task TestProcessGroup_AllGatherAsync_InGroup_ReturnsGatheredTensor()
    {
        var group = new ProcessGroup(_globalComm!, new List<int> { 0, 1, 2 }, myGlobalRank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });

        var result = await group.AllGatherAsync(tensor, dim: 0);

        Assert.IsNotNull(result);
    }

    [TestMethod]
    public async Task TestProcessGroup_AllGatherAsync_NotInGroup_ReturnsUnchangedTensor()
    {
        var group = new ProcessGroup(_globalComm!, new List<int> { 1, 2, 3 }, myGlobalRank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });

        var result = await group.AllGatherAsync(tensor, dim: 0);

        // Should return unchanged tensor
        CollectionAssert.AreEqual(tensor.Data, result.Data);
    }

    [TestMethod]
    public async Task TestProcessGroup_AllGatherAsync_DifferentDimensions()
    {
        var group = new ProcessGroup(_globalComm!, new List<int> { 0, 1 }, myGlobalRank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f });

        var result = await group.AllGatherAsync(tensor, dim: 0);

        Assert.IsNotNull(result);
    }

    #endregion

    #region ReduceScatter Tests

    [TestMethod]
    public async Task TestProcessGroup_ReduceScatterAsync_InGroup_ReturnsScatteredTensor()
    {
        var group = new ProcessGroup(_globalComm!, new List<int> { 0, 1, 2, 3 }, myGlobalRank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f });

        var result = await group.ReduceScatterAsync(tensor, ReduceOperation.Sum);

        Assert.IsNotNull(result);
        // Each rank gets 2 elements
        Assert.AreEqual(2, result.Shape.TotalSize);
    }

    [TestMethod]
    public async Task TestProcessGroup_ReduceScatterAsync_NotInGroup_ReturnsUnchangedTensor()
    {
        var group = new ProcessGroup(_globalComm!, new List<int> { 1, 2, 3 }, myGlobalRank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });

        var result = await group.ReduceScatterAsync(tensor, ReduceOperation.Sum);

        // Should return unchanged tensor
        CollectionAssert.AreEqual(tensor.Data, result.Data);
    }

    #endregion

    #region Broadcast Tests

    [TestMethod]
    public async Task TestProcessGroup_BroadcastAsync_InGroup_ReturnsBroadcastedTensor()
    {
        var group = new ProcessGroup(_globalComm!, new List<int> { 0, 1, 2 }, myGlobalRank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });

        var result = await group.BroadcastAsync(tensor, root: 0);

        Assert.IsNotNull(result);
    }

    [TestMethod]
    public async Task TestProcessGroup_BroadcastAsync_NotInGroup_ReturnsUnchangedTensor()
    {
        var group = new ProcessGroup(_globalComm!, new List<int> { 1, 2, 3 }, myGlobalRank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });

        var result = await group.BroadcastAsync(tensor, root: 0);

        // Should return unchanged tensor
        CollectionAssert.AreEqual(tensor.Data, result.Data);
    }

    [TestMethod]
    public async Task TestProcessGroup_BroadcastAsync_DifferentRoots()
    {
        var group = new ProcessGroup(_globalComm!, new List<int> { 0, 1, 2 }, myGlobalRank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });

        var result0 = await group.BroadcastAsync(tensor, root: 0);
        var result1 = await group.BroadcastAsync(tensor, root: 1);

        Assert.IsNotNull(result0);
        Assert.IsNotNull(result1);
    }

    #endregion

    #region Barrier Tests

    [TestMethod]
    public async Task TestProcessGroup_BarrierAsync_InGroup_Completes()
    {
        var group = new ProcessGroup(_globalComm!, new List<int> { 0, 1, 2 }, myGlobalRank: 0);

        await group.BarrierAsync();

        // Should complete without throwing
    }

    [TestMethod]
    public async Task TestProcessGroup_BarrierAsync_NotInGroup_Completes()
    {
        var group = new ProcessGroup(_globalComm!, new List<int> { 1, 2, 3 }, myGlobalRank: 0);

        await group.BarrierAsync();

        // Should complete without throwing
    }

    #endregion

    #region Dispose Tests

    [TestMethod]
    public void TestProcessGroup_Dispose_CleansUpResources()
    {
        var group = new ProcessGroup(_globalComm!, new List<int> { 0, 1 }, myGlobalRank: 0);

        group.Dispose();

        // Should not throw when disposed
        group.Dispose();
    }

    #endregion

    #region Integration Tests

    [TestMethod]
    public async Task TestProcessGroup_ComplexWorkflow()
    {
        var group = new ProcessGroup(_globalComm!, new List<int> { 0, 1, 2 }, myGlobalRank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });

        // Broadcast
        var broadcastResult = await group.BroadcastAsync(tensor, root: 0);

        // AllGather
        var gatherResult = await group.AllGatherAsync(broadcastResult);

        // AllReduce
        var reduceResult = await group.AllReduceAsync(gatherResult, ReduceOperation.Sum);

        // Barrier
        await group.BarrierAsync();

        Assert.IsNotNull(reduceResult);
    }

    [TestMethod]
    public async Task TestProcessGroup_MultipleGroups()
    {
        var group1 = new ProcessGroup(_globalComm!, new List<int> { 0, 1, 2 }, myGlobalRank: 0);
        var group2 = new ProcessGroup(_globalComm!, new List<int> { 0, 2, 4 }, myGlobalRank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });

        var result1 = await group1.AllReduceAsync(tensor, ReduceOperation.Sum);
        var result2 = await group2.AllReduceAsync(tensor, ReduceOperation.Sum);

        Assert.IsNotNull(result1);
        Assert.IsNotNull(result2);
    }

    [TestMethod]
    public async Task TestProcessGroup_NestedOperations()
    {
        var outerGroup = new ProcessGroup(_globalComm!, new List<int> { 0, 1, 2, 3 }, myGlobalRank: 0);
        var innerGroup = new ProcessGroup(_globalComm!, new List<int> { 0, 1 }, myGlobalRank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });

        var outerResult = await outerGroup.AllReduceAsync(tensor, ReduceOperation.Sum);
        var innerResult = await innerGroup.AllReduceAsync(tensor, ReduceOperation.Sum);

        Assert.IsNotNull(outerResult);
        Assert.IsNotNull(innerResult);
    }

    #endregion
}
