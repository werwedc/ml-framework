namespace MLFramework.Tests.Distributed.Communication;

using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;
using System;
using System.Threading.Tasks;

[TestClass]
public class CollectiveOperationTests
{
    private MockCommunicator? _comm1;
    private MockCommunicator? _comm2;
    private MockCommunicator? _comm3;
    private MockCommunicator? _comm4;

    [TestInitialize]
    public void Setup()
    {
        _comm1 = new MockCommunicator(worldSize: 4, rank: 0);
        _comm2 = new MockCommunicator(worldSize: 4, rank: 1);
        _comm3 = new MockCommunicator(worldSize: 4, rank: 2);
        _comm4 = new MockCommunicator(worldSize: 4, rank: 3);
    }

    [TestCleanup]
    public void TearDown()
    {
        _comm1?.Dispose();
        _comm2?.Dispose();
        _comm3?.Dispose();
        _comm4?.Dispose();
    }

    #region Broadcast Tests

    [TestMethod]
    public async Task TestBroadcast_Success()
    {
        // Arrange
        var rootTensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f });

        // Act
        var result1 = await _comm1!.BroadcastAsync(rootTensor, 0);
        var result2 = await _comm2!.BroadcastAsync(rootTensor, 0);
        var result3 = await _comm3!.BroadcastAsync(rootTensor, 0);
        var result4 = await _comm4!.BroadcastAsync(rootTensor, 0);

        // Assert
        Assert.IsNotNull(result1);
        Assert.IsNotNull(result2);
        Assert.IsNotNull(result3);
        Assert.IsNotNull(result4);

        CollectionAssert.AreEqual(rootTensor.Data, result1.Data);
        CollectionAssert.AreEqual(rootTensor.Data, result2.Data);
        CollectionAssert.AreEqual(rootTensor.Data, result3.Data);
        CollectionAssert.AreEqual(rootTensor.Data, result4.Data);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public async Task TestBroadcast_InvalidRootRank_ThrowsException()
    {
        var tensor = CreateTestTensor(10);
        await _comm1!.BroadcastAsync(tensor, 10);
    }

    [TestMethod]
    public async Task TestBroadcast_DifferentRoots()
    {
        // Test broadcasting from different ranks
        var tensor1 = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        var tensor2 = Tensor.FromArray(new float[] { 3.0f, 4.0f });
        var tensor3 = Tensor.FromArray(new float[] { 5.0f, 6.0f });
        var tensor4 = Tensor.FromArray(new float[] { 7.0f, 8.0f });

        var result1 = await _comm1!.BroadcastAsync(tensor1, 0);
        var result2 = await _comm2!.BroadcastAsync(tensor2, 1);
        var result3 = await _comm3!.BroadcastAsync(tensor3, 2);
        var result4 = await _comm4!.BroadcastAsync(tensor4, 3);

        // All should get tensor from rank 0
        CollectionAssert.AreEqual(tensor1.Data, result1.Data);
        CollectionAssert.AreEqual(tensor1.Data, result2.Data);
        CollectionAssert.AreEqual(tensor1.Data, result3.Data);
        CollectionAssert.AreEqual(tensor1.Data, result4.Data);
    }

    #endregion

    #region AllReduce Tests

    [TestMethod]
    public async Task TestAllReduce_Sum()
    {
        var tensor1 = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        var tensor2 = Tensor.FromArray(new float[] { 3.0f, 4.0f });
        var tensor3 = Tensor.FromArray(new float[] { 5.0f, 6.0f });
        var tensor4 = Tensor.FromArray(new float[] { 7.0f, 8.0f });

        var result1 = await _comm1!.AllReduceAsync(tensor1, ReduceOperation.Sum);
        var result2 = await _comm2!.AllReduceAsync(tensor2, ReduceOperation.Sum);
        var result3 = await _comm3!.AllReduceAsync(tensor3, ReduceOperation.Sum);
        var result4 = await _comm4!.AllReduceAsync(tensor4, ReduceOperation.Sum);

        // Sum: [1+3+5+7, 2+4+6+8] = [16, 20]
        var expected = new float[] { 16.0f, 20.0f };
        CollectionAssert.AreEqual(expected, result1.Data);
        CollectionAssert.AreEqual(expected, result2.Data);
        CollectionAssert.AreEqual(expected, result3.Data);
        CollectionAssert.AreEqual(expected, result4.Data);
    }

    [TestMethod]
    public async Task TestAllReduce_AllReduceOperations()
    {
        var tensor1 = Tensor.FromArray(new float[] { 1.0f, 5.0f });
        var tensor2 = Tensor.FromArray(new float[] { 3.0f, 2.0f });
        var tensor3 = Tensor.FromArray(new float[] { 2.0f, 8.0f });
        var tensor4 = Tensor.FromArray(new float[] { 4.0f, 1.0f });

        var operations = new[] { ReduceOperation.Sum, ReduceOperation.Product, ReduceOperation.Max, ReduceOperation.Min, ReduceOperation.Avg };

        foreach (var op in operations)
        {
            var result1 = await _comm1!.AllReduceAsync(tensor1, op);
            var result2 = await _comm2!.AllReduceAsync(tensor2, op);
            var result3 = await _comm3!.AllReduceAsync(tensor3, op);
            var result4 = await _comm4!.AllReduceAsync(tensor4, op);

            Assert.IsNotNull(result1, $"AllReduce with {op} failed on rank 1");
            Assert.IsNotNull(result2, $"AllReduce with {op} failed on rank 2");
            Assert.IsNotNull(result3, $"AllReduce with {op} failed on rank 3");
            Assert.IsNotNull(result4, $"AllReduce with {op} failed on rank 4");
        }
    }

    [TestMethod]
    public async Task TestAllReduce_LargeTensor()
    {
        var size = 1000;
        var data1 = new float[size];
        var data2 = new float[size];
        var data3 = new float[size];
        var data4 = new float[size];

        for (int i = 0; i < size; i++)
        {
            data1[i] = 1.0f;
            data2[i] = 2.0f;
            data3[i] = 3.0f;
            data4[i] = 4.0f;
        }

        var tensor1 = Tensor.FromArray(data1);
        var tensor2 = Tensor.FromArray(data2);
        var tensor3 = Tensor.FromArray(data3);
        var tensor4 = Tensor.FromArray(data4);

        var result1 = await _comm1!.AllReduceAsync(tensor1, ReduceOperation.Sum);
        var result2 = await _comm2!.AllReduceAsync(tensor2, ReduceOperation.Sum);

        Assert.AreEqual(size, result1.Shape.TotalSize);
        Assert.AreEqual(size, result2.Shape.TotalSize);

        // All values should be 10 (1+2+3+4)
        for (int i = 0; i < size; i++)
        {
            Assert.AreEqual(10.0f, result1.Data[i], 0.001f);
            Assert.AreEqual(10.0f, result2.Data[i], 0.001f);
        }
    }

    #endregion

    #region AllGather Tests

    [TestMethod]
    public async Task TestAllGather_Success()
    {
        var tensor1 = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        var tensor2 = Tensor.FromArray(new float[] { 3.0f, 4.0f });
        var tensor3 = Tensor.FromArray(new float[] { 5.0f, 6.0f });
        var tensor4 = Tensor.FromArray(new float[] { 7.0f, 8.0f });

        var result1 = await _comm1!.AllGatherAsync(tensor1);
        var result2 = await _comm2!.AllGatherAsync(tensor2);
        var result3 = await _comm3!.AllGatherAsync(tensor3);
        var result4 = await _comm4!.AllGatherAsync(tensor4);

        // Gathered: [1, 2, 3, 4, 5, 6, 7, 8]
        var expected = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
        CollectionAssert.AreEqual(expected, result1.Data);
        CollectionAssert.AreEqual(expected, result2.Data);
        CollectionAssert.AreEqual(expected, result3.Data);
        CollectionAssert.AreEqual(expected, result4.Data);
    }

    [TestMethod]
    public async Task TestAllGather_MultipleDimensions()
    {
        // Create 2D tensors
        var tensor1 = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f });
        var tensor2 = Tensor.FromArray(new float[] { 4.0f, 5.0f, 6.0f });
        var tensor3 = Tensor.FromArray(new float[] { 7.0f, 8.0f, 9.0f });
        var tensor4 = Tensor.FromArray(new float[] { 10.0f, 11.0f, 12.0f });

        var result1 = await _comm1!.AllGatherAsync(tensor1, dim: 0);
        var result2 = await _comm2!.AllGatherAsync(tensor2, dim: 0);

        Assert.IsNotNull(result1);
        Assert.IsNotNull(result2);

        // Verify size is correct (should be 4 * 3 = 12 elements)
        Assert.AreEqual(12, result1.Shape.TotalSize);
        Assert.AreEqual(12, result2.Shape.TotalSize);
    }

    #endregion

    #region ReduceScatter Tests

    [TestMethod]
    public async Task TestReduceScatter_Success()
    {
        var tensor1 = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f });
        var tensor2 = Tensor.FromArray(new float[] { 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f });
        var tensor3 = Tensor.FromArray(new float[] { 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f });
        var tensor4 = Tensor.FromArray(new float[] { 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f });

        var result1 = await _comm1!.ReduceScatterAsync(tensor1, ReduceOperation.Sum);
        var result2 = await _comm2!.ReduceScatterAsync(tensor2, ReduceOperation.Sum);
        var result3 = await _comm3!.ReduceScatterAsync(tensor3, ReduceOperation.Sum);
        var result4 = await _comm4!.ReduceScatterAsync(tensor4, ReduceOperation.Sum);

        // Each rank gets 2 elements
        Assert.AreEqual(2, result1.Shape.TotalSize);
        Assert.AreEqual(2, result2.Shape.TotalSize);
        Assert.AreEqual(2, result3.Shape.TotalSize);
        Assert.AreEqual(2, result4.Shape.TotalSize);
    }

    [TestMethod]
    public async Task TestReduceScatter_DifferentOperations()
    {
        var tensor1 = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        var tensor2 = Tensor.FromArray(new float[] { 5.0f, 6.0f, 7.0f, 8.0f });
        var tensor3 = Tensor.FromArray(new float[] { 9.0f, 10.0f, 11.0f, 12.0f });
        var tensor4 = Tensor.FromArray(new float[] { 13.0f, 14.0f, 15.0f, 16.0f });

        var operations = new[] { ReduceOperation.Sum, ReduceOperation.Max, ReduceOperation.Min };

        foreach (var op in operations)
        {
            var result1 = await _comm1!.ReduceScatterAsync(tensor1, op);
            var result2 = await _comm2!.ReduceScatterAsync(tensor2, op);
            var result3 = await _comm3!.ReduceScatterAsync(tensor3, op);
            var result4 = await _comm4!.ReduceScatterAsync(tensor4, op);

            Assert.IsNotNull(result1, $"ReduceScatter with {op} failed");
            Assert.IsNotNull(result2, $"ReduceScatter with {op} failed");
            Assert.IsNotNull(result3, $"ReduceScatter with {op} failed");
            Assert.IsNotNull(result4, $"ReduceScatter with {op} failed");
        }
    }

    #endregion

    #region Barrier Tests

    [TestMethod]
    public async Task TestBarrier_Success()
    {
        // Act & Assert - should not throw
        await _comm1!.BarrierAsync();
        await _comm2!.BarrierAsync();
        await _comm3!.BarrierAsync();
        await _comm4!.BarrierAsync();
    }

    [TestMethod]
    public async Task TestBarrier_AfterOperations()
    {
        var tensor1 = Tensor.FromArray(new float[] { 1.0f });
        var tensor2 = Tensor.FromArray(new float[] { 2.0f });
        var tensor3 = Tensor.FromArray(new float[] { 3.0f });
        var tensor4 = Tensor.FromArray(new float[] { 4.0f });

        await _comm1!.AllReduceAsync(tensor1, ReduceOperation.Sum);
        await _comm2!.AllReduceAsync(tensor2, ReduceOperation.Sum);
        await _comm3!.AllReduceAsync(tensor3, ReduceOperation.Sum);
        await _comm4!.AllReduceAsync(tensor4, ReduceOperation.Sum);

        // Barrier should complete successfully
        await _comm1!.BarrierAsync();
        await _comm2!.BarrierAsync();
        await _comm3!.BarrierAsync();
        await _comm4!.BarrierAsync();
    }

    #endregion

    #region Integration Tests

    [TestMethod]
    public async Task TestComplexWorkflow_BroadcastAllGatherReduceScatter()
    {
        // Arrange
        var tensor1 = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        var tensor2 = Tensor.FromArray(new float[] { 3.0f, 4.0f });
        var tensor3 = Tensor.FromArray(new float[] { 5.0f, 6.0f });
        var tensor4 = Tensor.FromArray(new float[] { 7.0f, 8.0f });

        // Act: Broadcast from rank 0
        var broadcastResult1 = await _comm1!.BroadcastAsync(tensor1, 0);
        var broadcastResult2 = await _comm2!.BroadcastAsync(tensor2, 0);
        var broadcastResult3 = await _comm3!.BroadcastAsync(tensor3, 0);
        var broadcastResult4 = await _comm4!.BroadcastAsync(tensor4, 0);

        // AllGather
        var gatherResult1 = await _comm1!.AllGatherAsync(broadcastResult1);
        var gatherResult2 = await _comm2!.AllGatherAsync(broadcastResult2);

        // AllReduce
        var reduceResult1 = await _comm1!.AllReduceAsync(gatherResult1, ReduceOperation.Sum);
        var reduceResult2 = await _comm2!.AllReduceAsync(gatherResult2, ReduceOperation.Sum);

        // Assert
        Assert.IsNotNull(reduceResult1);
        Assert.IsNotNull(reduceResult2);
    }

    #endregion

    private Tensor CreateTestTensor(int size)
    {
        var data = new float[size];
        for (int i = 0; i < size; i++)
        {
            data[i] = (float)i;
        }
        return Tensor.FromArray(data);
    }
}
