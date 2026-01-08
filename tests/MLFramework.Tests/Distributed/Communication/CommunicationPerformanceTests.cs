namespace MLFramework.Tests.Distributed.Communication;

using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;
using System;
using System.Diagnostics;
using System.Threading.Tasks;

[TestClass]
public class CommunicationPerformanceTests
{
    [TestMethod]
    public async Task TestPerformance_Broadcast_SmallTensor()
    {
        var comm = new MockCommunicator(worldSize: 4, rank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f });

        var stopwatch = Stopwatch.StartNew();
        await comm.BroadcastAsync(tensor, 0);
        stopwatch.Stop();

        Assert.IsTrue(stopwatch.ElapsedMilliseconds < 100, $"Broadcast took {stopwatch.ElapsedMilliseconds}ms, expected < 100ms");
        comm.Dispose();
    }

    [TestMethod]
    public async Task TestPerformance_Broadcast_LargeTensor()
    {
        var comm = new MockCommunicator(worldSize: 4, rank: 0);
        var size = 10000;
        var data = new float[size];
        for (int i = 0; i < size; i++)
        {
            data[i] = (float)i;
        }
        var tensor = Tensor.FromArray(data);

        var stopwatch = Stopwatch.StartNew();
        await comm.BroadcastAsync(tensor, 0);
        stopwatch.Stop();

        Assert.IsTrue(stopwatch.ElapsedMilliseconds < 500, $"Broadcast took {stopwatch.ElapsedMilliseconds}ms, expected < 500ms");
        comm.Dispose();
    }

    [TestMethod]
    public async Task TestPerformance_AllReduce_SmallTensor()
    {
        var comm = new MockCommunicator(worldSize: 4, rank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f });

        var stopwatch = Stopwatch.StartNew();
        await comm.AllReduceAsync(tensor, ReduceOperation.Sum);
        stopwatch.Stop();

        Assert.IsTrue(stopwatch.ElapsedMilliseconds < 100, $"AllReduce took {stopwatch.ElapsedMilliseconds}ms, expected < 100ms");
        comm.Dispose();
    }

    [TestMethod]
    public async Task TestPerformance_AllReduce_LargeTensor()
    {
        var comm = new MockCommunicator(worldSize: 4, rank: 0);
        var size = 10000;
        var data = new float[size];
        for (int i = 0; i < size; i++)
        {
            data[i] = (float)i;
        }
        var tensor = Tensor.FromArray(data);

        var stopwatch = Stopwatch.StartNew();
        await comm.AllReduceAsync(tensor, ReduceOperation.Sum);
        stopwatch.Stop();

        Assert.IsTrue(stopwatch.ElapsedMilliseconds < 500, $"AllReduce took {stopwatch.ElapsedMilliseconds}ms, expected < 500ms");
        comm.Dispose();
    }

    [TestMethod]
    public async Task TestPerformance_AllGather_SmallTensor()
    {
        var comm = new MockCommunicator(worldSize: 4, rank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f });

        var stopwatch = Stopwatch.StartNew();
        await comm.AllGatherAsync(tensor);
        stopwatch.Stop();

        Assert.IsTrue(stopwatch.ElapsedMilliseconds < 100, $"AllGather took {stopwatch.ElapsedMilliseconds}ms, expected < 100ms");
        comm.Dispose();
    }

    [TestMethod]
    public async Task TestPerformance_AllGather_LargeTensor()
    {
        var comm = new MockCommunicator(worldSize: 4, rank: 0);
        var size = 10000;
        var data = new float[size];
        for (int i = 0; i < size; i++)
        {
            data[i] = (float)i;
        }
        var tensor = Tensor.FromArray(data);

        var stopwatch = Stopwatch.StartNew();
        await comm.AllGatherAsync(tensor);
        stopwatch.Stop();

        Assert.IsTrue(stopwatch.ElapsedMilliseconds < 500, $"AllGather took {stopwatch.ElapsedMilliseconds}ms, expected < 500ms");
        comm.Dispose();
    }

    [TestMethod]
    public async Task TestPerformance_ReduceScatter_SmallTensor()
    {
        var comm = new MockCommunicator(worldSize: 4, rank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f });

        var stopwatch = Stopwatch.StartNew();
        await comm.ReduceScatterAsync(tensor, ReduceOperation.Sum);
        stopwatch.Stop();

        Assert.IsTrue(stopwatch.ElapsedMilliseconds < 100, $"ReduceScatter took {stopwatch.ElapsedMilliseconds}ms, expected < 100ms");
        comm.Dispose();
    }

    [TestMethod]
    public async Task TestPerformance_ReduceScatter_LargeTensor()
    {
        var comm = new MockCommunicator(worldSize: 4, rank: 0);
        var size = 10000;
        var data = new float[size];
        for (int i = 0; i < size; i++)
        {
            data[i] = (float)i;
        }
        var tensor = Tensor.FromArray(data);

        var stopwatch = Stopwatch.StartNew();
        await comm.ReduceScatterAsync(tensor, ReduceOperation.Sum);
        stopwatch.Stop();

        Assert.IsTrue(stopwatch.ElapsedMilliseconds < 500, $"ReduceScatter took {stopwatch.ElapsedMilliseconds}ms, expected < 500ms");
        comm.Dispose();
    }

    [TestMethod]
    public async Task TestPerformance_Barrier()
    {
        var comm = new MockCommunicator(worldSize: 4, rank: 0);

        var stopwatch = Stopwatch.StartNew();
        await comm.BarrierAsync();
        stopwatch.Stop();

        Assert.IsTrue(stopwatch.ElapsedMilliseconds < 50, $"Barrier took {stopwatch.ElapsedMilliseconds}ms, expected < 50ms");
        comm.Dispose();
    }

    [TestMethod]
    public async Task TestPerformance_SequentialOperations()
    {
        var comm = new MockCommunicator(worldSize: 4, rank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f });

        var stopwatch = Stopwatch.StartNew();

        await comm.BroadcastAsync(tensor, 0);
        await comm.AllReduceAsync(tensor, ReduceOperation.Sum);
        await comm.AllGatherAsync(tensor);
        await comm.ReduceScatterAsync(Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f }), ReduceOperation.Sum);
        await comm.BarrierAsync();

        stopwatch.Stop();

        Assert.IsTrue(stopwatch.ElapsedMilliseconds < 500, $"Sequential operations took {stopwatch.ElapsedMilliseconds}ms, expected < 500ms");
        comm.Dispose();
    }

    [TestMethod]
    public async Task TestPerformance_AllReduce_DifferentOperations()
    {
        var comm = new MockCommunicator(worldSize: 4, rank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f });

        var operations = new[] { ReduceOperation.Sum, ReduceOperation.Product, ReduceOperation.Max, ReduceOperation.Min, ReduceOperation.Avg };

        var stopwatch = Stopwatch.StartNew();

        foreach (var op in operations)
        {
            await comm.AllReduceAsync(tensor, op);
        }

        stopwatch.Stop();

        Assert.IsTrue(stopwatch.ElapsedMilliseconds < 500, $"AllReduce operations took {stopwatch.ElapsedMilliseconds}ms, expected < 500ms");
        comm.Dispose();
    }

    [TestMethod]
    public async Task TestPerformance_ConcurrentOperations()
    {
        var comm = new MockCommunicator(worldSize: 4, rank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f });

        var stopwatch = Stopwatch.StartNew();

        var task1 = comm.AllReduceAsync(tensor, ReduceOperation.Sum);
        var task2 = comm.AllReduceAsync(tensor, ReduceOperation.Max);
        var task3 = comm.AllReduceAsync(tensor, ReduceOperation.Min);

        await Task.WhenAll(task1, task2, task3);

        stopwatch.Stop();

        Assert.IsTrue(stopwatch.ElapsedMilliseconds < 500, $"Concurrent operations took {stopwatch.ElapsedMilliseconds}ms, expected < 500ms");
        comm.Dispose();
    }

    [TestMethod]
    public async Task TestPerformance_MultipleRanks()
    {
        var comm1 = new MockCommunicator(worldSize: 8, rank: 0);
        var comm2 = new MockCommunicator(worldSize: 8, rank: 1);
        var comm3 = new MockCommunicator(worldSize: 8, rank: 2);
        var comm4 = new MockCommunicator(worldSize: 8, rank: 3);

        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f });

        var stopwatch = Stopwatch.StartNew();

        var task1 = comm1.AllReduceAsync(tensor, ReduceOperation.Sum);
        var task2 = comm2.AllReduceAsync(tensor, ReduceOperation.Sum);
        var task3 = comm3.AllReduceAsync(tensor, ReduceOperation.Sum);
        var task4 = comm4.AllReduceAsync(tensor, ReduceOperation.Sum);

        await Task.WhenAll(task1, task2, task3, task4);

        stopwatch.Stop();

        Assert.IsTrue(stopwatch.ElapsedMilliseconds < 500, $"Multiple ranks took {stopwatch.ElapsedMilliseconds}ms, expected < 500ms");

        comm1.Dispose();
        comm2.Dispose();
        comm3.Dispose();
        comm4.Dispose();
    }

    [TestMethod]
    public async Task TestPerformance_StressTest_MultipleOperations()
    {
        var comm = new MockCommunicator(worldSize: 4, rank: 0);
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f });

        var stopwatch = Stopwatch.StartNew();

        // Run 100 operations
        for (int i = 0; i < 100; i++)
        {
            await comm.AllReduceAsync(tensor, ReduceOperation.Sum);
        }

        stopwatch.Stop();

        Assert.IsTrue(stopwatch.ElapsedMilliseconds < 5000, $"100 operations took {stopwatch.ElapsedMilliseconds}ms, expected < 5000ms");
        comm.Dispose();
    }

    [TestMethod]
    public async Task TestPerformance_StressTest_LargeData()
    {
        var comm = new MockCommunicator(worldSize: 4, rank: 0);
        var size = 100000;
        var data = new float[size];
        for (int i = 0; i < size; i++)
        {
            data[i] = (float)i;
        }
        var tensor = Tensor.FromArray(data);

        var stopwatch = Stopwatch.StartNew();

        // Run 10 operations with large data
        for (int i = 0; i < 10; i++)
        {
            await comm.AllReduceAsync(tensor, ReduceOperation.Sum);
        }

        stopwatch.Stop();

        Assert.IsTrue(stopwatch.ElapsedMilliseconds < 5000, $"10 large operations took {stopwatch.ElapsedMilliseconds}ms, expected < 5000ms");
        comm.Dispose();
    }
}
