using NUnit.Framework;
using RitterFramework.Core.Tensor;
using MLFramework.HAL;
using MLFramework.HAL.CUDA;
using System.Threading.Tasks;
using System.Threading;

namespace MLFramework.Tests.HAL;

/// <summary>
/// Concurrency tests for the HAL system
/// Tests multi-threaded operations, async scenarios, and thread safety
/// </summary>
[TestFixture]
public class ConcurrencyTests
{
    [SetUp]
    public void Setup()
    {
        // Clear registry before each test to ensure clean state
        BackendRegistry.Clear();
    }

    [Test]
    public void ParallelAllocation_ThreadSafe()
    {
        var device = Device.CPU;
        var allocator = new CachingAllocator(device);

        var tasks = new Task[10];

        for (int i = 0; i < tasks.Length; i++)
        {
            tasks[i] = Task.Run(() =>
            {
                var buffer = allocator.Allocate(1024);
                Thread.Sleep(10);
                allocator.Free(buffer);
            });
        }

        Assert.DoesNotThrow(() => Task.WaitAll(tasks));

        // Clean up
        allocator.EmptyCache();
        allocator.Dispose();
    }

    [Test]
    public void ParallelAllocation_MultipleSizes_ThreadSafe()
    {
        var device = Device.CPU;
        var allocator = new CachingAllocator(device);

        var tasks = new Task[20];
        var random = new Random();

        for (int i = 0; i < tasks.Length; i++)
        {
            int size = (random.Next(1, 10) + 1) * 1024; // 1KB to 10KB
            tasks[i] = Task.Run(() =>
            {
                var buffer = allocator.Allocate(size);
                Thread.Sleep(random.Next(5, 20));
                allocator.Free(buffer);
            });
        }

        Assert.DoesNotThrow(() => Task.WaitAll(tasks));

        allocator.EmptyCache();
        allocator.Dispose();
    }

    [Test]
    public void ParallelBackendOperations_ThreadSafe()
    {
        BackendRegistry.Register(new CpuBackend());

        var tasks = new Task[10];

        for (int i = 0; i < tasks.Length; i++)
        {
            tasks[i] = Task.Run(() =>
            {
                var a = TensorHALExtensions.FromArray(new[] { 1.0f, 2.0f, 3.0f }, Device.CPU);
                var b = TensorHALExtensions.FromArray(new[] { 4.0f, 5.0f, 6.0f }, Device.CPU);
                var backend = BackendRegistry.GetBackend(DeviceType.CPU);

                var result = backend.ExecuteOperation(Operation.Add, new[] { a, b });
                Assert.IsNotNull(result);
            });
        }

        Assert.DoesNotThrow(() => Task.WaitAll(tasks));
    }

    [Test]
    public void ParallelDeviceTransfers_ThreadSafe()
    {
        if (!CudaAvailable())
            Assert.Inconclusive("CUDA not available");

        BackendRegistry.Register(new CpuBackend());
        BackendRegistry.Register(new CudaBackend());

        var tasks = new Task[5];

        for (int i = 0; i < tasks.Length; i++)
        {
            tasks[i] = Task.Run(() =>
            {
                var a = TensorHALExtensions.FromArray(new[] { 1.0f, 2.0f, 3.0f }, Device.CPU);
                var b = a.To(Device.CUDA(0));
                var c = b.To(Device.CPU);

                CollectionAssert.AreEqual(a.ToArray(), c.ToArray());
            });
        }

        Assert.DoesNotThrow(() => Task.WaitAll(tasks));
    }

    [Test]
    public void BackendRegistry_MultipleRegistrations_ThreadSafe()
    {
        var tasks = new Task[5];
        var exceptions = new System.Collections.Concurrent.ConcurrentBag<Exception>();

        for (int i = 0; i < tasks.Length; i++)
        {
            tasks[i] = Task.Run(() =>
            {
                try
                {
                    // Only one should succeed due to duplicate registration check
                    BackendRegistry.Register(new CpuBackend());
                }
                catch (Exception ex)
                {
                    exceptions.Add(ex);
                }
            });
        }

        Task.WaitAll(tasks);

        // We expect 4 failures (duplicate registrations) and 1 success
        Assert.AreEqual(4, exceptions.Count);
        Assert.IsTrue(BackendRegistry.IsDeviceAvailable(DeviceType.CPU));
    }

    [Test]
    public void ParallelCachingAllocator_EmptyCache_ThreadSafe()
    {
        var device = Device.CPU;
        var allocator = new CachingAllocator(device);

        // Allocate and free some buffers
        for (int i = 0; i < 10; i++)
        {
            var buffer = allocator.Allocate(1024);
            allocator.Free(buffer);
        }

        var tasks = new Task[5];

        for (int i = 0; i < tasks.Length; i++)
        {
            tasks[i] = Task.Run(() =>
            {
                allocator.EmptyCache();
            });
        }

        Assert.DoesNotThrow(() => Task.WaitAll(tasks));

        // Cache should be empty
        Assert.AreEqual(0, allocator.CacheSize);

        allocator.Dispose();
    }

    [Test]
    public void CpuBackend_ConcurrentReductionOperations_ThreadSafe()
    {
        BackendRegistry.Register(new CpuBackend());

        var tasks = new Task[20];
        var random = new Random();

        for (int i = 0; i < tasks.Length; i++)
        {
            tasks[i] = Task.Run(() =>
            {
                var size = random.Next(10, 100);
                var data = new float[size];
                for (int j = 0; j < size; j++)
                {
                    data[j] = (float)random.NextDouble() * 100;
                }

                var tensor = TensorHALExtensions.FromArray(data, Device.CPU);
                var backend = BackendRegistry.GetBackend(DeviceType.CPU);

                var sum = backend.ExecuteOperation(Operation.Sum, new[] { tensor });
                var mean = backend.ExecuteOperation(Operation.Mean, new[] { tensor });

                Assert.IsNotNull(sum);
                Assert.IsNotNull(mean);
                Assert.AreEqual(1, sum.Size);
                Assert.AreEqual(1, mean.Size);
            });
        }

        Assert.DoesNotThrow(() => Task.WaitAll(tasks));
    }

    [Test]
    public void MixedCPUAndGpuOperations_ThreadSafe()
    {
        if (!CudaAvailable())
            Assert.Inconclusive("CUDA not available");

        BackendRegistry.Register(new CpuBackend());
        BackendRegistry.Register(new CudaBackend());

        var tasks = new Task[10];
        var random = new Random();

        for (int i = 0; i < tasks.Length; i++)
        {
            int taskId = i;
            tasks[i] = Task.Run(() =>
            {
                var data = new float[] { 1.0f, 2.0f, 3.0f };

                if (taskId % 2 == 0)
                {
                    // CPU operations
                    var tensor = TensorHALExtensions.FromArray(data, Device.CPU);
                    var backend = BackendRegistry.GetBackend(DeviceType.CPU);
                    var result = backend.ExecuteOperation(Operation.ReLU, new[] { tensor });
                    Assert.IsNotNull(result);
                }
                else
                {
                    // GPU operations
                    var tensor = TensorHALExtensions.FromArray(data, Device.CPU);
                    var gpuTensor = tensor.To(Device.CUDA(0));
                    var backend = BackendRegistry.GetBackend(DeviceType.CUDA);
                    var result = backend.ExecuteOperation(Operation.ReLU, new[] { gpuTensor });
                    var cpuResult = result.To(Device.CPU);
                    Assert.IsNotNull(cpuResult);
                }
            });
        }

        Assert.DoesNotThrow(() => Task.WaitAll(tasks));
    }

    [Test]
    public void StressTest_HighAllocationFrequency_NoMemoryLeaks()
    {
        var device = Device.CPU;
        var allocator = new CachingAllocator(device);

        var tasks = new Task[10];

        for (int i = 0; i < tasks.Length; i++)
        {
            tasks[i] = Task.Run(() =>
            {
                for (int j = 0; j < 1000; j++)
                {
                    var buffer = allocator.Allocate(1024);
                    allocator.Free(buffer);
                }
            });
        }

        Assert.DoesNotThrow(() => Task.WaitAll(tasks));

        // Empty cache and verify
        allocator.EmptyCache();

        // After emptying cache, allocated size should be 0
        Assert.AreEqual(0, allocator.AllocatedSize);

        allocator.Dispose();
    }

    [Test]
    public void SimpleAllocator_ParallelOperations_ThreadSafe()
    {
        var device = Device.CPU;
        var allocator = new SimpleAllocator(device);

        var tasks = new Task[10];

        for (int i = 0; i < tasks.Length; i++)
        {
            tasks[i] = Task.Run(() =>
            {
                var buffer = allocator.Allocate(1024);
                Thread.Sleep(10);
                allocator.Free(buffer);
            });
        }

        Assert.DoesNotThrow(() => Task.WaitAll(tasks));

        allocator.Dispose();
    }

    private bool CudaAvailable()
    {
        try
        {
            var result = CudaApi.CudaGetDeviceCount(out int count);
            return result == CudaError.Success && count > 0;
        }
        catch (DllNotFoundException)
        {
            return false;
        }
    }
}
