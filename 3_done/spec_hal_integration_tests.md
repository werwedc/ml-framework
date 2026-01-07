# Spec: HAL Integration and Tests

## Overview
Create comprehensive integration tests for the HAL system.

## Responsibilities
- Test device transfers between CPU and GPU
- Test backend registration and device selection
- Test memory management and caching
- Test async operations with streams and events
- Test multi-device scenarios

## Files to Create/Modify
- `tests/HAL/IntegrationTests.cs` - Integration test suite
- `tests/HAL/ConcurrencyTests.cs` - Concurrency and multi-thread tests
- `tests/HAL/MemoryTests.cs` - Memory management tests

## Test Suite Design

### IntegrationTests.cs
```csharp
namespace MLFramework.Tests.HAL;

[TestFixture]
public class HALIntegrationTests
{
    [Test]
    public void FullWorkflow_CpuOnly()
    {
        // Register CPU backend
        BackendRegistry.Register(new CpuBackend());

        // Create tensors on CPU
        var a = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f }, Device.CPU);
        var b = Tensor.FromArray(new[] { 4.0f, 5.0f, 6.0f }, Device.CPU);

        // Perform operations
        var backend = BackendRegistry.GetBackend(DeviceType.CPU);
        var addResult = backend.ExecuteOperation(Operation.Add, new[] { a, b });
        var reluResult = backend.ExecuteOperation(Operation.ReLU, new[] { addResult });

        // Verify results
        var expected = new[] { 5.0f, 7.0f, 9.0f };
        Assert.AreEqual(expected, reluResult.ToArray());
    }

    [Test]
    public void DeviceTransfer_CpuToCpu()
    {
        var a = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f }, Device.CPU);
        var b = a.To(Device.CPU);

        CollectionAssert.AreEqual(a.ToArray(), b.ToArray());
    }

    [Test]
    public void DeviceTransfer_CpuToGpu()
    {
        if (!CudaAvailable())
            Assert.Inconclusive("CUDA not available");

        BackendRegistry.Register(new CpuBackend());
        BackendRegistry.Register(new CudaBackend());

        var a = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f }, Device.CPU);
        var b = a.To(Device.CUDA(0));

        Assert.AreEqual(DeviceType.CUDA, b.Device.DeviceType);

        var c = b.To(Device.CPU);
        CollectionAssert.AreEqual(a.ToArray(), c.ToArray());
    }

    [Test]
    public void BackendRegistry_GetAvailableDevices()
    {
        BackendRegistry.Clear();
        BackendRegistry.Register(new CpuBackend());

        if (CudaAvailable())
        {
            BackendRegistry.Register(new CudaBackend());
        }

        var devices = BackendRegistry.GetAvailableDevices().ToList();

        Assert.Contains(DeviceType.CPU, devices);

        if (CudaAvailable())
        {
            Assert.Contains(DeviceType.CUDA, devices);
        }
    }

    [Test]
    public void CachingAllocator_MemoryReuse()
    {
        var device = new CpuDevice();
        var allocator = new CachingAllocator(device);

        var ptr1 = allocator.Allocate(1024).Pointer;
        allocator.Free(allocator.Allocate(1024));

        var ptr2 = allocator.Allocate(1024).Pointer;

        Assert.AreEqual(ptr1, ptr2);
    }

    private bool CudaAvailable()
    {
        var result = CudaApi.CudaGetDeviceCount(out int count);
        return result == CudaError.Success && count > 0;
    }
}
```

### ConcurrencyTests.cs
```csharp
namespace MLFramework.Tests.HAL;

[TestFixture]
public class ConcurrencyTests
{
    [Test]
    public void MultipleStreams_ParallelExecution()
    {
        if (!CudaAvailable())
            Assert.Inconclusive("CUDA not available");

        var device = Device.CUDA(0);
        var stream1 = device.CreateStream();
        var stream2 = device.CreateStream();

        var a = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f }, Device.CUDA(0));
        var b = Tensor.FromArray(new[] { 4.0f, 5.0f, 6.0f }, Device.CUDA(0));

        // Enqueue operations on both streams
        stream1.Enqueue(() => { /* Launch kernel */ });
        stream2.Enqueue(() => { /* Launch kernel */ });

        // Synchronize both streams
        stream1.Synchronize();
        stream2.Synchronize();

        stream1.Dispose();
        stream2.Dispose();
    }

    [Test]
    public void EventSynchronization_Works()
    {
        if (!CudaAvailable())
            Assert.Inconclusive("CUDA not available");

        var device = Device.CUDA(0);
        var stream1 = device.CreateStream();
        var stream2 = device.CreateStream();

        var evt = stream1.RecordEvent();
        stream2.Wait(evt);

        stream1.Synchronize();
        stream2.Synchronize();

        Assert.IsTrue(evt.IsCompleted);

        stream1.Dispose();
        stream2.Dispose();
        evt.Dispose();
    }

    [Test]
    public void ParallelAllocation_ThreadSafe()
    {
        var device = new CpuDevice();
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

        Task.WaitAll(tasks);
    }

    private bool CudaAvailable()
    {
        var result = CudaApi.CudaGetDeviceCount(out int count);
        return result == CudaError.Success && count > 0;
    }
}
```

### MemoryTests.cs
```csharp
namespace MLFramework.Tests.HAL;

[TestFixture]
public class MemoryTests
{
    [Test]
    public void SimpleAllocator_NoCache()
    {
        var device = new CpuDevice();
        var allocator = new SimpleAllocator(device);

        var buffer1 = allocator.Allocate(1024);
        allocator.Free(buffer1);

        Assert.AreEqual(0, allocator.CacheSize);
    }

    [Test]
    public void CachingAllocator_BlockSplitting()
    {
        var device = new CpuDevice();
        var allocator = new CachingAllocator(device);

        var largeBlock = allocator.Allocate(2048);
        allocator.Free(largeBlock);

        var smallBlock1 = allocator.Allocate(1024);
        var smallBlock2 = allocator.Allocate(1024);

        Assert.AreNotEqual(smallBlock1.Pointer, smallBlock2.Pointer);
    }

    [Test]
    public void CachingAllocator_BlockMerging()
    {
        var device = new CpuDevice();
        var allocator = new CachingAllocator(device);

        var block1 = allocator.Allocate(1024);
        var block2 = allocator.Allocate(1024);

        allocator.Free(block1);
        allocator.Free(block2);

        allocator.Allocate(2048);
    }

    [Test]
    public void EmptyCache_ReleasesMemory()
    {
        var device = new CpuDevice();
        var allocator = new CachingAllocator(device);

        for (int i = 0; i < 10; i++)
        {
            var buffer = allocator.Allocate(1024);
            allocator.Free(buffer);
        }

        var cacheSizeBefore = allocator.CacheSize;
        allocator.EmptyCache();
        var cacheSizeAfter = allocator.CacheSize;

        Assert.Less(cacheSizeAfter, cacheSizeBefore);
    }

    [Test]
    public void NoMemoryLeaks_StressTest()
    {
        var device = new CpuDevice();
        var allocator = new CachingAllocator(device);

        for (int i = 0; i < 10000; i++)
        {
            var buffer = allocator.Allocate(1024);
            allocator.Free(buffer);
        }

        allocator.EmptyCache();

        // Check that allocated size is 0
        Assert.AreEqual(0, allocator.AllocatedSize);
    }
}
```

## Testing Requirements

### Test Categories
1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Cross-component workflows
3. **Concurrency Tests**: Multi-thread and async scenarios
4. **Memory Tests**: Allocator correctness and leak detection
5. **Platform Tests**: CPU-only and GPU-enabled scenarios

### Test Execution
- CPU tests run on all platforms
- GPU tests run only when CUDA is available
- Use Assert.Inconclusive for skipped tests
- Stress tests with large iterations to find memory leaks

## Acceptance Criteria
- [ ] Full workflow test passes (CPU only)
- [ ] Device transfer tests pass
- [ ] Backend registry tests pass
- [ ] Caching allocator tests pass (reuse, splitting, merging)
- [ ] Concurrency tests pass (thread safety)
- [ ] Memory leak tests pass
- [ ] GPU tests pass when CUDA available
- [ ] All tests properly cleanup resources

## Notes for Coder
- These are comprehensive integration tests
- Some tests depend on GPU availability
- Use SetUp/TearDown or TestFixtureSetup/TestFixtureTearDown for cleanup
- Ensure BackendRegistry.Clear() is called between tests
- Memory leak tests should be run in isolation
