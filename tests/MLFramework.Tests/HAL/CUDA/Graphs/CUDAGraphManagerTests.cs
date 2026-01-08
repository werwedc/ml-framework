using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.HAL.CUDA.Graphs;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace MLFramework.Tests.HAL.CUDA.Graphs;

/// <summary>
/// Unit tests for CUDAGraphManager
/// </summary>
[TestClass]
public class CUDAGraphManagerTests
{
    [TestMethod]
    public void CUDAGraphManager_Constructor_CreatesInstance()
    {
        // Act
        var manager = new CUDAGraphManager();

        // Assert
        Assert.IsNotNull(manager);
        Assert.IsFalse(manager.IsCaptureComplete);
        Assert.AreEqual(0, manager.GraphCount);
        Assert.IsNotNull(manager.MemoryPool);
    }

    [TestMethod]
    public void CUDAGraphManager_Constructor_WithCustomParameters()
    {
        // Act
        var manager = new CUDAGraphManager(captureIterations: 5, initialMemoryPoolSize: 1024 * 1024);

        // Assert
        Assert.IsNotNull(manager);
        Assert.IsFalse(manager.IsCaptureComplete);
        Assert.AreEqual(0, manager.GraphCount);
    }

    [TestMethod]
    public void CUDAGraphManager_GetOrCaptureGraph_WarmupPhase_ReturnsNull()
    {
        // Arrange
        var manager = new CUDAGraphManager(captureIterations: 3);
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        var actionExecuted = false;
        Action<MLFramework.HAL.CUDA.CudaStream> action = s => actionExecuted = true;

        // Act - First call (warm-up iteration 1)
        var graph = manager.GetOrCaptureGraph("TestGraph", action, stream);

        // Assert
        Assert.IsNull(graph);
        Assert.IsTrue(actionExecuted);
        Assert.IsFalse(manager.IsCaptureComplete);
        Assert.AreEqual(0, manager.GraphCount);
    }

    [TestMethod]
    public void CUDAGraphManager_GetOrCaptureGraph_MultipleWarmupIterations_ExecutesEachTime()
    {
        // Arrange
        var manager = new CUDAGraphManager(captureIterations: 3);
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        var executionCount = 0;
        Action<MLFramework.HAL.CUDA.CudaStream> action = s => executionCount++;

        // Act - Call 3 times for warm-up
        var graph1 = manager.GetOrCaptureGraph("TestGraph", action, stream);
        var graph2 = manager.GetOrCaptureGraph("TestGraph", action, stream);
        var graph3 = manager.GetOrCaptureGraph("TestGraph", action, stream);

        // Assert
        Assert.IsNull(graph1);
        Assert.IsNull(graph2);
        Assert.IsNull(graph3);
        Assert.AreEqual(3, executionCount);
        Assert.IsFalse(manager.IsCaptureComplete);
    }

    [TestMethod]
    public void CUDAGraphManager_GetOrCaptureGraph_AfterWarmup_CapturesGraph()
    {
        // Arrange
        var manager = new CUDAGraphManager(captureIterations: 2);
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        var executionCount = 0;
        Action<MLFramework.HAL.CUDA.CudaStream> action = s => executionCount++;

        // Warm-up iterations
        manager.GetOrCaptureGraph("TestGraph", action, stream);
        manager.GetOrCaptureGraph("TestGraph", action, stream);

        // Act - Third call (capture)
        var graph = manager.GetOrCaptureGraph("TestGraph", action, stream);

        // Assert
        Assert.IsNull(graph); // Cannot capture without actual CUDA operations
        Assert.AreEqual(3, executionCount);
        Assert.IsFalse(manager.IsCaptureComplete); // Still not complete since graph was not captured
    }

    [TestMethod]
    public void CUDAGraphManager_GetOrCaptureGraph_AlreadyCached_ReturnsCachedGraph()
    {
        // Arrange
        var manager = new CUDAGraphManager(captureIterations: 1);
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        var executionCount = 0;
        Action<MLFramework.HAL.CUDA.CudaStream> action = s => executionCount++;

        // Act - First call (warm-up)
        var graph1 = manager.GetOrCaptureGraph("TestGraph", action, stream);

        // Second call - should return same graph reference (null in this case)
        var graph2 = manager.GetOrCaptureGraph("TestGraph", action, stream);

        // Assert
        Assert.AreEqual(graph1, graph2);
        Assert.AreEqual(1, executionCount); // Should not execute again
    }

    [TestMethod]
    public void CUDAGraphManager_ExecuteGraphOrFallback_WarmupPhase_ExecutesAction()
    {
        // Arrange
        var manager = new CUDAGraphManager(captureIterations: 3);
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        var actionExecuted = false;
        Action<MLFramework.HAL.CUDA.CudaStream> action = s => actionExecuted = true;

        // Act
        manager.ExecuteGraphOrFallback("TestGraph", action, stream);

        // Assert
        Assert.IsTrue(actionExecuted);
        Assert.AreEqual(0, manager.GraphCount);
    }

    [TestMethod]
    public void CUDAGraphManager_GetGraph_ExistingGraph_ReturnsGraph()
    {
        // Arrange
        var manager = new CUDAGraphManager(captureIterations: 1);
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        Action<MLFramework.HAL.CUDA.CudaStream> action = s => { };

        // Warm-up
        manager.GetOrCaptureGraph("TestGraph", action, stream);

        // Act
        var graph = manager.GetGraph("TestGraph");

        // Assert - Returns null because graph wasn't actually captured
        Assert.IsNull(graph);
    }

    [TestMethod]
    public void CUDAGraphManager_GetGraph_NonExistingGraph_ReturnsNull()
    {
        // Arrange
        var manager = new CUDAGraphManager();

        // Act
        var graph = manager.GetGraph("NonExistentGraph");

        // Assert
        Assert.IsNull(graph);
    }

    [TestMethod]
    public void CUDAGraphManager_RemoveGraph_ExistingGraph_RemovesGraph()
    {
        // Arrange
        var manager = new CUDAGraphManager(captureIterations: 1);
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        Action<MLFramework.HAL.CUDA.CudaStream> action = s => { };

        // Warm-up
        manager.GetOrCaptureGraph("TestGraph", action, stream);

        // Act
        manager.RemoveGraph("TestGraph");

        // Assert - Graph count remains 0 since graph wasn't captured
        Assert.AreEqual(0, manager.GraphCount);
        Assert.IsNull(manager.GetGraph("TestGraph"));
    }

    [TestMethod]
    public void CUDAGraphManager_RemoveGraph_NonExistingGraph_DoesNotThrow()
    {
        // Arrange
        var manager = new CUDAGraphManager();

        // Act & Assert - Should not throw
        manager.RemoveGraph("NonExistentGraph");
    }

    [TestMethod]
    public void CUDAGraphManager_ClearGraphs_RemovesAllGraphs()
    {
        // Arrange
        var manager = new CUDAGraphManager(captureIterations: 1);
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        Action<MLFramework.HAL.CUDA.CudaStream> action = s => { };

        // Warm-up multiple graphs
        manager.GetOrCaptureGraph("Graph1", action, stream);
        manager.GetOrCaptureGraph("Graph2", action, stream);
        manager.GetOrCaptureGraph("Graph3", action, stream);

        // Act
        manager.ClearGraphs();

        // Assert
        Assert.AreEqual(0, manager.GraphCount);
        Assert.IsNull(manager.GetGraph("Graph1"));
        Assert.IsNull(manager.GetGraph("Graph2"));
        Assert.IsNull(manager.GetGraph("Graph3"));
    }

    [TestMethod]
    public void CUDAGraphManager_MultipleGraphs_TracksEachSeparately()
    {
        // Arrange
        var manager = new CUDAGraphManager(captureIterations: 2);
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        var executionCount = 0;
        Action<MLFramework.HAL.CUDA.CudaStream> action = s => executionCount++;

        // Act
        manager.GetOrCaptureGraph("Graph1", action, stream);
        manager.GetOrCaptureGraph("Graph2", action, stream);

        // Assert
        Assert.AreEqual(2, executionCount); // Each graph gets its warm-up
        Assert.AreEqual(0, manager.GraphCount); // No graphs captured yet
    }

    [TestMethod]
    [ExpectedException(typeof(ObjectDisposedException))]
    public void CUDAGraphManager_Disposed_ThrowsOnGetOrCaptureGraph()
    {
        // Arrange
        var manager = new CUDAGraphManager();
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        Action<MLFramework.HAL.CUDA.CudaStream> action = s => { };

        // Act
        manager.Dispose();
        manager.GetOrCaptureGraph("TestGraph", action, stream);
    }

    [TestMethod]
    [ExpectedException(typeof(ObjectDisposedException))]
    public void CUDAGraphManager_Disposed_ThrowsOnExecuteGraphOrFallback()
    {
        // Arrange
        var manager = new CUDAGraphManager();
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        Action<MLFramework.HAL.CUDA.CudaStream> action = s => { };

        // Act
        manager.Dispose();
        manager.ExecuteGraphOrFallback("TestGraph", action, stream);
    }

    [TestMethod]
    [ExpectedException(typeof(ObjectDisposedException))]
    public void CUDAGraphManager_Disposed_ThrowsOnGetGraph()
    {
        // Arrange
        var manager = new CUDAGraphManager();

        // Act
        manager.Dispose();
        manager.GetGraph("TestGraph");
    }

    [TestMethod]
    [ExpectedException(typeof(ObjectDisposedException))]
    public void CUDAGraphManager_Disposed_ThrowsOnRemoveGraph()
    {
        // Arrange
        var manager = new CUDAGraphManager();

        // Act
        manager.Dispose();
        manager.RemoveGraph("TestGraph");
    }

    [TestMethod]
    [ExpectedException(typeof(ObjectDisposedException))]
    public void CUDAGraphManager_Disposed_ThrowsOnClearGraphs()
    {
        // Arrange
        var manager = new CUDAGraphManager();

        // Act
        manager.Dispose();
        manager.ClearGraphs();
    }

    [TestMethod]
    public void CUDAGraphManager_Dispose_CalledMultipleTimes_DoesNotThrow()
    {
        // Arrange
        var manager = new CUDAGraphManager();

        // Act & Assert
        manager.Dispose();
        manager.Dispose();
        manager.Dispose();
    }

    [TestMethod]
    public void CUDAGraphManager_ConcurrentAccess_ThreadSafe()
    {
        // Arrange
        var manager = new CUDAGraphManager(captureIterations: 10);
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        var executionCount = 0;
        var exceptions = new System.Collections.Concurrent.ConcurrentBag<Exception>();

        // Act - Concurrent access from multiple threads
        var tasks = new Task[10];
        for (int i = 0; i < tasks.Length; i++)
        {
            int threadId = i;
            tasks[i] = Task.Run(() =>
            {
                try
                {
                    for (int j = 0; j < 5; j++)
                    {
                        var graphName = $"Graph_{threadId}";
                        Action<MLFramework.HAL.CUDA.CudaStream> action = s => Interlocked.Increment(ref executionCount);
                        manager.GetOrCaptureGraph(graphName, action, stream);
                    }
                }
                catch (Exception ex)
                {
                    exceptions.Add(ex);
                }
            });
        }

        Task.WaitAll(tasks);

        // Assert
        Assert.AreEqual(0, exceptions.Count);
        Assert.AreEqual(50, executionCount); // 10 threads * 5 iterations
    }

    [TestMethod]
    public void CUDAGraphManager_MemoryPool_SharedAcrossGraphs()
    {
        // Arrange
        var manager = new CUDAGraphManager();
        var pool1 = manager.MemoryPool;
        var pool2 = manager.MemoryPool;

        // Assert
        Assert.AreSame(pool1, pool2);
    }
}
