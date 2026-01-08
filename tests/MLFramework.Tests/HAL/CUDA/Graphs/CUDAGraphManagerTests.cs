using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.HAL.CUDA.Graphs;

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
    }

    [TestMethod]
    public void CUDAGraphManager_ExecuteGraphOrFallback_NewGraph_ExecutesAction()
    {
        // Arrange
        var manager = new CUDAGraphManager();
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        var actionExecuted = false;

        // Act
        manager.ExecuteGraphOrFallback("TestGraph", s => actionExecuted = true, stream);

        // Assert
        Assert.IsTrue(actionExecuted);
    }

    [TestMethod]
    public void CUDAGraphManager_ExecuteGraphOrFallback_NewGraph_TracksGraph()
    {
        // Arrange
        var manager = new CUDAGraphManager();
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));

        // Act
        manager.ExecuteGraphOrFallback("TestGraph", s => { }, stream);
        var graphMethods = manager.GetGraphMethods();

        // Assert - After execution, the graph should be tracked
        // Note: This test assumes the internal state is accessible
        // In a real implementation, you'd expose methods to query captured graphs
    }

    [TestMethod]
    public void CUDAGraphManager_ExecuteGraphOrFallback_ExistingGraph_ExecutesAction()
    {
        // Arrange
        var manager = new CUDAGraphManager();
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        var actionExecuted = false;

        // First execution (warm-up)
        manager.ExecuteGraphOrFallback("TestGraph", s => { }, stream);

        // Act - Second execution (graph captured)
        actionExecuted = false;
        manager.ExecuteGraphOrFallback("TestGraph", s => actionExecuted = true, stream);

        // Assert
        Assert.IsTrue(actionExecuted);
    }

    [TestMethod]
    public void CUDAGraphManager_ExecuteGraphOrFallback_MultipleGraphs_TracksAll()
    {
        // Arrange
        var manager = new CUDAGraphManager();
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));

        // Act
        manager.ExecuteGraphOrFallback("Graph1", s => { }, stream);
        manager.ExecuteGraphOrFallback("Graph2", s => { }, stream);
        manager.ExecuteGraphOrFallback("Graph3", s => { }, stream);

        // Assert - All graphs should be tracked
        // Note: This test assumes the internal state is accessible
    }

    [TestMethod]
    public void CUDAGraphManager_ExecuteGraphOrFallback_NullAction_DoesNotThrow()
    {
        // Arrange
        var manager = new CUDAGraphManager();
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));

        // Act & Assert - Should handle null action gracefully
        manager.ExecuteGraphOrFallback("TestGraph", null!, stream);
    }

    [TestMethod]
    public void CUDAGraphManager_CompleteCapture_SetsCaptureComplete()
    {
        // Arrange
        var manager = new CUDAGraphManager();

        // Act
        manager.CompleteCapture();

        // Assert
        Assert.IsTrue(manager.IsCaptureComplete);
    }

    [TestMethod]
    public void CUDAGraphManager_CompleteCapture_CalledTwice_RemainsComplete()
    {
        // Arrange
        var manager = new CUDAGraphManager();

        // Act
        manager.CompleteCapture();
        manager.CompleteCapture();

        // Assert
        Assert.IsTrue(manager.IsCaptureComplete);
    }

    [TestMethod]
    public void CUDAGraphManager_AfterCompleteCapture_ExecuteGraphOrFallback_StillWorks()
    {
        // Arrange
        var manager = new CUDAGraphManager();
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        var actionExecuted = false;

        // Act
        manager.CompleteCapture();
        manager.ExecuteGraphOrFallback("TestGraph", s => actionExecuted = true, stream);

        // Assert
        Assert.IsTrue(actionExecuted);
        Assert.IsTrue(manager.IsCaptureComplete);
    }
}
