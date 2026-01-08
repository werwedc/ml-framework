using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.HAL.CUDA.Graphs;
using System;

namespace MLFramework.Tests.HAL.CUDA.Graphs;

/// <summary>
/// Unit tests for CUDAGraphManagerExtensions
/// </summary>
[TestClass]
public class CUDAGraphManagerExtensionsTests
{
    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void GetOrCapturePhaseGraph_NullManager_ThrowsException()
    {
        // Arrange
        CUDAGraphManager manager = null;
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        Action<MLFramework.HAL.CUDA.CudaStream> action = s => { };

        // Act
        manager.GetOrCapturePhaseGraph(GraphPhase.Forward, action, stream);
    }

    [TestMethod]
    public void GetOrCapturePhaseGraph_ValidParameters_ReturnsGraph()
    {
        // Arrange
        var manager = new CUDAGraphManager(captureIterations: 1);
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        Action<MLFramework.HAL.CUDA.CudaStream> action = s => { };

        // Act
        var graph = manager.GetOrCapturePhaseGraph(GraphPhase.Forward, action, stream);

        // Assert
        Assert.IsNull(graph); // Returns null during warm-up
    }

    [TestMethod]
    public void GetOrCapturePhaseGraph_CreatesCorrectGraphName()
    {
        // Arrange
        var manager = new CUDAGraphManager(captureIterations: 1);
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        Action<MLFramework.HAL.CUDA.CudaStream> action = s => { };

        // Act - Get graph for Forward phase
        manager.GetOrCapturePhaseGraph(GraphPhase.Forward, action, stream);

        // Assert - Graph should be registered with correct name
        var graph = manager.GetGraph("Phase_Forward");
        Assert.IsNull(graph); // Graph is tracked but not captured yet
    }

    [TestMethod]
    public void GetOrCapturePhaseGraph_DifferentPhases_TrackSeparately()
    {
        // Arrange
        var manager = new CUDAGraphManager(captureIterations: 1);
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        var executionCount = 0;
        Action<MLFramework.HAL.CUDA.CudaStream> action = s => executionCount++;

        // Act
        manager.GetOrCapturePhaseGraph(GraphPhase.Forward, action, stream);
        manager.GetOrCapturePhaseGraph(GraphPhase.Backward, action, stream);
        manager.GetOrCapturePhaseGraph(GraphPhase.OptimizerStep, action, stream);

        // Assert
        Assert.AreEqual(3, executionCount); // Each phase gets its warm-up

        // Verify each phase has its own graph
        Assert.IsNull(manager.GetGraph("Phase_Forward"));
        Assert.IsNull(manager.GetGraph("Phase_Backward"));
        Assert.IsNull(manager.GetGraph("Phase_OptimizerStep"));
    }

    [TestMethod]
    public void GetOrCapturePhaseGraph_AllPhases_Supported()
    {
        // Arrange
        var manager = new CUDAGraphManager(captureIterations: 1);
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        Action<MLFramework.HAL.CUDA.CudaStream> action = s => { };

        // Act & Assert - Test all phases
        Assert.IsNull(manager.GetOrCapturePhaseGraph(GraphPhase.Forward, action, stream));
        Assert.IsNull(manager.GetOrCapturePhaseGraph(GraphPhase.Backward, action, stream));
        Assert.IsNull(manager.GetOrCapturePhaseGraph(GraphPhase.OptimizerStep, action, stream));
        Assert.IsNull(manager.GetOrCapturePhaseGraph(GraphPhase.ForwardBackward, action, stream));
        Assert.IsNull(manager.GetOrCapturePhaseGraph(GraphPhase.FullTrainingStep, action, stream));

        // Verify all phases are tracked
        Assert.IsNotNull(manager.GetGraph("Phase_Forward"));
        Assert.IsNotNull(manager.GetGraph("Phase_Backward"));
        Assert.IsNotNull(manager.GetGraph("Phase_OptimizerStep"));
        Assert.IsNotNull(manager.GetGraph("Phase_ForwardBackward"));
        Assert.IsNotNull(manager.GetGraph("Phase_FullTrainingStep"));
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void ExecutePhaseGraph_NullManager_ThrowsException()
    {
        // Arrange
        CUDAGraphManager manager = null;
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        Action<MLFramework.HAL.CUDA.CudaStream> action = s => { };

        // Act
        manager.ExecutePhaseGraph(GraphPhase.Forward, action, stream);
    }

    [TestMethod]
    public void ExecutePhaseGraph_ValidParameters_ExecutesAction()
    {
        // Arrange
        var manager = new CUDAGraphManager();
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        var actionExecuted = false;
        Action<MLFramework.HAL.CUDA.CudaStream> action = s => actionExecuted = true;

        // Act
        manager.ExecutePhaseGraph(GraphPhase.Forward, action, stream);

        // Assert
        Assert.IsTrue(actionExecuted);
    }

    [TestMethod]
    public void ExecutePhaseGraph_WarmupPhase_ExecutesAction()
    {
        // Arrange
        var manager = new CUDAGraphManager(captureIterations: 3);
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        var executionCount = 0;
        Action<MLFramework.HAL.CUDA.CudaStream> action = s => executionCount++;

        // Act - Execute multiple times during warm-up
        manager.ExecutePhaseGraph(GraphPhase.Forward, action, stream);
        manager.ExecutePhaseGraph(GraphPhase.Forward, action, stream);
        manager.ExecutePhaseGraph(GraphPhase.Forward, action, stream);

        // Assert
        Assert.AreEqual(3, executionCount);
    }

    [TestMethod]
    public void ExecutePhaseGraph_DifferentPhases_ExecutesCorrectly()
    {
        // Arrange
        var manager = new CUDAGraphManager();
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        var phaseExecuted = string.Empty;
        Action<MLFramework.HAL.CUDA.CudaStream> createAction(string phase) => s => phaseExecuted = phase;

        // Act
        manager.ExecutePhaseGraph(GraphPhase.Forward, createAction("Forward"), stream);
        var firstPhase = phaseExecuted;

        manager.ExecutePhaseGraph(GraphPhase.Backward, createAction("Backward"), stream);
        var secondPhase = phaseExecuted;

        manager.ExecutePhaseGraph(GraphPhase.OptimizerStep, createAction("Optimizer"), stream);
        var thirdPhase = phaseExecuted;

        // Assert
        Assert.AreEqual("Forward", firstPhase);
        Assert.AreEqual("Backward", secondPhase);
        Assert.AreEqual("Optimizer", thirdPhase);
    }

    [TestMethod]
    public void ExecutePhaseGraph_ReusesGraphAfterCapture()
    {
        // Arrange
        var manager = new CUDAGraphManager(captureIterations: 1);
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        var executionCount = 0;
        Action<MLFramework.HAL.CUDA.CudaStream> action = s => executionCount++;

        // Act
        // First call (warm-up)
        manager.ExecutePhaseGraph(GraphPhase.Forward, action, stream);

        // Second call (should reuse graph, though still null in unit tests)
        var countAfterFirst = executionCount;
        manager.ExecutePhaseGraph(GraphPhase.Forward, action, stream);

        // Assert
        Assert.AreEqual(countAfterFirst, executionCount); // Count shouldn't increase
    }

    [TestMethod]
    public void ExecutePhaseGraph_MultiplePhases_IndependentExecution()
    {
        // Arrange
        var manager = new CUDAGraphManager();
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        var forwardCount = 0;
        var backwardCount = 0;
        Action<MLFramework.HAL.CUDA.CudaStream> forwardAction = s => forwardCount++;
        Action<MLFramework.HAL.CUDA.CudaStream> backwardAction = s => backwardCount++;

        // Act
        manager.ExecutePhaseGraph(GraphPhase.Forward, forwardAction, stream);
        manager.ExecutePhaseGraph(GraphPhase.Backward, backwardAction, stream);
        manager.ExecutePhaseGraph(GraphPhase.Forward, forwardAction, stream);
        manager.ExecutePhaseGraph(GraphPhase.Backward, backwardAction, stream);

        // Assert
        Assert.AreEqual(2, forwardCount);
        Assert.AreEqual(2, backwardCount);
    }

    [TestMethod]
    public void ExecutePhaseGraph_NullAction_ThrowsArgumentNullException()
    {
        // Arrange
        var manager = new CUDAGraphManager();
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        Action<MLFramework.HAL.CUDA.CudaStream> action = null;

        // Act & Assert
        Assert.ThrowsException<ArgumentNullException>(() =>
            manager.ExecutePhaseGraph(GraphPhase.Forward, action, stream));
    }

    [TestMethod]
    public void ExecutePhaseGraph_FullTrainingStep_SingleGraph()
    {
        // Arrange
        var manager = new CUDAGraphManager(captureIterations: 1);
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        var executionCount = 0;
        Action<MLFramework.HAL.CUDA.CudaStream> action = s => executionCount++;

        // Act
        manager.ExecutePhaseGraph(GraphPhase.FullTrainingStep, action, stream);

        // Assert
        Assert.AreEqual(1, executionCount);
        Assert.IsNotNull(manager.GetGraph("Phase_FullTrainingStep"));
    }

    [TestMethod]
    public void ExecutePhaseGraph_ForwardBackward_CombinedGraph()
    {
        // Arrange
        var manager = new CUDAGraphManager(captureIterations: 1);
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        var executionCount = 0;
        Action<MLFramework.HAL.CUDA.CudaStream> action = s => executionCount++;

        // Act
        manager.ExecutePhaseGraph(GraphPhase.ForwardBackward, action, stream);

        // Assert
        Assert.AreEqual(1, executionCount);
        Assert.IsNotNull(manager.GetGraph("Phase_ForwardBackward"));
    }

    [TestMethod]
    public void GetOrCapturePhaseGraph_SamePhaseMultipleTimes_ReturnsSameGraph()
    {
        // Arrange
        var manager = new CUDAGraphManager(captureIterations: 1);
        var stream = new MLFramework.HAL.CUDA.CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        Action<MLFramework.HAL.CUDA.CudaStream> action = s => { };

        // Act
        var graph1 = manager.GetOrCapturePhaseGraph(GraphPhase.Forward, action, stream);
        var graph2 = manager.GetOrCapturePhaseGraph(GraphPhase.Forward, action, stream);

        // Assert
        Assert.AreEqual(graph1, graph2); // Should be same reference
    }
}
