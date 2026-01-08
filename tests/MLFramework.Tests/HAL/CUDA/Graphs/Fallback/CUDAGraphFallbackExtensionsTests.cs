using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.HAL.CUDA;
using MLFramework.HAL.CUDA.Graphs;
using Moq;
using System;

namespace MLFramework.Tests.HAL.CUDA.Graphs;

/// <summary>
/// Unit tests for CUDA Graph fallback extensions
/// </summary>
[TestClass]
public class CUDAGraphFallbackExtensionsTests
{
    private Mock<CudaStream> _mockStream;
    private Mock<CUDAGraphManager> _mockManager;
    private Mock<ICUDAGraph> _mockGraph;
    private bool _regularFuncCalled;

    [TestInitialize]
    public void Setup()
    {
        _mockStream = new Mock<CudaStream>();
        _mockManager = new Mock<CUDAGraphManager>();
        _mockGraph = new Mock<ICUDAGraph>();
        _regularFuncCalled = false;
    }

    [TestMethod]
    public void WithFallback_WithDefaultParameters_CreatesHandlerWithDefaultStrategy()
    {
        // Act
        var handler = _mockManager.Object.WithFallback();

        // Assert
        Assert.IsNotNull(handler);
        Assert.AreEqual(CUDAGraphFallbackStrategy.CaptureOrFallback, handler.Strategy);
    }

    [TestMethod]
    public void WithFallback_WithCustomStrategy_CreatesHandlerWithCustomStrategy()
    {
        // Act
        var handler = _mockManager.Object.WithFallback(
            CUDAGraphFallbackStrategy.NeverCapture, 5);

        // Assert
        Assert.IsNotNull(handler);
        Assert.AreEqual(CUDAGraphFallbackStrategy.NeverCapture, handler.Strategy);
    }

    [TestMethod]
    public void WithFallback_WithCustomMaxRetries_CreatesHandlerWithCustomMaxRetries()
    {
        // Act
        var handler = _mockManager.Object.WithFallback(
            CUDAGraphFallbackStrategy.RetryThenFallback, 7);

        // Assert
        Assert.IsNotNull(handler);
        Assert.AreEqual(CUDAGraphFallbackStrategy.RetryThenFallback, handler.Strategy);
    }

    [TestMethod]
    public void ExecuteWithFallback_WithSuccessfulCapture_ExecutesGraph()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.CaptureOrFallback);
        string graphName = "test-graph";
        Action<CudaStream> captureAction = s => _regularFuncCalled = true;

        _mockManager.Setup(m => m.GetOrCaptureGraph(
            graphName, captureAction, _mockStream.Object))
            .Returns(_mockGraph.Object);

        // Act
        handler.ExecuteWithFallback(graphName, captureAction, _mockStream.Object, _mockManager.Object);

        // Assert
        _mockGraph.Verify(g => g.Execute(_mockStream.Object), Times.Once);
    }

    [TestMethod]
    public void ExecuteWithFallback_WithFailedCapture_UsesFallback()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.CaptureOrFallback);
        string graphName = "test-graph";
        Action<CudaStream> captureAction = s => _regularFuncCalled = true;

        _mockManager.Setup(m => m.GetOrCaptureGraph(
            graphName, captureAction, _mockStream.Object))
            .Throws(new InvalidOperationException("Capture failed"));

        // Act
        handler.ExecuteWithFallback(graphName, captureAction, _mockStream.Object, _mockManager.Object);

        // Assert
        Assert.IsTrue(_regularFuncCalled);
        _mockGraph.Verify(g => g.Execute(It.IsAny<CudaStream>()), Times.Never);
    }

    [TestMethod]
    public void ExecuteWithFallback_WithGraphName_CallsManagerWithCorrectName()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.CaptureOrFallback);
        string graphName = "my-graph";
        Action<CudaStream> captureAction = s => { };

        _mockManager.Setup(m => m.GetOrCaptureGraph(
            graphName, captureAction, _mockStream.Object))
            .Returns(_mockGraph.Object);

        // Act
        handler.ExecuteWithFallback(graphName, captureAction, _mockStream.Object, _mockManager.Object);

        // Assert
        _mockManager.Verify(m => m.GetOrCaptureGraph(
            graphName, captureAction, _mockStream.Object), Times.Once);
    }

    [TestMethod]
    public void ExecuteWithFallback_WithNeverCaptureStrategy_CallsRegularFunc()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.NeverCapture);
        string graphName = "test-graph";
        Action<CudaStream> captureAction = s => _regularFuncCalled = true;

        // Act
        handler.ExecuteWithFallback(graphName, captureAction, _mockStream.Object, _mockManager.Object);

        // Assert
        Assert.IsTrue(_regularFuncCalled);
        _mockManager.Verify(m => m.GetOrCaptureGraph(
            It.IsAny<string>(), It.IsAny<Action<CudaStream>>(), It.IsAny<CudaStream>()),
            Times.Never);
    }

    [TestMethod]
    public void ExecuteWithFallback_WithCaptureOnlyAndFailure_ThrowsException()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.CaptureOnly);
        string graphName = "test-graph";
        Action<CudaStream> captureAction = s => { };

        _mockManager.Setup(m => m.GetOrCaptureGraph(
            graphName, captureAction, _mockStream.Object))
            .Throws(new InvalidOperationException("Capture failed"));

        // Act & Assert
        Assert.ThrowsException<CUDAGraphCaptureException>(() =>
            handler.ExecuteWithFallback(graphName, captureAction, _mockStream.Object, _mockManager.Object));

        _mockManager.Verify(m => m.GetOrCaptureGraph(
            graphName, captureAction, _mockStream.Object), Times.Once);
    }

    [TestMethod]
    public void ExecuteWithFallback_WithTryOnceThenFallback_PermentlySwitchesAfterFailure()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.TryOnceThenFallback);
        string graphName = "test-graph";
        Action<CudaStream> captureAction = s => _regularFuncCalled = true;

        _mockManager.Setup(m => m.GetOrCaptureGraph(
            graphName, captureAction, _mockStream.Object))
            .Throws(new InvalidOperationException("Capture failed"));

        // Act - first attempt
        handler.ExecuteWithFallback(graphName, captureAction, _mockStream.Object, _mockManager.Object);

        // Assert
        Assert.IsTrue(handler.ShouldUseFallback);
        Assert.IsTrue(_regularFuncCalled);

        // Reset flag
        _regularFuncCalled = false;

        // Act - second attempt
        handler.ExecuteWithFallback(graphName, captureAction, _mockStream.Object, _mockManager.Object);

        // Assert
        Assert.IsTrue(_regularFuncCalled);
        _mockManager.Verify(m => m.GetOrCaptureGraph(
            graphName, captureAction, _mockStream.Object), Times.Once); // Only called once
    }

    [TestMethod]
    public void ExecuteWithFallback_WithRetryThenFallback_RetriesBeforeFallingBack()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.RetryThenFallback, maxRetries: 2);
        string graphName = "test-graph";
        Action<CudaStream> captureAction = s => _regularFuncCalled = true;

        _mockManager.Setup(m => m.GetOrCaptureGraph(
            graphName, captureAction, _mockStream.Object))
            .Throws(new InvalidOperationException("Capture failed"));

        // Act - attempt 1
        handler.ExecuteWithFallback(graphName, captureAction, _mockStream.Object, _mockManager.Object);

        // Assert
        Assert.IsFalse(handler.ShouldUseFallback);
        Assert.AreEqual(1, handler.CaptureAttempts);

        // Act - attempt 2
        handler.ExecuteWithFallback(graphName, captureAction, _mockStream.Object, _mockManager.Object);

        // Assert
        Assert.IsFalse(handler.ShouldUseFallback);
        Assert.AreEqual(2, handler.CaptureAttempts);

        // Act - attempt 3 (should permanently fallback)
        handler.ExecuteWithFallback(graphName, captureAction, _mockStream.Object, _mockManager.Object);

        // Assert
        Assert.IsTrue(handler.ShouldUseFallback);
        Assert.AreEqual(3, handler.CaptureAttempts);

        _mockManager.Verify(m => m.GetOrCaptureGraph(
            graphName, captureAction, _mockStream.Object), Times.Exactly(3));
    }

    [TestMethod]
    public void ExecuteWithFallback_WithNullStream_ThrowsArgumentNullException()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.CaptureOrFallback);
        string graphName = "test-graph";
        Action<CudaStream> captureAction = s => { };

        // Act & Assert
        Assert.ThrowsException<ArgumentNullException>(() =>
            handler.ExecuteWithFallback(graphName, captureAction, null!, _mockManager.Object));
    }

    [TestMethod]
    public void ExecuteWithFallback_WithNullManager_ThrowsArgumentNullException()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.CaptureOrFallback);
        string graphName = "test-graph";
        Action<CudaStream> captureAction = s => { };

        // Act & Assert
        Assert.ThrowsException<ArgumentNullException>(() =>
            handler.ExecuteWithFallback(graphName, captureAction, _mockStream.Object, null!));
    }

    [TestMethod]
    public void ExecuteWithFallback_WithNullCaptureAction_ThrowsArgumentNullException()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.CaptureOrFallback);
        string graphName = "test-graph";
        Action<CudaStream> captureAction = null!;

        // Act & Assert
        Assert.ThrowsException<ArgumentNullException>(() =>
            handler.ExecuteWithFallback(graphName, captureAction, _mockStream.Object, _mockManager.Object));
    }

    [TestMethod]
    public void ExecuteWithFallback_WithNullGraphName_ThrowsArgumentNullException()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.CaptureOrFallback);
        string graphName = null!;
        Action<CudaStream> captureAction = s => { };

        // Act & Assert
        Assert.ThrowsException<ArgumentNullException>(() =>
            handler.ExecuteWithFallback(graphName!, captureAction, _mockStream.Object, _mockManager.Object));
    }
}
