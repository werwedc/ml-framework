using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.HAL.CUDA;
using MLFramework.HAL.CUDA.Graphs;
using Moq;
using System;

namespace MLFramework.Tests.HAL.CUDA.Graphs;

/// <summary>
/// Unit tests for CUDA Graph fallback handler
/// </summary>
[TestClass]
public class CUDAGraphFallbackHandlerTests
{
    private Mock<ICUDAGraph> _mockGraph;
    private Mock<CudaStream> _mockStream;
    private bool _regularFuncCalled;
    private bool _captureFuncCalled;

    [TestInitialize]
    public void Setup()
    {
        _mockGraph = new Mock<ICUDAGraph>();
        _mockStream = new Mock<CudaStream>();
        _regularFuncCalled = false;
        _captureFuncCalled = false;
    }

    [TestMethod]
    public void Constructor_WithDefaultStrategy_InitializesCorrectly()
    {
        // Act
        var handler = new CUDAGraphFallbackHandler();

        // Assert
        Assert.AreEqual(CUDAGraphFallbackStrategy.CaptureOrFallback, handler.Strategy);
        Assert.AreEqual(0, handler.CaptureAttempts);
        Assert.IsFalse(handler.ShouldUseFallback);
    }

    [TestMethod]
    public void Constructor_WithCustomStrategy_InitializesCorrectly()
    {
        // Act
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.NeverCapture, 5);

        // Assert
        Assert.AreEqual(CUDAGraphFallbackStrategy.NeverCapture, handler.Strategy);
        Assert.IsFalse(handler.ShouldUseFallback);
    }

    [TestMethod]
    public void ShouldUseFallback_WithNeverCapture_ReturnsTrue()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.NeverCapture);

        // Assert
        Assert.IsTrue(handler.ShouldUseFallback);
    }

    [TestMethod]
    public void ShouldUseFallback_WithCaptureOrFallback_ReturnsFalseInitially()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.CaptureOrFallback);

        // Assert
        Assert.IsFalse(handler.ShouldUseFallback);
    }

    [TestMethod]
    public void ExecuteWithFallback_WithNeverCapture_CallsRegularFunc()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.NeverCapture);
        Action<CudaStream> regularFunc = s => _regularFuncCalled = true;
        Func<ICUDAGraph> captureFunc = () =>
        {
            _captureFuncCalled = true;
            return _mockGraph.Object;
        };

        // Act
        handler.ExecuteWithFallback<object>(captureFunc, regularFunc, _mockStream.Object);

        // Assert
        Assert.IsTrue(_regularFuncCalled);
        Assert.IsFalse(_captureFuncCalled);
        Assert.AreEqual(0, handler.CaptureAttempts);
    }

    [TestMethod]
    public void ExecuteWithFallback_WithSuccessfulCapture_CallsCaptureFunc()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.CaptureOrFallback);
        Action<CudaStream> regularFunc = s => _regularFuncCalled = true;
        Func<ICUDAGraph> captureFunc = () =>
        {
            _captureFuncCalled = true;
            return _mockGraph.Object;
        };

        // Act
        handler.ExecuteWithFallback<object>(captureFunc, regularFunc, _mockStream.Object);

        // Assert
        Assert.IsTrue(_captureFuncCalled);
        Assert.IsFalse(_regularFuncCalled);
        Assert.AreEqual(1, handler.CaptureAttempts);
        _mockGraph.Verify(g => g.Execute(_mockStream.Object), Times.Once);
    }

    [TestMethod]
    public void ExecuteWithFallback_WithCaptureOrFallbackAndFailure_CallsRegularFunc()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.CaptureOrFallback);
        Action<CudaStream> regularFunc = s => _regularFuncCalled = true;
        Func<ICUDAGraph> captureFunc = () =>
        {
            _captureFuncCalled = true;
            throw new InvalidOperationException("Capture failed");
        };

        // Act
        handler.ExecuteWithFallback<object>(captureFunc, regularFunc, _mockStream.Object);

        // Assert
        Assert.IsTrue(_captureFuncCalled);
        Assert.IsTrue(_regularFuncCalled);
        Assert.AreEqual(1, handler.CaptureAttempts);
        Assert.IsTrue(handler.ShouldUseFallback);
    }

    [TestMethod]
    public void ExecuteWithFallback_WithCaptureOnlyAndFailure_ThrowsException()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.CaptureOnly);
        Action<CudaStream> regularFunc = s => _regularFuncCalled = true;
        Func<ICUDAGraph> captureFunc = () =>
        {
            throw new InvalidOperationException("Capture failed");
        };

        // Act & Assert
        Assert.ThrowsException<CUDAGraphCaptureException>(() =>
            handler.ExecuteWithFallback<object>(captureFunc, regularFunc, _mockStream.Object));

        Assert.IsFalse(_regularFuncCalled);
    }

    [TestMethod]
    public void ExecuteWithFallback_WithTryOnceThenFallback_PermentlySwitchesAfterFailure()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.TryOnceThenFallback);
        Action<CudaStream> regularFunc = s => _regularFuncCalled = true;
        Func<ICUDAGraph> captureFunc = () =>
        {
            _captureFuncCalled = true;
            throw new InvalidOperationException("Capture failed");
        };

        // Act - first attempt
        handler.ExecuteWithFallback<object>(captureFunc, regularFunc, _mockStream.Object);

        // Assert
        Assert.IsTrue(handler.ShouldUseFallback);
        Assert.IsTrue(_regularFuncCalled);

        // Reset flags
        _regularFuncCalled = false;
        _captureFuncCalled = false;

        // Act - second attempt (should use fallback without calling capture)
        handler.ExecuteWithFallback<object>(captureFunc, regularFunc, _mockStream.Object);

        // Assert
        Assert.IsTrue(_regularFuncCalled);
        Assert.IsFalse(_captureFuncCalled);
        Assert.AreEqual(1, handler.CaptureAttempts); // Only one attempt was made
    }

    [TestMethod]
    public void ExecuteWithFallback_WithRetryThenFallback_RetriesBeforeFallingBack()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.RetryThenFallback, maxRetries: 3);
        Action<CudaStream> regularFunc = s => _regularFuncCalled = true;
        Func<ICUDAGraph> captureFunc = () =>
        {
            _captureFuncCalled = true;
            throw new InvalidOperationException("Capture failed");
        };

        // Act - attempt 1
        handler.ExecuteWithFallback<object>(captureFunc, regularFunc, _mockStream.Object);

        // Assert
        Assert.IsFalse(handler.ShouldUseFallback); // Not yet permanently in fallback
        Assert.AreEqual(1, handler.CaptureAttempts);

        // Act - attempt 2
        handler.ExecuteWithFallback<object>(captureFunc, regularFunc, _mockStream.Object);

        // Assert
        Assert.IsFalse(handler.ShouldUseFallback);
        Assert.AreEqual(2, handler.CaptureAttempts);

        // Act - attempt 3
        handler.ExecuteWithFallback<object>(captureFunc, regularFunc, _mockStream.Object);

        // Assert
        Assert.IsFalse(handler.ShouldUseFallback);
        Assert.AreEqual(3, handler.CaptureAttempts);

        // Act - attempt 4 (should permanently fallback)
        handler.ExecuteWithFallback<object>(captureFunc, regularFunc, _mockStream.Object);

        // Assert
        Assert.IsTrue(handler.ShouldUseFallback); // Now permanently in fallback
        Assert.AreEqual(4, handler.CaptureAttempts);
    }

    [TestMethod]
    public void ExecuteWithFallback_WithRetryThenFallbackAndSuccess_ResetsFallbackFlag()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.RetryThenFallback, maxRetries: 2);
        Action<CudaStream> regularFunc = s => _regularFuncCalled = true;
        Func<ICUDAGraph> captureFunc = () =>
        {
            _captureFuncCalled = true;
            throw new InvalidOperationException("Capture failed");
        };

        // Act - first attempt fails
        handler.ExecuteWithFallback<object>(captureFunc, regularFunc, _mockStream.Object);

        // Assert
        Assert.AreEqual(1, handler.CaptureAttempts);

        // Arrange - now make capture succeed
        _captureFuncCalled = false;
        captureFunc = () =>
        {
            _captureFuncCalled = true;
            return _mockGraph.Object;
        };

        // Act - second attempt succeeds
        handler.ExecuteWithFallback<object>(captureFunc, regularFunc, _mockStream.Object);

        // Assert
        Assert.IsTrue(_captureFuncCalled);
        Assert.IsFalse(_regularFuncCalled);
        Assert.IsFalse(handler.ShouldUseFallback); // Fallback flag reset
        Assert.AreEqual(2, handler.CaptureAttempts);
    }

    [TestMethod]
    public void TryExecuteWithFallback_WithSuccessfulCapture_SetsUsedGraphToTrue()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.CaptureOrFallback);
        Action<CudaStream> regularFunc = s => _regularFuncCalled = true;
        Func<ICUDAGraph> captureFunc = () => _mockGraph.Object;

        // Act
        handler.TryExecuteWithFallback(captureFunc, regularFunc, _mockStream.Object, out bool usedGraph);

        // Assert
        Assert.IsTrue(usedGraph);
        Assert.IsFalse(_regularFuncCalled);
    }

    [TestMethod]
    public void TryExecuteWithFallback_WithFailureAndFallback_SetsUsedGraphToFalse()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.CaptureOrFallback);
        Action<CudaStream> regularFunc = s => _regularFuncCalled = true;
        Func<ICUDAGraph> captureFunc = () => throw new InvalidOperationException("Capture failed");

        // Act
        handler.TryExecuteWithFallback(captureFunc, regularFunc, _mockStream.Object, out bool usedGraph);

        // Assert
        Assert.IsFalse(usedGraph);
        Assert.IsTrue(_regularFuncCalled);
    }

    [TestMethod]
    public void TryExecuteWithFallback_WithNeverCapture_SetsUsedGraphToFalse()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.NeverCapture);
        Action<CudaStream> regularFunc = s => _regularFuncCalled = true;
        Func<ICUDAGraph> captureFunc = () => _mockGraph.Object;

        // Act
        handler.TryExecuteWithFallback(captureFunc, regularFunc, _mockStream.Object, out bool usedGraph);

        // Assert
        Assert.IsFalse(usedGraph);
        Assert.IsTrue(_regularFuncCalled);
    }

    [TestMethod]
    public void Reset_ResetsHandlerState()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.CaptureOrFallback);
        Action<CudaStream> regularFunc = s => _regularFuncCalled = true;
        Func<ICUDAGraph> captureFunc = () => throw new InvalidOperationException("Capture failed");

        // Act - cause failure
        handler.ExecuteWithFallback<object>(captureFunc, regularFunc, _mockStream.Object);
        Assert.IsTrue(handler.ShouldUseFallback);
        Assert.AreEqual(1, handler.CaptureAttempts);

        // Act - reset
        handler.Reset();

        // Assert
        Assert.IsFalse(handler.ShouldUseFallback);
        Assert.AreEqual(0, handler.CaptureAttempts);
    }

    [TestMethod]
    public void Dispose_MarksHandlerAsDisposed()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler();

        // Act
        handler.Dispose();

        // Assert
        Assert.IsTrue(handler.ShouldUseFallback); // Can't check disposed field directly, but no exception should be thrown
    }

    [TestMethod]
    public void ExecuteWithFallback_WithValidationFailure_UsesFallback()
    {
        // Arrange
        var handler = new CUDAGraphFallbackHandler(CUDAGraphFallbackStrategy.CaptureOrFallback);
        Action<CudaStream> regularFunc = s => _regularFuncCalled = true;
        Func<ICUDAGraph> captureFunc = () => throw new CUDAGraphCaptureException("Validation failed");

        // Act
        handler.ExecuteWithFallback<object>(captureFunc, regularFunc, _mockStream.Object);

        // Assert
        Assert.IsTrue(_regularFuncCalled);
        Assert.IsTrue(handler.ShouldUseFallback);
    }
}
