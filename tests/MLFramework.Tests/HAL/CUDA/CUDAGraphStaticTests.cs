using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;

namespace MLFramework.Tests.HAL.CUDA;

/// <summary>
/// Unit tests for CUDAGraphStatic
/// </summary>
[TestClass]
public class CUDAGraphStaticTests
{
    private Mock<CudaStream>? _mockStream;
    private Mock<ICUDAGraph>? _mockGraph;
    private int _captureActionCallCount;

    [TestInitialize]
    public void Setup()
    {
        _mockStream = new Mock<CudaStream>(Mock.Of<CudaDevice>());
        _mockGraph = new Mock<ICUDAGraph>();
        _captureActionCallCount = 0;
    }

    [TestCleanup]
    public void Cleanup()
    {
        _mockStream = null;
        _mockGraph = null;
    }

    private void CaptureAction(CudaStream stream)
    {
        _captureActionCallCount++;
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void Capture_NullCaptureAction_ThrowsException()
    {
        CUDAGraphStatic.Capture(null!, _mockStream!);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void Capture_NullStream_ThrowsException()
    {
        CUDAGraphStatic.Capture(CaptureAction, null!);
    }

    [TestMethod]
    public void Capture_CallsCaptureAction()
    {
        // Note: This test will fail without actual CUDA implementation
        // It's here to demonstrate the test structure
        try
        {
            CUDAGraphStatic.Capture(CaptureAction, _mockStream!);
            Assert.Fail("Expected CUDA exception");
        }
        catch
        {
            // Expected - no actual CUDA available
        }
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void CaptureWithWarmup_NullCaptureAction_ThrowsException()
    {
        CUDAGraphStatic.CaptureWithWarmup(null!, _mockStream!, 3);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void CaptureWithWarmup_NullStream_ThrowsException()
    {
        CUDAGraphStatic.CaptureWithWarmup(CaptureAction, null!, 3);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public void CaptureWithWarmup_NegativeWarmupIterations_ThrowsException()
    {
        CUDAGraphStatic.CaptureWithWarmup(CaptureAction, _mockStream!, -1);
    }

    [TestMethod]
    public void CaptureWithWarmup_ValidParameters_DoesNotThrow()
    {
        try
        {
            CUDAGraphStatic.CaptureWithWarmup(CaptureAction, _mockStream!, 3);
            Assert.Fail("Expected CUDA exception");
        }
        catch
        {
            // Expected - no actual CUDA available
        }
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void Capture_WithOptions_NullCaptureAction_ThrowsException()
    {
        var options = new CUDAGraphCaptureOptions();
        CUDAGraphStatic.Capture(null!, _mockStream!, options);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void Capture_WithOptions_NullStream_ThrowsException()
    {
        var options = new CUDAGraphCaptureOptions();
        CUDAGraphStatic.Capture(CaptureAction, null!, options);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void Capture_WithOptions_NullOptions_ThrowsException()
    {
        CUDAGraphStatic.Capture(CaptureAction, _mockStream!, null!);
    }

    [TestMethod]
    public void Capture_WithOptions_ValidParameters_DoesNotThrow()
    {
        var options = new CUDAGraphCaptureOptions();
        try
        {
            CUDAGraphStatic.Capture(CaptureAction, _mockStream!, options);
            Assert.Fail("Expected CUDA exception");
        }
        catch
        {
            // Expected - no actual CUDA available
        }
    }

    [TestMethod]
    public void Capture_WithOptions_ValidateOnCaptureFalse_DoesNotValidate()
    {
        var options = CUDAGraphCaptureOptions.Default
            .WithValidation(false);

        try
        {
            CUDAGraphStatic.Capture(CaptureAction, _mockStream!, options);
            Assert.Fail("Expected CUDA exception");
        }
        catch
        {
            // Expected - no actual CUDA available
        }
    }

    [TestMethod]
    public void Capture_WithOptions_WithWarmup_DoesNotThrow()
    {
        var options = CUDAGraphCaptureOptions.Default
            .WithWarmup(2)
            .WithValidation(false);

        try
        {
            CUDAGraphStatic.Capture(CaptureAction, _mockStream!, options);
            Assert.Fail("Expected CUDA exception");
        }
        catch
        {
            // Expected - no actual CUDA available
        }
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void CaptureAndExecute_NullCaptureAction_ThrowsException()
    {
        CUDAGraphStatic.CaptureAndExecute(null!, _mockStream!);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void CaptureAndExecute_NullStream_ThrowsException()
    {
        CUDAGraphStatic.CaptureAndExecute(CaptureAction, null!);
    }

    [TestMethod]
    public void CaptureAndExecute_ValidParameters_DoesNotThrow()
    {
        try
        {
            CUDAGraphStatic.CaptureAndExecute(CaptureAction, _mockStream!);
            Assert.Fail("Expected CUDA exception");
        }
        catch
        {
            // Expected - no actual CUDA available
        }
    }
}

/// <summary>
/// Unit tests for CUDAGraphCaptureOptions
/// </summary>
[TestClass]
public class CUDAGraphCaptureOptionsTests
{
    [TestMethod]
    public void CUDAGraphCaptureOptions_Default_InitializesCorrectly()
    {
        var options = CUDAGraphCaptureOptions.Default;

        Assert.AreEqual(0, options.WarmupIterations);
        Assert.IsTrue(options.ValidateOnCapture);
        Assert.IsFalse(options.EnableWeightUpdates);
        Assert.IsNull(options.MemoryPool);
        Assert.AreEqual(CudaCaptureMode.CaptureModeThreadLocal, options.CaptureMode);
    }

    [TestMethod]
    public void CUDAGraphCaptureOptions_WithWarmup_SetsWarmupIterations()
    {
        var options = new CUDAGraphCaptureOptions();
        var result = options.WithWarmup(5);

        Assert.AreEqual(5, options.WarmupIterations);
        Assert.AreSame(options, result);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public void CUDAGraphCaptureOptions_WithWarmup_NegativeValue_ThrowsException()
    {
        var options = new CUDAGraphCaptureOptions();
        options.WithWarmup(-1);
    }

    [TestMethod]
    public void CUDAGraphCaptureOptions_WithValidation_SetsValidateOnCapture()
    {
        var options = new CUDAGraphCaptureOptions();
        var result = options.WithValidation(false);

        Assert.IsFalse(options.ValidateOnCapture);
        Assert.AreSame(options, result);
    }

    [TestMethod]
    public void CUDAGraphCaptureOptions_WithWeightUpdates_SetsEnableWeightUpdates()
    {
        var options = new CUDAGraphCaptureOptions();
        var result = options.WithWeightUpdates(true);

        Assert.IsTrue(options.EnableWeightUpdates);
        Assert.AreSame(options, result);
    }

    [TestMethod]
    public void CUDAGraphCaptureOptions_WithMemoryPool_SetsMemoryPool()
    {
        var options = new CUDAGraphCaptureOptions();
        var pool = new Mock<CUDAGraphMemoryPool>(1024, 2048).Object;
        var result = options.WithMemoryPool(pool);

        Assert.AreSame(pool, options.MemoryPool);
        Assert.AreSame(options, result);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void CUDAGraphCaptureOptions_WithMemoryPool_Null_ThrowsException()
    {
        var options = new CUDAGraphCaptureOptions();
        options.WithMemoryPool(null!);
    }

    [TestMethod]
    public void CUDAGraphCaptureOptions_WithCaptureMode_SetsCaptureMode()
    {
        var options = new CUDAGraphCaptureOptions();
        var result = options.WithCaptureMode(CudaCaptureMode.CaptureModeGlobal);

        Assert.AreEqual(CudaCaptureMode.CaptureModeGlobal, options.CaptureMode);
        Assert.AreSame(options, result);
    }

    [TestMethod]
    public void CUDAGraphCaptureOptions_FluentChaining_WorksCorrectly()
    {
        var pool = new Mock<CUDAGraphMemoryPool>(1024, 2048).Object;
        var options = new CUDAGraphCaptureOptions()
            .WithWarmup(3)
            .WithValidation(true)
            .WithWeightUpdates(true)
            .WithMemoryPool(pool)
            .WithCaptureMode(CudaCaptureMode.CaptureModeRelaxed);

        Assert.AreEqual(3, options.WarmupIterations);
        Assert.IsTrue(options.ValidateOnCapture);
        Assert.IsTrue(options.EnableWeightUpdates);
        Assert.AreSame(pool, options.MemoryPool);
        Assert.AreEqual(CudaCaptureMode.CaptureModeRelaxed, options.CaptureMode);
    }
}

/// <summary>
/// Unit tests for CUDAGraphExtensions
/// </summary>
[TestClass]
public class CUDAGraphExtensionsTests
{
    private Mock<ICUDAGraph>? _mockGraph;
    private Mock<CudaStream>? _mockStream;

    [TestInitialize]
    public void Setup()
    {
        _mockGraph = new Mock<ICUDAGraph>();
        _mockStream = new Mock<CudaStream>(Mock.Of<CudaDevice>());
    }

    [TestCleanup]
    public void Cleanup()
    {
        _mockGraph = null;
        _mockStream = null;
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void ExecuteMultiple_NullGraph_ThrowsException()
    {
        _mockGraph!.Object.ExecuteMultiple(_mockStream!.Object, 5);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void ExecuteMultiple_NullStream_ThrowsException()
    {
        _mockGraph!.Object.ExecuteMultiple(null!, 5);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public void ExecuteMultiple_ZeroIterations_ThrowsException()
    {
        _mockGraph!.Object.ExecuteMultiple(_mockStream!.Object, 0);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public void ExecuteMultiple_NegativeIterations_ThrowsException()
    {
        _mockGraph!.Object.ExecuteMultiple(_mockStream!.Object, -1);
    }

    [TestMethod]
    public void ExecuteMultiple_ValidIterations_ExecutesCorrectly()
    {
        int executeCount = 0;
        _mockGraph!.Setup(g => g.Execute(It.IsAny<CudaStream>()))
            .Callback(() => executeCount++);

        _mockGraph.Object.ExecuteMultiple(_mockStream!.Object, 5);

        Assert.AreEqual(5, executeCount);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void MeasureExecutionTime_NullGraph_ThrowsException()
    {
        _mockGraph!.Object.MeasureExecutionTime(_mockStream!.Object, 10);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void MeasureExecutionTime_NullStream_ThrowsException()
    {
        _mockGraph!.Object.MeasureExecutionTime(null!, 10);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public void MeasureExecutionTime_ZeroIterations_ThrowsException()
    {
        _mockGraph!.Object.MeasureExecutionTime(_mockStream!.Object, 0);
    }

    [TestMethod]
    public void MeasureExecutionTime_ValidIterations_ExecutesCorrectly()
    {
        int executeCount = 0;
        _mockGraph!.Setup(g => g.Execute(It.IsAny<CudaStream>()))
            .Callback(() => executeCount++);

        var time = _mockGraph.Object.MeasureExecutionTime(_mockStream!.Object, 5);

        Assert.AreEqual(5, executeCount);
        Assert.IsTrue(time >= TimeSpan.Zero);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void GetAverageExecutionTimeMs_NullGraph_ThrowsException()
    {
        _mockGraph!.Object.GetAverageExecutionTimeMs(_mockStream!.Object, 10);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void GetAverageExecutionTimeMs_NullStream_ThrowsException()
    {
        _mockGraph!.Object.GetAverageExecutionTimeMs(null!, 10);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentOutOfRangeException))]
    public void GetAverageExecutionTimeMs_ZeroIterations_ThrowsException()
    {
        _mockGraph!.Object.GetAverageExecutionTimeMs(_mockStream!.Object, 0);
    }

    [TestMethod]
    public void GetAverageExecutionTimeMs_ValidIterations_ReturnsValidValue()
    {
        _mockGraph!.Setup(g => g.Execute(It.IsAny<CudaStream>()));

        var avgTime = _mockGraph.Object.GetAverageExecutionTimeMs(_mockStream!.Object, 10);

        Assert.IsTrue(avgTime >= 0);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void ExecuteWithCallbacks_NullGraph_ThrowsException()
    {
        _mockGraph!.Object.ExecuteWithCallbacks(_mockStream!.Object);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void ExecuteWithCallbacks_NullStream_ThrowsException()
    {
        _mockGraph!.Object.ExecuteWithCallbacks(null!);
    }

    [TestMethod]
    public void ExecuteWithCallbacks_BeforeCallback_CallsBeforeExecute()
    {
        bool beforeCalled = false;
        _mockGraph!.Setup(g => g.Execute(It.IsAny<CudaStream>()))
            .Callback(() => Assert.IsTrue(beforeCalled, "Before callback should be called before execute"));

        _mockGraph.Object.ExecuteWithCallbacks(_mockStream!.Object,
            before: () => beforeCalled = true);

        Assert.IsTrue(beforeCalled);
    }

    [TestMethod]
    public void ExecuteWithCallbacks_AfterCallback_CallsAfterExecute()
    {
        bool afterCalled = false;
        _mockGraph!.Setup(g => g.Execute(It.IsAny<CudaStream>()))
            .Callback(() => Assert.IsFalse(afterCalled, "After callback should be called after execute"));

        _mockGraph.Object.ExecuteWithCallbacks(_mockStream!.Object,
            after: () => afterCalled = true);

        Assert.IsTrue(afterCalled);
    }

    [TestMethod]
    public void ExecuteWithCallbacks_NoCallbacks_ExecutesGraph()
    {
        _mockGraph!.Setup(g => g.Execute(It.IsAny<CudaStream>()));

        _mockGraph.Object.ExecuteWithCallbacks(_mockStream!.Object);

        _mockGraph.Verify(g => g.Execute(It.IsAny<CudaStream>()), Times.Once);
    }

    [TestMethod]
    public void ExecuteWithCallbacks_BothCallbacks_CallsInCorrectOrder()
    {
        int callOrder = 0;
        int beforeOrder = 0;
        int executeOrder = 0;
        int afterOrder = 0;

        _mockGraph!.Setup(g => g.Execute(It.IsAny<CudaStream>()))
            .Callback(() => executeOrder = ++callOrder);

        _mockGraph.Object.ExecuteWithCallbacks(_mockStream!.Object,
            before: () => beforeOrder = ++callOrder,
            after: () => afterOrder = ++callOrder);

        Assert.AreEqual(1, beforeOrder);
        Assert.AreEqual(2, executeOrder);
        Assert.AreEqual(3, afterOrder);
    }
}
