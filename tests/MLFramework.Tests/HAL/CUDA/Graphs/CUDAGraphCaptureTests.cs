using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.HAL.CUDA;

namespace MLFramework.Tests.HAL.CUDA.Graphs;

/// <summary>
/// Unit tests for CUDAGraphCapture
/// </summary>
[TestClass]
public class CUDAGraphCaptureTests
{
    private CudaStream? _stream;
    private CUDAGraphCapture? _capture;

    [TestInitialize]
    public void Setup()
    {
        // Note: Real stream creation requires GPU
        // For unit tests without GPU, we'll create a minimal mock or skip
        // This is a placeholder - actual tests would need GPU mocking
    }

    [TestCleanup]
    public void Cleanup()
    {
        _capture?.Dispose();
        _stream?.Dispose();
    }

    [TestMethod]
    public void CUDAGraphCapture_Constructor_InitializesCorrectly()
    {
        // Act
        var capture = new CUDAGraphCapture();

        // Assert
        Assert.IsNotNull(capture);
        Assert.IsFalse(capture.IsCapturing);
    }

    [TestMethod]
    public void CUDAGraphCapture_IsCapturing_InitiallyFalse()
    {
        // Arrange
        var capture = new CUDAGraphCapture();

        // Act
        var isCapturing = capture.IsCapturing;

        // Assert
        Assert.IsFalse(isCapturing);
    }

    [TestMethod]
    public void CUDAGraphCapture_BeginCapture_SetsIsCapturingToTrue()
    {
        // Arrange
        // Skip this test if GPU is not available
        // In a real test environment, you'd check for GPU availability
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        var device = new CudaDevice(0);
        var stream = new CudaStream(device);
        var capture = new CUDAGraphCapture();

        // Act
        capture.BeginCapture(stream);

        // Assert
        Assert.IsTrue(capture.IsCapturing);

        // Cleanup
        capture.Dispose();
        stream.Dispose();
        device.Dispose();
    }

    [TestMethod]
    [ExpectedException(typeof(ObjectDisposedException))]
    public void CUDAGraphCapture_BeginCapture_AfterDispose_ThrowsException()
    {
        // Arrange
        // Skip this test if GPU is not available
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        var device = new CudaDevice(0);
        var stream = new CudaStream(device);
        var capture = new CUDAGraphCapture();
        capture.Dispose();

        // Act
        capture.BeginCapture(stream);

        // Cleanup
        stream.Dispose();
        device.Dispose();
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void CUDAGraphCapture_BeginCapture_NullStream_ThrowsException()
    {
        // Arrange
        var capture = new CUDAGraphCapture();

        // Act
        capture.BeginCapture(null!);
    }

    [TestMethod]
    [ExpectedException(typeof(InvalidOperationException))]
    public void CUDAGraphCapture_BeginCapture_WhenAlreadyCapturing_ThrowsException()
    {
        // Arrange
        // Skip this test if GPU is not available
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        var device = new CudaDevice(0);
        var stream = new CudaStream(device);
        var capture = new CUDAGraphCapture();
        capture.BeginCapture(stream);

        // Act
        capture.BeginCapture(stream);

        // Cleanup
        capture.Dispose();
        stream.Dispose();
        device.Dispose();
    }

    [TestMethod]
    [ExpectedException(typeof(InvalidOperationException))]
    public void CUDAGraphCapture_EndCapture_WithoutBeginCapture_ThrowsException()
    {
        // Arrange
        var capture = new CUDAGraphCapture();

        // Act
        capture.EndCapture();
    }

    [TestMethod]
    public void CUDAGraphCapture_EndCapture_AfterBeginCapture_ReturnsGraph()
    {
        // Arrange
        // Skip this test if GPU is not available
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        var device = new CudaDevice(0);
        var stream = new CudaStream(device);
        var capture = new CUDAGraphCapture();
        capture.BeginCapture(stream);

        // Act
        var graph = capture.EndCapture();

        // Assert
        Assert.IsNotNull(graph);
        Assert.IsFalse(capture.IsCapturing);

        // Cleanup
        capture.Dispose();
        stream.Dispose();
        device.Dispose();
    }

    [TestMethod]
    public void CUDAGraphCapture_EndCapture_SetsIsCapturingToFalse()
    {
        // Arrange
        // Skip this test if GPU is not available
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        var device = new CudaDevice(0);
        var stream = new CudaStream(device);
        var capture = new CUDAGraphCapture();
        capture.BeginCapture(stream);

        // Act
        capture.EndCapture();

        // Assert
        Assert.IsFalse(capture.IsCapturing);

        // Cleanup
        capture.Dispose();
        stream.Dispose();
        device.Dispose();
    }

    [TestMethod]
    public void CUDAGraphCapture_Dispose_WhileCapturing_AbortsCapture()
    {
        // Arrange
        // Skip this test if GPU is not available
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        var device = new CudaDevice(0);
        var stream = new CudaStream(device);
        var capture = new CUDAGraphCapture();
        capture.BeginCapture(stream);

        // Act
        capture.Dispose();

        // Assert
        Assert.IsFalse(capture.IsCapturing);

        // Cleanup
        stream.Dispose();
        device.Dispose();
    }

    [TestMethod]
    [ExpectedException(typeof(ObjectDisposedException))]
    public void CUDAGraphCapture_Dispose_CalledMultipleTimes_ThrowsException()
    {
        // Arrange
        // Skip this test if GPU is not available
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        var device = new CudaDevice(0);
        var stream = new CudaStream(device);
        var capture = new CUDAGraphCapture();
        capture.BeginCapture(stream);
        capture.Dispose();

        // Act
        capture.BeginCapture(stream);

        // Cleanup
        stream.Dispose();
        device.Dispose();
    }
}
