using NUnit.Framework;
using MLFramework.HAL.CUDA;

namespace MLFramework.Tests.HAL.CUDA;

/// <summary>
/// Tests for CUDA Graph Capture functionality
/// Note: These tests require CUDA hardware to be available
/// </summary>
[TestFixture]
public class CUDAGraphCaptureTests
{
    private CudaDevice? _device;
    private CudaStream? _stream;

    [SetUp]
    public void Setup()
    {
        if (!CudaAvailable())
            Assert.Inconclusive("CUDA not available");

        _device = new CudaDevice(0);
        _stream = new CudaStream(_device);
    }

    [TearDown]
    public void TearDown()
    {
        _stream?.Dispose();
        _device?.Dispose();
    }

    [Test]
    public void Constructor_InitializesWithNoCapture()
    {
        var capture = new CUDAGraphCapture();

        Assert.IsFalse(capture.IsCapturing);
    }

    [Test]
    public void BeginCapture_SetsIsCapturingToTrue()
    {
        var capture = new CUDAGraphCapture();

        capture.BeginCapture(_stream!);

        Assert.IsTrue(capture.IsCapturing);
        capture.Dispose();
    }

    [Test]
    public void BeginCapture_WithNullStream_ThrowsArgumentNullException()
    {
        var capture = new CUDAGraphCapture();

        Assert.Throws<ArgumentNullException>(() =>
        {
            capture.BeginCapture(null!);
        });
    }

    [Test]
    public void BeginCapture_WhileAlreadyCapturing_ThrowsInvalidOperationException()
    {
        var capture = new CUDAGraphCapture();

        capture.BeginCapture(_stream!);

        Assert.Throws<InvalidOperationException>(() =>
        {
            capture.BeginCapture(_stream!);
        });

        capture.Dispose();
    }

    [Test]
    public void EndCapture_SetsIsCapturingToFalse()
    {
        var capture = new CUDAGraphCapture();

        capture.BeginCapture(_stream!);
        var graph = capture.EndCapture();

        Assert.IsFalse(capture.IsCapturing);
        Assert.IsNotNull(graph);
        graph.Dispose();
    }

    [Test]
    public void EndCapture_WithoutBeginCapture_ThrowsInvalidOperationException()
    {
        var capture = new CUDAGraphCapture();

        Assert.Throws<InvalidOperationException>(() =>
        {
            capture.EndCapture();
        });
    }

    [Test]
    public void EndCapture_ReturnsValidGraph()
    {
        var capture = new CUDAGraphCapture();

        capture.BeginCapture(_stream!);
        var graph = capture.EndCapture();

        Assert.IsNotNull(graph);
        Assert.IsInstanceOf<ICUDAGraph>(graph);
        Assert.IsNotEmpty(graph.GraphId);
        Assert.AreEqual(CUDAGraphState.Created, graph.State);

        graph.Dispose();
    }

    [Test]
    public void Dispose_WhileCapturing_AbortsCapture()
    {
        var capture = new CUDAGraphCapture();

        capture.BeginCapture(_stream!);
        Assert.IsTrue(capture.IsCapturing);

        capture.Dispose();

        // After disposal, the capture should no longer be capturing
        // Note: We can't check IsCapturing property on disposed object
        // This test mainly verifies no exception is thrown during disposal
    }

    [Test]
    public void BeginCapture_AfterDispose_ThrowsObjectDisposedException()
    {
        var capture = new CUDAGraphCapture();
        capture.Dispose();

        Assert.Throws<ObjectDisposedException>(() =>
        {
            capture.BeginCapture(_stream!);
        });
    }

    [Test]
    public void EndCapture_AfterDispose_ThrowsObjectDisposedException()
    {
        var capture = new CUDAGraphCapture();
        capture.Dispose();

        Assert.Throws<ObjectDisposedException>(() =>
        {
            capture.EndCapture();
        });
    }

    [Test]
    public void MultipleBeginEndCapture_Cycles_WorksCorrectly()
    {
        var capture = new CUDAGraphCapture();

        // First cycle
        capture.BeginCapture(_stream!);
        var graph1 = capture.EndCapture();
        graph1.Dispose();

        // Second cycle
        capture.BeginCapture(_stream!);
        var graph2 = capture.EndCapture();
        graph2.Dispose();

        Assert.IsFalse(capture.IsCapturing);
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
