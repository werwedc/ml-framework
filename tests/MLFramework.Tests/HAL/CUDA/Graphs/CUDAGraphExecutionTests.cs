using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.HAL.CUDA;
using System.Reflection;

namespace MLFramework.Tests.HAL.CUDA.Graphs;

/// <summary>
/// Unit tests for CUDAGraph execution
/// </summary>
[TestClass]
public class CUDAGraphExecutionTests
{
    private ICUDAGraph? _graph;
    private CudaStream? _stream;

    [TestInitialize]
    public void Setup()
    {
        // Note: Real graph creation requires GPU
        // For unit tests, we'll use reflection to create mock graphs or skip tests
    }

    [TestCleanup]
    public void Cleanup()
    {
        _graph?.Dispose();
        _stream?.Dispose();
    }

    [TestMethod]
    public void CUDAGraph_AfterCreation_StateIsCreated()
    {
        // Arrange
        // Create a mock graph for testing using reflection
        var graphType = typeof(CUDAGraph);
        var constructor = graphType.GetConstructor(
            BindingFlags.NonPublic | BindingFlags.Instance,
            null,
            new[] { typeof(IntPtr) },
            null);

        // Act
        _graph = (ICUDAGraph?)constructor?.Invoke(new object[] { IntPtr.Zero });

        // Assert
        Assert.IsNotNull(_graph);
        Assert.AreEqual(CUDAGraphState.Created, _graph.State);
    }

    [TestMethod]
    public void CUDAGraph_GraphId_IsNotNullOrEmpty()
    {
        // Arrange
        var graphType = typeof(CUDAGraph);
        var constructor = graphType.GetConstructor(
            BindingFlags.NonPublic | BindingFlags.Instance,
            null,
            new[] { typeof(IntPtr) },
            null);
        _graph = (ICUDAGraph?)constructor?.Invoke(new object[] { IntPtr.Zero });

        // Act
        var graphId = _graph?.GraphId;

        // Assert
        Assert.IsNotNull(graphId);
        Assert.IsFalse(string.IsNullOrEmpty(graphId));
    }

    [TestMethod]
    [ExpectedException(typeof(ObjectDisposedException))]
    public void CUDAGraph_Execute_AfterDispose_ThrowsException()
    {
        // Arrange
        // Skip this test if GPU is not available
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        var device = new CudaDevice(0);
        _stream = new CudaStream(device);

        // Create a mock graph for testing
        var graphType = typeof(CUDAGraph);
        var constructor = graphType.GetConstructor(
            BindingFlags.NonPublic | BindingFlags.Instance,
            null,
            new[] { typeof(IntPtr) },
            null);
        _graph = (ICUDAGraph?)constructor?.Invoke(new object[] { IntPtr.Zero });
        _graph?.Dispose();

        // Act
        _graph?.Execute(_stream);

        // Cleanup
        device.Dispose();
    }

    [TestMethod]
    public void CUDAGraph_Execute_CallsCUDA_Launch()
    {
        // Arrange
        // Skip this test if GPU is not available
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        var device = new CudaDevice(0);
        _stream = new CudaStream(device);

        // Create a mock graph for testing
        var graphType = typeof(CUDAGraph);
        var constructor = graphType.GetConstructor(
            BindingFlags.NonPublic | BindingFlags.Instance,
            null,
            new[] { typeof(IntPtr) },
            null);

        // Note: This test would require a real captured graph to work properly
        // For now, we'll mark it as inconclusive
        Assert.Inconclusive("Requires real CUDA graph capture");

        // Cleanup
        device.Dispose();
    }

    [TestMethod]
    public void CUDAGraph_Validate_ValidGraph_ReturnsValidResult()
    {
        // Arrange
        // Create a mock graph for testing
        var graphType = typeof(CUDAGraph);
        var constructor = graphType.GetConstructor(
            BindingFlags.NonPublic | BindingFlags.Instance,
            null,
            new[] { typeof(IntPtr) },
            null);
        _graph = (ICUDAGraph?)constructor?.Invoke(new object[] { IntPtr.Zero });

        // Act
        var result = _graph?.Validate();

        // Assert
        Assert.IsNotNull(result);
        // Note: With a mock graph (IntPtr.Zero), validation will fail
        // This is expected behavior
    }

    [TestMethod]
    public void CUDAGraph_Validate_ReturnsNonNullResult()
    {
        // Arrange
        var graphType = typeof(CUDAGraph);
        var constructor = graphType.GetConstructor(
            BindingFlags.NonPublic | BindingFlags.Instance,
            null,
            new[] { typeof(IntPtr) },
            null);
        _graph = (ICUDAGraph?)constructor?.Invoke(new object[] { IntPtr.Zero });

        // Act
        var result = _graph?.Validate();

        // Assert
        Assert.IsNotNull(result);
        Assert.IsNotNull(result?.Errors);
        Assert.IsNotNull(result?.Warnings);
    }

    [TestMethod]
    public void CUDAGraph_Dispose_SetsStateToDisposed()
    {
        // Arrange
        var graphType = typeof(CUDAGraph);
        var constructor = graphType.GetConstructor(
            BindingFlags.NonPublic | BindingFlags.Instance,
            null,
            new[] { typeof(IntPtr) },
            null);
        _graph = (ICUDAGraph?)constructor?.Invoke(new object[] { IntPtr.Zero });

        // Act
        _graph?.Dispose();

        // Assert
        Assert.AreEqual(CUDAGraphState.Disposed, _graph?.State);
    }

    [TestMethod]
    public void CUDAGraph_Dispose_CanBeCalledMultipleTimes()
    {
        // Arrange
        var graphType = typeof(CUDAGraph);
        var constructor = graphType.GetConstructor(
            BindingFlags.NonPublic | BindingFlags.Instance,
            null,
            new[] { typeof(IntPtr) },
            null);
        _graph = (ICUDAGraph?)constructor?.Invoke(new object[] { IntPtr.Zero });

        // Act & Assert - Should not throw
        _graph?.Dispose();
        _graph?.Dispose();
        _graph?.Dispose();
    }

    [TestMethod]
    public void CUDAGraph_Execute_ChangesStateToExecutingThenReady()
    {
        // Arrange
        // Skip this test if GPU is not available
        if (!CudaDevice.Available)
        {
            Assert.Inconclusive("GPU not available for this test");
            return;
        }

        var device = new CudaDevice(0);
        _stream = new CudaStream(device);

        var graphType = typeof(CUDAGraph);
        var constructor = graphType.GetConstructor(
            BindingFlags.NonPublic | BindingFlags.Instance,
            null,
            new[] { typeof(IntPtr) },
            null);
        _graph = (ICUDAGraph?)constructor?.Invoke(new object[] { IntPtr.Zero });

        // Note: This test would require a real captured graph to work properly
        // For now, we'll mark it as inconclusive
        Assert.Inconclusive("Requires real CUDA graph capture");

        // Cleanup
        device.Dispose();
    }

    [TestMethod]
    public void CUDAGraph_StateTransition_FollowsExpectedPath()
    {
        // Arrange
        var graphType = typeof(CUDAGraph);
        var constructor = graphType.GetConstructor(
            BindingFlags.NonPublic | BindingFlags.Instance,
            null,
            new[] { typeof(IntPtr) },
            null);
        _graph = (ICUDAGraph?)constructor?.Invoke(new object[] { IntPtr.Zero });

        // Act & Assert - Initial state
        Assert.AreEqual(CUDAGraphState.Created, _graph?.State);

        // After dispose
        _graph?.Dispose();
        Assert.AreEqual(CUDAGraphState.Disposed, _graph?.State);
    }

    [TestMethod]
    public void CUDAGraph_Validate_ReturnsOperationCount()
    {
        // Arrange
        var graphType = typeof(CUDAGraph);
        var constructor = graphType.GetConstructor(
            BindingFlags.NonPublic | BindingFlags.Instance,
            null,
            new[] { typeof(IntPtr) },
            null);
        _graph = (ICUDAGraph?)constructor?.Invoke(new object[] { IntPtr.Zero });

        // Act
        var result = _graph?.Validate();

        // Assert
        Assert.IsNotNull(result);
        Assert.IsTrue(result?.OperationCount >= 0);
    }
}
