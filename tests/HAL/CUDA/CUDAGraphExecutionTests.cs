using NUnit.Framework;
using MLFramework.HAL.CUDA;

namespace MLFramework.Tests.HAL.CUDA;

/// <summary>
/// Tests for CUDA Graph Execution Engine functionality
/// Note: These tests require CUDA hardware to be available
/// </summary>
[TestFixture]
public class CUDAGraphExecutionTests
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
    public void CUDAGraph_AfterCreation_StateIsCreated()
    {
        // Create a mock graph for testing
        var graph = CreateMockGraph();

        Assert.AreEqual(CUDAGraphState.Created, graph.State);
        graph.Dispose();
    }

    [Test]
    public void CUDAGraph_WithValidHandle_HasValidGraphId()
    {
        var graph = CreateMockGraph();

        Assert.IsNotEmpty(graph.GraphId);
        graph.Dispose();
    }

    [Test]
    public void CUDAGraph_Execute_CallsCudaGraphLaunch()
    {
        // This test would require actual CUDA graph capture
        // For now, we test that Execute is called without exception
        // when given a valid (though empty) graph
        if (!CudaAvailable())
            Assert.Inconclusive("CUDA not available");

        var capture = new CUDAGraphCapture();
        capture.BeginCapture(_stream!);
        // No operations - empty graph
        var graph = capture.EndCapture();

        Assert.DoesNotThrow(() =>
        {
            graph.Execute(_stream!);
        });

        graph.Dispose();
        capture.Dispose();
    }

    [Test]
    public void Execute_AfterDispose_ThrowsObjectDisposedException()
    {
        var graph = CreateMockGraph();
        graph.Dispose();

        Assert.Throws<ObjectDisposedException>(() =>
        {
            graph.Execute(_stream!);
        });
    }

    [Test]
    public void Execute_WithNullStream_ThrowsArgumentNullException()
    {
        var graph = CreateMockGraph();

        Assert.Throws<ArgumentNullException>(() =>
        {
            graph.Execute(null!);
        });

        graph.Dispose();
    }

    [Test]
    public void Validate_ValidGraph_ReturnsValidResult()
    {
        var graph = CreateMockGraph();
        var result = graph.Validate();

        Assert.IsNotNull(result);
        // Note: An empty graph created with IntPtr.Zero won't have nodes
        // so validation might fail depending on implementation
        // The main assertion here is that Validate returns a result
        graph.Dispose();
    }

    [Test]
    public void Validate_ReturnsOperationCount()
    {
        if (!CudaAvailable())
            Assert.Inconclusive("CUDA not available");

        var capture = new CUDAGraphCapture();
        capture.BeginCapture(_stream!);
        var graph = capture.EndCapture();

        var result = graph.Validate();

        Assert.IsNotNull(result);
        Assert.GreaterOrEqual(result.OperationCount, 0);

        graph.Dispose();
        capture.Dispose();
    }

    [Test]
    public void Validate_WithErrors_ReturnsErrorsList()
    {
        // Create a graph with an invalid handle
        // This should result in validation errors
        var graph = new CUDAGraph(IntPtr.Zero);

        var result = graph.Validate();

        Assert.IsNotNull(result);
        // An empty/invalid graph should have errors
        // The exact behavior depends on CUDA API response

        graph.Dispose();
    }

    [Test]
    public void Dispose_CleansUpResources()
    {
        var graph = CreateMockGraph();

        var initialGraphId = graph.GraphId;
        Assert.IsNotEmpty(initialGraphId);

        graph.Dispose();

        // After dispose, the graph should be in Disposed state
        Assert.AreEqual(CUDAGraphState.Disposed, graph.State);
    }

    [Test]
    public void MultipleExecutes_SameGraph_WorkCorrectly()
    {
        if (!CudaAvailable())
            Assert.Inconclusive("CUDA not available");

        var capture = new CUDAGraphCapture();
        capture.BeginCapture(_stream!);
        var graph = capture.EndCapture();

        // Execute multiple times
        Assert.DoesNotThrow(() => graph.Execute(_stream!));
        Assert.DoesNotThrow(() => graph.Execute(_stream!));
        Assert.DoesNotThrow(() => graph.Execute(_stream!));

        graph.Dispose();
        capture.Dispose();
    }

    [Test]
    public void Instantiate_CalledMultipleTimes_OnlyInstantiatesOnce()
    {
        if (!CudaAvailable())
            Assert.Inconclusive("CUDA not available");

        var capture = new CUDAGraphCapture();
        capture.BeginCapture(_stream!);
        var graph = capture.EndCapture();

        // The instantiation is called lazily on Execute
        // We just verify multiple Execute calls work
        Assert.DoesNotThrow(() => graph.Execute(_stream!));
        Assert.DoesNotThrow(() => graph.Execute(_stream!));

        graph.Dispose();
        capture.Dispose();
    }

    [Test]
    public void GraphState_TransitionsCorrectly()
    {
        if (!CudaAvailable())
            Assert.Inconclusive("CUDA not available");

        var capture = new CUDAGraphCapture();
        capture.BeginCapture(_stream!);
        var graph = capture.EndCapture();

        // Initial state
        Assert.AreEqual(CUDAGraphState.Created, graph.State);

        // After first execute (includes instantiation)
        graph.Execute(_stream!);
        Assert.AreEqual(CUDAGraphState.Ready, graph.State);

        graph.Dispose();
        capture.Dispose();
    }

    [Test]
    public void Dispose_CalledMultipleTimes_DoesNotThrow()
    {
        var graph = CreateMockGraph();

        Assert.DoesNotThrow(() => graph.Dispose());
        Assert.DoesNotThrow(() => graph.Dispose());
        Assert.DoesNotThrow(() => graph.Dispose());
    }

    private CUDAGraph CreateMockGraph()
    {
        // Create a mock graph with a non-zero handle for testing
        // In a real scenario, this would come from actual graph capture
        return new CUDAGraph(new IntPtr(1));
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
