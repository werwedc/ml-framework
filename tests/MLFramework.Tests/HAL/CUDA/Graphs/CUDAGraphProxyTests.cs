using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.HAL.CUDA.Graphs;
using MLFramework.HAL.CUDA.Graphs.Attributes;
using MLFramework.HAL.CUDA.Graphs.Proxies;
using System.Reflection;

namespace MLFramework.Tests.HAL.CUDA.Graphs;

/// <summary>
/// Test model with CaptureGraph attributes
/// </summary>
public class TestModel
{
    public int CallCount { get; private set; }

    [CaptureGraph("ForwardPass")]
    public string Forward(string input)
    {
        CallCount++;
        return $"Processed: {input}";
    }

    [CaptureGraph("BackwardPass")]
    public int Backward(int gradient)
    {
        CallCount++;
        return gradient * 2;
    }

    [CaptureGraph]
    public void TrainingStep()
    {
        CallCount++;
    }

    public void NonGraphMethod()
    {
        CallCount++;
    }
}

/// <summary>
/// Unit tests for CUDAGraphProxy
/// </summary>
[TestClass]
public class CUDAGraphProxyTests
{
    private CudaStream? _stream;
    private TestModel? _model;

    [TestInitialize]
    public void Setup()
    {
        _stream = new CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
        _model = new TestModel();
    }

    [TestCleanup]
    public void Cleanup()
    {
        _stream?.Dispose();
    }

    [TestMethod]
    public void CUDAGraphProxy_Constructor_NullTarget_ThrowsException()
    {
        // Act & Assert
        Assert.ThrowsException<ArgumentNullException>(() =>
            new CUDAGraphProxy<TestModel>(null!, _stream!));
    }

    [TestMethod]
    public void CUDAGraphProxy_Constructor_NullStream_ThrowsException()
    {
        // Act & Assert
        Assert.ThrowsException<ArgumentNullException>(() =>
            new CUDAGraphProxy<TestModel>(_model!, null!));
    }

    [TestMethod]
    public void CUDAGraphProxy_Constructor_ValidParameters_CreatesProxy()
    {
        // Arrange & Act
        var proxy = new CUDAGraphProxy<TestModel>(_model!, _stream!);

        // Assert
        Assert.IsNotNull(proxy);
        Assert.IsNotNull(proxy.GraphManager);
    }

    [TestMethod]
    public void CUDAGraphProxy_Constructor_CustomManager_UsesProvidedManager()
    {
        // Arrange
        var customManager = new CUDAGraphManager();

        // Act
        var proxy = new CUDAGraphProxy<TestModel>(_model!, _stream!, customManager);

        // Assert
        Assert.AreSame(customManager, proxy.GraphManager);
    }

    [TestMethod]
    public void CUDAGraphProxy_Create_ValidParameters_ReturnsInstance()
    {
        // Act
        var result = CUDAGraphProxy<TestModel>.Create(_model!, _stream!);

        // Assert
        Assert.IsNotNull(result);
    }

    [TestMethod]
    public void CUDAGraphProxy_DiscoverGraphMethods_FindsAllMarkedMethods()
    {
        // Arrange
        var proxy = new CUDAGraphProxy<TestModel>(_model!, _stream!);

        // Act
        var graphMethods = proxy.GetGraphMethods();

        // Assert
        Assert.AreEqual(3, graphMethods.Count);
        Assert.IsTrue(graphMethods.ContainsKey("ForwardPass"));
        Assert.IsTrue(graphMethods.ContainsKey("BackwardPass"));
        Assert.IsTrue(graphMethods.ContainsKey($"{nameof(TestModel)}.TrainingStep"));
    }

    [TestMethod]
    public void CUDAGraphProxy_GetGraphAttributes_ReturnsCorrectAttributes()
    {
        // Arrange
        var proxy = new CUDAGraphProxy<TestModel>(_model!, _stream!);

        // Act
        var attributes = proxy.GetGraphAttributes();

        // Assert
        Assert.AreEqual(3, attributes.Count);
        Assert.IsTrue(attributes.ContainsKey("ForwardPass"));
        Assert.IsTrue(attributes.ContainsKey("BackwardPass"));
        Assert.AreEqual(3, attributes["ForwardPass"].WarmupIterations);
    }

    [TestMethod]
    public void CUDAGraphProxy_ExecuteWithGraph_ValidGraph_ExecutesMethod()
    {
        // Arrange
        var proxy = new CUDAGraphProxy<TestModel>(_model!, _stream!);

        // Act
        var result = proxy.ExecuteWithGraph<string>("ForwardPass", "test");

        // Assert
        Assert.IsNotNull(result);
        Assert.AreEqual("Processed: test", result);
        Assert.AreEqual(1, _model!.CallCount);
    }

    [TestMethod]
    public void CUDAGraphProxy_ExecuteWithGraph_UnknownGraph_ThrowsException()
    {
        // Arrange
        var proxy = new CUDAGraphProxy<TestModel>(_model!, _stream!);

        // Act & Assert
        Assert.ThrowsException<KeyNotFoundException>(() =>
            proxy.ExecuteWithGraph<string>("UnknownGraph", "test"));
    }

    [TestMethod]
    public void CUDAGraphProxy_ExecuteWithGraph_VoidReturn_ExecutesMethod()
    {
        // Arrange
        var proxy = new CUDAGraphProxy<TestModel>(_model!, _stream!);

        // Act
        proxy.ExecuteWithGraph($"{nameof(TestModel)}.TrainingStep");

        // Assert
        Assert.AreEqual(1, _model!.CallCount);
    }

    [TestMethod]
    public void CUDAGraphProxy_ExecuteWithGraph_MultipleExecutions_IncrementsCallCount()
    {
        // Arrange
        var proxy = new CUDAGraphProxy<TestModel>(_model!, _stream!);

        // Act
        proxy.ExecuteWithGraph<string>("ForwardPass", "test1");
        proxy.ExecuteWithGraph<string>("ForwardPass", "test2");
        proxy.ExecuteWithGraph<string>("ForwardPass", "test3");

        // Assert
        Assert.AreEqual(3, _model!.CallCount);
    }

    [TestMethod]
    public void CUDAGraphProxy_ExecuteWithGraph_DifferentGraphs_ExecutesCorrectMethods()
    {
        // Arrange
        var proxy = new CUDAGraphProxy<TestModel>(_model!, _stream!);

        // Act
        var forwardResult = proxy.ExecuteWithGraph<string>("ForwardPass", "input");
        var backwardResult = proxy.ExecuteWithGraph<int>("BackwardPass", 10);

        // Assert
        Assert.AreEqual("Processed: input", forwardResult);
        Assert.AreEqual(20, backwardResult);
        Assert.AreEqual(2, _model!.CallCount);
    }

    [TestMethod]
    public void CUDAGraphProxy_GetGraphMethods_DoesNotIncludeNonGraphMethods()
    {
        // Arrange
        var proxy = new CUDAGraphProxy<TestModel>(_model!, _stream!);

        // Act
        var graphMethods = proxy.GetGraphMethods();

        // Assert
        foreach (var method in graphMethods.Values)
        {
            Assert.AreNotEqual("NonGraphMethod", method.Name);
        }
    }
}
