using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.HAL.CUDA.Graphs;
using MLFramework.HAL.CUDA.Graphs.Attributes;
using MLFramework.HAL.CUDA.Graphs.Proxies;

namespace MLFramework.Tests.HAL.CUDA.Graphs;

/// <summary>
/// Test model for extension methods
/// </summary>
public class ExtensionTestModel
{
    [CaptureGraph("TestGraph")]
    public string TestMethod(string input)
    {
        return $"Result: {input}";
    }
}

/// <summary>
/// Unit tests for CUDAGraphProxyExtensions
/// </summary>
[TestClass]
public class CUDAGraphProxyExtensionsTests
{
    private CudaStream? _stream;

    [TestInitialize]
    public void Setup()
    {
        _stream = new CudaStream(new MLFramework.HAL.CUDA.CudaDevice(0));
    }

    [TestCleanup]
    public void Cleanup()
    {
        _stream?.Dispose();
    }

    [TestMethod]
    public void WithGraphCapture_NullInstance_ThrowsException()
    {
        // Act & Assert
        Assert.ThrowsException<ArgumentNullException>(() =>
            ((ExtensionTestModel)null!).WithGraphCapture(_stream!));
    }

    [TestMethod]
    public void WithGraphCapture_NullStream_ThrowsException()
    {
        // Arrange
        var model = new ExtensionTestModel();

        // Act & Assert
        Assert.ThrowsException<ArgumentNullException>(() =>
            model.WithGraphCapture((CudaStream)null!));
    }

    [TestMethod]
    public void WithGraphCapture_ValidParameters_ReturnsInstance()
    {
        // Arrange
        var model = new ExtensionTestModel();

        // Act
        var result = model.WithGraphCapture(_stream!);

        // Assert
        Assert.IsNotNull(result);
        Assert.IsInstanceOfType(result, typeof(ExtensionTestModel));
    }

    [TestMethod]
    public void WithGraphCapture_WithConfigAction_CallsConfigure()
    {
        // Arrange
        var model = new ExtensionTestModel();
        CUDAGraphProxy<ExtensionTestModel>? capturedProxy = null;

        // Act
        var result = model.WithGraphCapture(_stream!, proxy =>
        {
            capturedProxy = proxy;
        });

        // Assert
        Assert.IsNotNull(capturedProxy);
        Assert.IsNotNull(result);
    }

    [TestMethod]
    public void WithGraphCapture_WithNullConfigAction_DoesNotThrow()
    {
        // Arrange
        var model = new ExtensionTestModel();

        // Act & Assert - Should not throw
        var result = model.WithGraphCapture(_stream!, null);

        // Assert
        Assert.IsNotNull(result);
    }

    [TestMethod]
    public void WithGraphCapture_CustomManager_NullInstance_ThrowsException()
    {
        // Arrange
        var manager = new CUDAGraphManager();

        // Act & Assert
        Assert.ThrowsException<ArgumentNullException>(() =>
            ((ExtensionTestModel)null!).WithGraphCapture(_stream!, manager));
    }

    [TestMethod]
    public void WithGraphCapture_CustomManager_NullStream_ThrowsException()
    {
        // Arrange
        var model = new ExtensionTestModel();
        var manager = new CUDAGraphManager();

        // Act & Assert
        Assert.ThrowsException<ArgumentNullException>(() =>
            model.WithGraphCapture((CudaStream)null!, manager));
    }

    [TestMethod]
    public void WithGraphCapture_CustomManager_NullManager_ThrowsException()
    {
        // Arrange
        var model = new ExtensionTestModel();

        // Act & Assert
        Assert.ThrowsException<ArgumentNullException>(() =>
            model.WithGraphCapture(_stream!, (CUDAGraphManager)null!));
    }

    [TestMethod]
    public void WithGraphCapture_CustomManager_ValidParameters_ReturnsInstance()
    {
        // Arrange
        var model = new ExtensionTestModel();
        var manager = new CUDAGraphManager();

        // Act
        var result = model.WithGraphCapture(_stream!, manager);

        // Assert
        Assert.IsNotNull(result);
        Assert.IsInstanceOfType(result, typeof(ExtensionTestModel));
    }

    [TestMethod]
    public void CreateGraphProxy_NullInstance_ThrowsException()
    {
        // Act & Assert
        Assert.ThrowsException<ArgumentNullException>(() =>
            ((ExtensionTestModel)null!).CreateGraphProxy(_stream!));
    }

    [TestMethod]
    public void CreateGraphProxy_NullStream_ThrowsException()
    {
        // Arrange
        var model = new ExtensionTestModel();

        // Act & Assert
        Assert.ThrowsException<ArgumentNullException>(() =>
            model.CreateGraphProxy((CudaStream)null!));
    }

    [TestMethod]
    public void CreateGraphProxy_ValidParameters_ReturnsProxy()
    {
        // Arrange
        var model = new ExtensionTestModel();

        // Act
        var result = model.CreateGraphProxy(_stream!);

        // Assert
        Assert.IsNotNull(result);
        Assert.IsInstanceOfType(result, typeof(CUDAGraphProxy<ExtensionTestModel>));
    }

    [TestMethod]
    public void CreateGraphProxy_CustomManager_NullInstance_ThrowsException()
    {
        // Arrange
        var manager = new CUDAGraphManager();

        // Act & Assert
        Assert.ThrowsException<ArgumentNullException>(() =>
            ((ExtensionTestModel)null!).CreateGraphProxy(_stream!, manager));
    }

    [TestMethod]
    public void CreateGraphProxy_CustomManager_NullStream_ThrowsException()
    {
        // Arrange
        var model = new ExtensionTestModel();
        var manager = new CUDAGraphManager();

        // Act & Assert
        Assert.ThrowsException<ArgumentNullException>(() =>
            model.CreateGraphProxy((CudaStream)null!, manager));
    }

    [TestMethod]
    public void CreateGraphProxy_CustomManager_NullManager_ThrowsException()
    {
        // Arrange
        var model = new ExtensionTestModel();

        // Act & Assert
        Assert.ThrowsException<ArgumentNullException>(() =>
            model.CreateGraphProxy(_stream!, (CUDAGraphManager)null!));
    }

    [TestMethod]
    public void CreateGraphProxy_CustomManager_ValidParameters_ReturnsProxy()
    {
        // Arrange
        var model = new ExtensionTestModel();
        var manager = new CUDAGraphManager();

        // Act
        var result = model.CreateGraphProxy(_stream!, manager);

        // Assert
        Assert.IsNotNull(result);
        Assert.IsInstanceOfType(result, typeof(CUDAGraphProxy<ExtensionTestModel>));
        Assert.AreSame(manager, result.GraphManager);
    }
}
