using NUnit.Framework;
using RitterFramework.Core.Tensor;
using MLFramework.HAL;
using MLFramework.HAL.CUDA;

namespace MLFramework.Tests.HAL.CUDA;

/// <summary>
/// Tests for CUDA backend
/// Note: These tests require CUDA hardware to be available
/// </summary>
[TestFixture]
public class CudaBackendTests
{
    private CudaBackend? _backend;

    [SetUp]
    public void Setup()
    {
        if (!CudaAvailable())
            Assert.Inconclusive("CUDA not available");

        _backend = new CudaBackend();
        _backend.Initialize();
    }

    [TearDown]
    public void TearDown()
    {
        _backend?.Dispose();
    }

    [Test]
    public void Name_ReturnsCUDA()
    {
        Assert.AreEqual("CUDA", _backend!.Name);
    }

    [Test]
    public void Type_ReturnsDeviceTypeCUDA()
    {
        Assert.AreEqual(DeviceType.CUDA, _backend!.Type);
    }

    [Test]
    public void IsAvailable_ReturnsTrueWhenCudaPresent()
    {
        Assert.IsTrue(_backend!.IsAvailable);
    }

    [Test]
    public void SupportsOperation_Add_ReturnsTrue()
    {
        Assert.IsTrue(_backend!.SupportsOperation(Operation.Add));
    }

    [Test]
    public void SupportsOperation_SupportedOperations_ReturnTrue()
    {
        var supportedOps = new[]
        {
            Operation.Add, Operation.Subtract, Operation.Multiply, Operation.Divide,
            Operation.Sum, Operation.Mean, Operation.Max, Operation.Min,
            Operation.ReLU, Operation.Sigmoid, Operation.Tanh, Operation.Copy, Operation.Fill
        };

        foreach (var op in supportedOps)
        {
            Assert.IsTrue(_backend!.SupportsOperation(op), $"Operation {op} should be supported");
        }
    }

    [Test]
    public void ExecuteOperation_Add_WorksCorrectly()
    {
        // Create tensors on CPU first, then move to CUDA
        var aCpu = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f });
        var bCpu = Tensor.FromArray(new[] { 4.0f, 5.0f, 6.0f });

        var a = aCpu.To(Device.CUDA(0));
        var b = bCpu.To(Device.CUDA(0));

        var result = _backend!.ExecuteOperation(Operation.Add, new[] { a, b });
        var resultCpu = result.To(Device.CPU);

        var expected = new[] { 5.0f, 7.0f, 9.0f };
        Assert.AreEqual(expected, resultCpu.ToArray());
    }

    [Test]
    public void ExecuteOperation_Subtract_WorksCorrectly()
    {
        var aCpu = Tensor.FromArray(new[] { 5.0f, 7.0f, 9.0f });
        var bCpu = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f });

        var a = aCpu.To(Device.CUDA(0));
        var b = bCpu.To(Device.CUDA(0));

        var result = _backend!.ExecuteOperation(Operation.Subtract, new[] { a, b });
        var resultCpu = result.To(Device.CPU);

        var expected = new[] { 4.0f, 5.0f, 6.0f };
        Assert.AreEqual(expected, resultCpu.ToArray());
    }

    [Test]
    public void ExecuteOperation_Multiply_WorksCorrectly()
    {
        var aCpu = Tensor.FromArray(new[] { 2.0f, 3.0f, 4.0f });
        var bCpu = Tensor.FromArray(new[] { 5.0f, 6.0f, 7.0f });

        var a = aCpu.To(Device.CUDA(0));
        var b = bCpu.To(Device.CUDA(0));

        var result = _backend!.ExecuteOperation(Operation.Multiply, new[] { a, b });
        var resultCpu = result.To(Device.CPU);

        var expected = new[] { 10.0f, 18.0f, 28.0f };
        Assert.AreEqual(expected, resultCpu.ToArray());
    }

    [Test]
    public void ExecuteOperation_Divide_WorksCorrectly()
    {
        var aCpu = Tensor.FromArray(new[] { 10.0f, 12.0f, 14.0f });
        var bCpu = Tensor.FromArray(new[] { 2.0f, 3.0f, 7.0f });

        var a = aCpu.To(Device.CUDA(0));
        var b = bCpu.To(Device.CUDA(0));

        var result = _backend!.ExecuteOperation(Operation.Divide, new[] { a, b });
        var resultCpu = result.To(Device.CPU);

        var expected = new[] { 5.0f, 4.0f, 2.0f };
        CollectionAssert.AreEqual(expected, resultCpu.ToArray(), new FloatComparer(0.001f));
    }

    [Test]
    public void ExecuteOperation_Sum_WorksCorrectly()
    {
        var aCpu = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f });
        var a = aCpu.To(Device.CUDA(0));

        var result = _backend!.ExecuteOperation(Operation.Sum, new[] { a });
        var resultCpu = result.To(Device.CPU);

        Assert.AreEqual(6.0f, resultCpu.ToArray()[0], 0.001f);
    }

    [Test]
    public void ExecuteOperation_Mean_WorksCorrectly()
    {
        var aCpu = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f, 4.0f });
        var a = aCpu.To(Device.CUDA(0));

        var result = _backend!.ExecuteOperation(Operation.Mean, new[] { a });
        var resultCpu = result.To(Device.CPU);

        Assert.AreEqual(2.5f, resultCpu.ToArray()[0], 0.001f);
    }

    [Test]
    public void ExecuteOperation_Max_WorksCorrectly()
    {
        var aCpu = Tensor.FromArray(new[] { 1.0f, 5.0f, 3.0f, 9.0f, 2.0f });
        var a = aCpu.To(Device.CUDA(0));

        var result = _backend!.ExecuteOperation(Operation.Max, new[] { a });
        var resultCpu = result.To(Device.CPU);

        Assert.AreEqual(9.0f, resultCpu.ToArray()[0], 0.001f);
    }

    [Test]
    public void ExecuteOperation_Min_WorksCorrectly()
    {
        var aCpu = Tensor.FromArray(new[] { 7.0f, 5.0f, 3.0f, 9.0f, 2.0f });
        var a = aCpu.To(Device.CUDA(0));

        var result = _backend!.ExecuteOperation(Operation.Min, new[] { a });
        var resultCpu = result.To(Device.CPU);

        Assert.AreEqual(2.0f, resultCpu.ToArray()[0], 0.001f);
    }

    [Test]
    public void ExecuteOperation_ReLU_WorksCorrectly()
    {
        var aCpu = Tensor.FromArray(new[] { -1.0f, 0.0f, 1.0f, -2.0f, 3.0f });
        var a = aCpu.To(Device.CUDA(0));

        var result = _backend!.ExecuteOperation(Operation.ReLU, new[] { a });
        var resultCpu = result.To(Device.CPU);

        var expected = new[] { 0.0f, 0.0f, 1.0f, 0.0f, 3.0f };
        CollectionAssert.AreEqual(expected, resultCpu.ToArray(), new FloatComparer(0.001f));
    }

    [Test]
    public void ExecuteOperation_Sigmoid_WorksCorrectly()
    {
        var aCpu = Tensor.FromArray(new[] { 0.0f });
        var a = aCpu.To(Device.CUDA(0));

        var result = _backend!.ExecuteOperation(Operation.Sigmoid, new[] { a });
        var resultCpu = result.To(Device.CPU);

        // sigmoid(0) = 1 / (1 + exp(0)) = 0.5
        Assert.AreEqual(0.5f, resultCpu.ToArray()[0], 0.001f);
    }

    [Test]
    public void ExecuteOperation_Tanh_WorksCorrectly()
    {
        var aCpu = Tensor.FromArray(new[] { 0.0f });
        var a = aCpu.To(Device.CUDA(0));

        var result = _backend!.ExecuteOperation(Operation.Tanh, new[] { a });
        var resultCpu = result.To(Device.CPU);

        // tanh(0) = 0
        Assert.AreEqual(0.0f, resultCpu.ToArray()[0], 0.001f);
    }

    [Test]
    public void ExecuteOperation_Copy_WorksCorrectly()
    {
        var aCpu = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f });
        var a = aCpu.To(Device.CUDA(0));

        var result = _backend!.ExecuteOperation(Operation.Copy, new[] { a });
        var resultCpu = result.To(Device.CPU);

        var expected = new[] { 1.0f, 2.0f, 3.0f };
        CollectionAssert.AreEqual(expected, resultCpu.ToArray(), new FloatComparer(0.001f));
    }

    [Test]
    public void ExecuteOperation_WithEmptyInputs_ThrowsException()
    {
        Assert.Throws<ArgumentException>(() =>
        {
            _backend!.ExecuteOperation(Operation.Add, Array.Empty<Tensor>());
        });
    }

    [Test]
    public void ExecuteOperation_UnsupportedOperation_ThrowsException()
    {
        var aCpu = Tensor.FromArray(new[] { 1.0f });
        var a = aCpu.To(Device.CUDA(0));

        Assert.Throws<NotSupportedException>(() =>
        {
            _backend!.ExecuteOperation(Operation.Conv2D, new[] { a });
        });
    }

    [Test]
    public void ExecuteOperation_WithWrongInputCount_ThrowsException()
    {
        var aCpu = Tensor.FromArray(new[] { 1.0f });
        var a = aCpu.To(Device.CUDA(0));

        Assert.Throws<ArgumentException>(() =>
        {
            _backend!.ExecuteOperation(Operation.Add, new[] { a });
        });
    }

    [Test]
    public void DeviceTransfer_CpuToCpu_NoOp()
    {
        var aCpu = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f });
        var b = aCpu.To(Device.CPU);

        CollectionAssert.AreEqual(aCpu.ToArray(), b.ToArray());
    }

    [Test]
    public void DeviceTransfer_CpuToGpu_ReturnsCorrectData()
    {
        var aCpu = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f });
        var aGpu = aCpu.To(Device.CUDA(0));

        Assert.IsNotNull(aGpu);
        Assert.AreEqual(3, aGpu.Size);

        var result = aGpu.To(Device.CPU);
        CollectionAssert.AreEqual(aCpu.ToArray(), result.ToArray());
    }

    [Test]
    public void DeviceTransfer_GpuToCpu_ReturnsCorrectData()
    {
        var aCpu = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f });
        var aGpu = aCpu.To(Device.CUDA(0));
        var result = aGpu.To(Device.CPU);

        CollectionAssert.AreEqual(aCpu.ToArray(), result.ToArray());
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

    /// <summary>
    /// Helper class for comparing floating point arrays with tolerance
    /// </summary>
    private class FloatComparer : IComparer
    {
        private readonly float _tolerance;

        public FloatComparer(float tolerance = 0.0001f)
        {
            _tolerance = tolerance;
        }

        public int Compare(object? x, object? y)
        {
            var a = (float)x!;
            var b = (float)y!;
            return Math.Abs(a - b) < _tolerance ? 0 : a.CompareTo(b);
        }
    }
}
