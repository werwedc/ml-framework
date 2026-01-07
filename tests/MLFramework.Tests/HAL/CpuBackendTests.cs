using MLFramework.HAL;
using RitterFramework.Core.Tensor;
using Xunit;

namespace MLFramework.Tests.HAL;

/// <summary>
/// Tests for CpuBackend class
/// </summary>
public class CpuBackendTests
{
    private readonly CpuBackend _backend;

    public CpuBackendTests()
    {
        _backend = new CpuBackend();
    }

    [Fact]
    public void Name_ReturnsManagedCPU()
    {
        Assert.Equal("ManagedCPU", _backend.Name);
    }

    [Fact]
    public void Type_ReturnsCPU()
    {
        Assert.Equal(DeviceType.CPU, _backend.Type);
    }

    [Fact]
    public void IsAvailable_ReturnsTrue()
    {
        Assert.True(_backend.IsAvailable);
    }

    [Fact]
    public void SupportsOperation_SupportedOperations_ReturnsTrue()
    {
        Assert.True(_backend.SupportsOperation(Operation.Add));
        Assert.True(_backend.SupportsOperation(Operation.Subtract));
        Assert.True(_backend.SupportsOperation(Operation.Multiply));
        Assert.True(_backend.SupportsOperation(Operation.Divide));
        Assert.True(_backend.SupportsOperation(Operation.Sum));
        Assert.True(_backend.SupportsOperation(Operation.Mean));
        Assert.True(_backend.SupportsOperation(Operation.ReLU));
        Assert.True(_backend.SupportsOperation(Operation.Sigmoid));
        Assert.True(_backend.SupportsOperation(Operation.Tanh));
        Assert.True(_backend.SupportsOperation(Operation.Copy));
        Assert.True(_backend.SupportsOperation(Operation.Cast));
        Assert.True(_backend.SupportsOperation(Operation.Reshape));
    }

    [Fact]
    public void SupportsOperation_UnsupportedOperations_ReturnsFalse()
    {
        Assert.False(_backend.SupportsOperation(Operation.MatMul));
        Assert.False(_backend.SupportsOperation(Operation.Conv2D));
        Assert.False(_backend.SupportsOperation(Operation.MaxPool2D));
    }

    [Fact]
    public void ExecuteOperation_Add_WorksCorrectly()
    {
        var a = TensorHALExtensions.FromArray(new[] { 1.0f, 2.0f, 3.0f }, Device.CPU);
        var b = TensorHALExtensions.FromArray(new[] { 4.0f, 5.0f, 6.0f }, Device.CPU);

        var result = _backend.ExecuteOperation(Operation.Add, new[] { a, b });

        var expected = new[] { 5.0f, 7.0f, 9.0f };
        Assert.Equal(expected, result.Data);
    }

    [Fact]
    public void ExecuteOperation_Subtract_WorksCorrectly()
    {
        var a = TensorHALExtensions.FromArray(new[] { 5.0f, 7.0f, 9.0f }, Device.CPU);
        var b = TensorHALExtensions.FromArray(new[] { 1.0f, 2.0f, 3.0f }, Device.CPU);

        var result = _backend.ExecuteOperation(Operation.Subtract, new[] { a, b });

        var expected = new[] { 4.0f, 5.0f, 6.0f };
        Assert.Equal(expected, result.Data);
    }

    [Fact]
    public void ExecuteOperation_Multiply_WorksCorrectly()
    {
        var a = TensorHALExtensions.FromArray(new[] { 2.0f, 3.0f, 4.0f }, Device.CPU);
        var b = TensorHALExtensions.FromArray(new[] { 3.0f, 4.0f, 5.0f }, Device.CPU);

        var result = _backend.ExecuteOperation(Operation.Multiply, new[] { a, b });

        var expected = new[] { 6.0f, 12.0f, 20.0f };
        Assert.Equal(expected, result.Data);
    }

    [Fact]
    public void ExecuteOperation_Divide_WorksCorrectly()
    {
        var a = TensorHALExtensions.FromArray(new[] { 6.0f, 12.0f, 20.0f }, Device.CPU);
        var b = TensorHALExtensions.FromArray(new[] { 2.0f, 3.0f, 4.0f }, Device.CPU);

        var result = _backend.ExecuteOperation(Operation.Divide, new[] { a, b });

        var expected = new[] { 3.0f, 4.0f, 5.0f };
        Assert.Equal(expected, result.Data);
    }

    [Fact]
    public void ExecuteOperation_Sum_WorksCorrectly()
    {
        var a = TensorHALExtensions.FromArray(new[] { 1.0f, 2.0f, 3.0f }, Device.CPU);

        var result = _backend.ExecuteOperation(Operation.Sum, new[] { a });

        Assert.Single(result.Data);
        Assert.Equal(6.0f, result.Data[0]);
    }

    [Fact]
    public void ExecuteOperation_Mean_WorksCorrectly()
    {
        var a = TensorHALExtensions.FromArray(new[] { 2.0f, 4.0f, 6.0f }, Device.CPU);

        var result = _backend.ExecuteOperation(Operation.Mean, new[] { a });

        Assert.Single(result.Data);
        Assert.Equal(4.0f, result.Data[0], 0.001f); // Small tolerance for floating point
    }

    [Fact]
    public void ExecuteOperation_ReLU_WorksCorrectly()
    {
        var a = TensorHALExtensions.FromArray(new[] { -1.0f, 0.0f, 1.0f, 2.0f, -3.0f }, Device.CPU);

        var result = _backend.ExecuteOperation(Operation.ReLU, new[] { a });

        var expected = new[] { 0.0f, 0.0f, 1.0f, 2.0f, 0.0f };
        Assert.Equal(expected, result.Data);
    }

    [Fact]
    public void ExecuteOperation_Sigmoid_WorksCorrectly()
    {
        var a = TensorHALExtensions.FromArray(new[] { 0.0f }, Device.CPU);

        var result = _backend.ExecuteOperation(Operation.Sigmoid, new[] { a });

        // Sigmoid(0) = 0.5
        Assert.Single(result.Data);
        Assert.Equal(0.5f, result.Data[0], 0.001f);
    }

    [Fact]
    public void ExecuteOperation_Tanh_WorksCorrectly()
    {
        var a = TensorHALExtensions.FromArray(new[] { 0.0f }, Device.CPU);

        var result = _backend.ExecuteOperation(Operation.Tanh, new[] { a });

        // Tanh(0) = 0
        Assert.Single(result.Data);
        Assert.Equal(0.0f, result.Data[0], 0.001f);
    }

    [Fact]
    public void ExecuteOperation_Copy_WorksCorrectly()
    {
        var a = TensorHALExtensions.FromArray(new[] { 1.0f, 2.0f, 3.0f }, Device.CPU);

        var result = _backend.ExecuteOperation(Operation.Copy, new[] { a });

        Assert.Equal(a.Data, result.Data);
        Assert.NotSame(a, result); // Should be different objects
    }

    [Fact]
    public void ExecuteOperation_Cast_WorksCorrectly()
    {
        var a = TensorHALExtensions.FromArray(new[] { 1.0f, 2.0f, 3.0f }, Device.CPU);

        var result = _backend.ExecuteOperation(Operation.Cast, new[] { a });

        Assert.Equal(a.Data, result.Data);
        Assert.NotSame(a, result);
    }

    [Fact]
    public void ExecuteOperation_Reshape_WorksCorrectly()
    {
        var a = TensorHALExtensions.FromArray(new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }, Device.CPU);
        var shape = TensorHALExtensions.FromArray(new[] { 2.0f, 3.0f }, Device.CPU);

        var result = _backend.ExecuteOperation(Operation.Reshape, new[] { a, shape });

        Assert.Equal(new[] { 2, 3 }, result.Shape);
        Assert.Equal(6, result.Size);
    }

    [Fact]
    public void ExecuteOperation_WithNullInputs_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => _backend.ExecuteOperation(Operation.Add, null!));
    }

    [Fact]
    public void ExecuteOperation_WithEmptyInputs_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => _backend.ExecuteOperation(Operation.Add, Array.Empty<Tensor>()));
    }

    [Fact]
    public void ExecuteOperation_UnsupportedOperation_ThrowsNotSupportedException()
    {
        var a = TensorHALExtensions.FromArray(new[] { 1.0f }, Device.CPU);

        Assert.Throws<NotSupportedException>(() =>
            _backend.ExecuteOperation(Operation.Conv2D, new[] { a }));
    }

    [Fact]
    public void Initialize_DoesNotThrow()
    {
        var exception = Record.Exception(() => _backend.Initialize());

        Assert.Null(exception);
    }

    [Fact]
    public void ExecuteOperation_AddWithDifferentShapes_ThrowsArgumentException()
    {
        var a = TensorHALExtensions.FromArray(new[] { 1.0f, 2.0f }, Device.CPU);
        var b = TensorHALExtensions.FromArray(new[] { 1.0f }, Device.CPU);

        Assert.Throws<ArgumentException>(() =>
            _backend.ExecuteOperation(Operation.Add, new[] { a, b }));
    }

    [Fact]
    public void ExecuteOperation_MultiplyWithDifferentShapes_ThrowsArgumentException()
    {
        var a = TensorHALExtensions.FromArray(new[] { 1.0f, 2.0f }, Device.CPU);
        var b = TensorHALExtensions.FromArray(new[] { 1.0f }, Device.CPU);

        Assert.Throws<ArgumentException>(() =>
            _backend.ExecuteOperation(Operation.Multiply, new[] { a, b }));
    }
}
