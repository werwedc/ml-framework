using MLFramework.HAL;
using RitterFramework.Core.Tensor;
using System;
using Xunit;

namespace RitterFramework.Tests.HAL;

public class DeviceTransferTests
{
    [Fact]
    public void To_SameDevice_ReturnsSameTensor()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var shape = new int[] { 2, 2 };
        var tensor = new Tensor(data, shape);

        // Act - Use HAL extension explicitly
        var result = TensorHALExtensions.To(tensor, Device.CPU);

        // Assert
        Assert.Same(tensor, result);
    }

    [Fact]
    public void To_CpuDevice_CopiesData()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f };
        var tensor = Tensor.FromArray(data);

        // Act - CPU to CPU is a no-op, should return same tensor
        // Use HAL extension explicitly
        var result = TensorHALExtensions.To(tensor, Device.CPU);

        // Assert
        Assert.Same(tensor, result);
        Assert.Equal(data, tensor.Data);
    }

    [Fact]
    public void To_NullTensor_ThrowsArgumentNullException()
    {
        // Arrange
        Tensor? tensor = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => TensorHALExtensions.To(tensor!, Device.CPU));
    }

    [Fact]
    public void To_NullDevice_ThrowsArgumentNullException()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f };
        var tensor = Tensor.FromArray(data);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => TensorHALExtensions.To(tensor, null!));
    }

    [Fact]
    public void To_UnsupportedDevice_ThrowsNotImplementedException()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f };
        var tensor = Tensor.FromArray(data);

        // Act & Assert
        // CUDA is not yet implemented, Device.CUDA throws NotImplementedException
        Assert.Throws<NotImplementedException>(() => TensorHALExtensions.To(tensor, Device.CUDA(0)));
    }

    [Fact]
    public void Zeros_OnDevice_CreatesTensor()
    {
        // Arrange
        var shape = new int[] { 2, 3 };

        // Act
        var tensor = TensorHALExtensions.Zeros(shape, Device.CPU);

        // Assert
        Assert.Equal(shape, tensor.Shape);
        Assert.Equal(6, tensor.Size);
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Assert.Equal(0f, tensor[new int[] { i, j }]);
            }
        }
    }

    [Fact]
    public void FromArray_OnDevice_CreatesTensor()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f };

        // Act
        var tensor = TensorHALExtensions.FromArray(data, Device.CPU);

        // Assert
        Assert.Equal(data.Length, tensor.Size);
        Assert.Equal(data, tensor.Data);
    }

    [Fact]
    public void FromArray_OnUnsupportedDevice_ThrowsNotImplementedException()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f };

        // Act & Assert
        // CUDA is not yet implemented, Device.CUDA throws NotImplementedException
        Assert.Throws<NotImplementedException>(() => TensorHALExtensions.FromArray(data, Device.CUDA(0)));
    }

    [Fact]
    public void WithDataPointer_PinsMemoryAndExecutesAction()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f };
        var tensor = Tensor.FromArray(data);
        IntPtr capturedPointer = IntPtr.Zero;

        // Act
        tensor.WithDataPointer(ptr => capturedPointer = ptr);

        // Assert
        Assert.NotEqual(IntPtr.Zero, capturedPointer);
    }

    [Fact]
    public void ToLongArray_ConvertsIntArrayToLongArray()
    {
        // Arrange
        var intShape = new int[] { 1, 2, 3 };

        // Act
        var longShape = intShape.ToLongArray();

        // Assert
        Assert.Equal(3, longShape.Length);
        Assert.Equal(1L, longShape[0]);
        Assert.Equal(2L, longShape[1]);
        Assert.Equal(3L, longShape[2]);
    }

    [Fact]
    public void ToIntArray_ConvertsLongArrayToIntArray()
    {
        // Arrange
        var longShape = new long[] { 1L, 2L, 3L };

        // Act
        var intShape = longShape.ToIntArray();

        // Assert
        Assert.Equal(3, intShape.Length);
        Assert.Equal(1, intShape[0]);
        Assert.Equal(2, intShape[1]);
        Assert.Equal(3, intShape[2]);
    }

    [Fact]
    public void ToIntArray_Overflow_ThrowsOverflowException()
    {
        // Arrange
        var longShape = new long[] { 1L, (long)int.MaxValue + 1L, 3L };

        // Act & Assert
        Assert.Throws<OverflowException>(() => longShape.ToIntArray());
    }
}
