namespace MLFramework.Communication.Tests;

using MLFramework.Communication;
using MLFramework.Communication.Operations;
using RitterFramework.Core.Tensor;
using Xunit;

/// <summary>
/// Unit tests for synchronous reduce-scatter operation
/// </summary>
public class ReduceScatterTests
{
    [Fact]
    public void ReduceScatterTensor_WithNullBackend_ThrowsArgumentNullException()
    {
        // Arrange
        ICommunicationBackend backend = null;
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            ReduceScatter.ReduceScatterTensor(backend, tensor, ReduceOp.Sum));
    }

    [Fact]
    public void ReduceScatterTensor_WithNullTensor_ThrowsArgumentNullException()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        Tensor tensor = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            ReduceScatter.ReduceScatterTensor(backend, tensor, ReduceOp.Sum));
    }

    [Fact]
    public void ReduceScatterTensor_WithValidParameters_Succeeds()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act
        var result = ReduceScatter.ReduceScatterTensor(backend, tensor, ReduceOp.Sum);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(tensor.Size, result.Size);
    }

    [Fact]
    public void ReduceScatterTensor_WithSumOperation_Succeeds()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act
        var result = ReduceScatter.ReduceScatterTensor(backend, tensor, ReduceOp.Sum);

        // Assert
        Assert.NotNull(result);
    }

    [Fact]
    public void ReduceScatterTensor_WithMaxOperation_Succeeds()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act
        var result = ReduceScatter.ReduceScatterTensor(backend, tensor, ReduceOp.Max);

        // Assert
        Assert.NotNull(result);
    }

    [Fact]
    public void ReduceScatterTensors_WithMultipleTensors_Succeeds()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        var tensors = new[]
        {
            Tensor.FromArray(new float[] { 1, 2, 3 }),
            Tensor.FromArray(new float[] { 4, 5, 6 })
        };

        // Act
        var results = ReduceScatter.ReduceScatterTensors(backend, tensors, ReduceOp.Sum);

        // Assert
        Assert.NotNull(results);
        Assert.Equal(2, results.Count);
    }
}
