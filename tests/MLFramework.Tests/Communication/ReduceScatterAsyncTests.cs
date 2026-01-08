namespace MLFramework.Communication.Tests;

using MLFramework.Communication;
using MLFramework.Communication.Operations.Async;
using RitterFramework.Core.Tensor;
using Xunit;

/// <summary>
/// Unit tests for asynchronous reduce-scatter operation
/// </summary>
public class ReduceScatterAsyncTests
{
    [Fact]
    public void ReduceScatterTensorAsync_WithNullBackend_ThrowsArgumentNullException()
    {
        // Arrange
        IAsyncCommunicationBackend backend = null;
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            ReduceScatterAsync.ReduceScatterTensorAsync(backend, tensor, ReduceOp.Sum));
    }

    [Fact]
    public void ReduceScatterTensorAsync_WithNullTensor_ThrowsArgumentNullException()
    {
        // Arrange
        var backend = new MockAsyncCommunicationBackend(0, 4);
        Tensor tensor = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            ReduceScatterAsync.ReduceScatterTensorAsync(backend, tensor, ReduceOp.Sum));
    }

    [Fact]
    public void ReduceScatterTensorAsync_WithValidParameters_Succeeds()
    {
        // Arrange
        var backend = new MockAsyncCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act
        var handle = ReduceScatterAsync.ReduceScatterTensorAsync(backend, tensor, ReduceOp.Sum);

        // Assert
        Assert.NotNull(handle);
        Assert.False(handle.IsCompleted);

        // Wait for completion
        handle.Wait();
        Assert.True(handle.IsCompleted);
    }

    [Fact]
    public void ReduceScatterTensorAsync_WaitAndGetResult_Succeeds()
    {
        // Arrange
        var backend = new MockAsyncCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act
        var handle = ReduceScatterAsync.ReduceScatterTensorAsync(backend, tensor, ReduceOp.Sum);
        handle.Wait();
        var result = handle.GetResult();

        // Assert
        Assert.NotNull(result);
        Assert.Equal(tensor.Size, result.Size);
    }

    [Fact]
    public void ReduceScatterTensorAsync_WithMaxOperation_Succeeds()
    {
        // Arrange
        var backend = new MockAsyncCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act
        var handle = ReduceScatterAsync.ReduceScatterTensorAsync(backend, tensor, ReduceOp.Max);
        handle.Wait();

        // Assert
        Assert.True(handle.IsCompleted);
        Assert.NotNull(handle.GetResult());
    }
}
