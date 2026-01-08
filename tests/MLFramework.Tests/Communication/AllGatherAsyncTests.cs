namespace MLFramework.Communication.Tests;

using MLFramework.Communication;
using MLFramework.Communication.Operations.Async;
using RitterFramework.Core.Tensor;
using Xunit;

/// <summary>
/// Unit tests for asynchronous all-gather operation
/// </summary>
public class AllGatherAsyncTests
{
    [Fact]
    public void AllGatherTensorAsync_WithNullBackend_ThrowsArgumentNullException()
    {
        // Arrange
        IAsyncCommunicationBackend backend = null;
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            AllGatherAsync.AllGatherTensorAsync(backend, tensor));
    }

    [Fact]
    public void AllGatherTensorAsync_WithNullTensor_ThrowsArgumentNullException()
    {
        // Arrange
        var backend = new MockAsyncCommunicationBackend(0, 4);
        Tensor tensor = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            AllGatherAsync.AllGatherTensorAsync(backend, tensor));
    }

    [Fact]
    public void AllGatherTensorAsync_WithValidParameters_Succeeds()
    {
        // Arrange
        var backend = new MockAsyncCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act
        var handle = AllGatherAsync.AllGatherTensorAsync(backend, tensor);

        // Assert
        Assert.NotNull(handle);
        Assert.False(handle.IsCompleted);

        // Wait for completion
        handle.Wait();
        Assert.True(handle.IsCompleted);
    }

    [Fact]
    public void AllGatherTensorAsync_WaitAndGetResult_Succeeds()
    {
        // Arrange
        var backend = new MockAsyncCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act
        var handle = AllGatherAsync.AllGatherTensorAsync(backend, tensor);
        handle.Wait();
        var result = handle.GetResult();

        // Assert
        Assert.NotNull(result);
        Assert.Equal(tensor.Size, result.Size);
    }

    [Fact]
    public void AllGatherTensorsAsync_WithValidParameters_Succeeds()
    {
        // Arrange
        var backend = new MockAsyncCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act
        var handle = AllGatherAsync.AllGatherTensorsAsync(backend, tensor);

        // Assert
        Assert.NotNull(handle);
        handle.Wait();
        Assert.True(handle.IsCompleted);
    }
}
