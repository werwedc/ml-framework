namespace MLFramework.Communication.Tests;

using MLFramework.Communication.Operations;
using MLFramework.Communication.Operations.Async;
using RitterFramework.Core.Tensor;
using Xunit;

/// <summary>
/// Unit tests for asynchronous broadcast operation
/// </summary>
public class BroadcastAsyncTests
{
    [Fact]
    public void BroadcastTensorAsync_WithNullBackend_ThrowsArgumentNullException()
    {
        // Arrange
        IAsyncCommunicationBackend backend = null;
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => 
            BroadcastAsync.BroadcastTensorAsync(backend, tensor, 0));
    }

    [Fact]
    public void BroadcastTensorAsync_WithNullTensor_ThrowsArgumentNullException()
    {
        // Arrange
        var backend = new MockAsyncCommunicationBackend(0, 4);
        Tensor tensor = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => 
            BroadcastAsync.BroadcastTensorAsync(backend, tensor, 0));
    }

    [Fact]
    public void BroadcastTensorAsync_WithInvalidRootRank_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var backend = new MockAsyncCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => 
            BroadcastAsync.BroadcastTensorAsync(backend, tensor, -1));
        
        Assert.Throws<ArgumentOutOfRangeException>(() => 
            BroadcastAsync.BroadcastTensorAsync(backend, tensor, 4));
    }

    [Fact]
    public void BroadcastTensorAsync_WithValidParameters_ReturnsHandle()
    {
        // Arrange
        var backend = new MockAsyncCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act
        var handle = BroadcastAsync.BroadcastTensorAsync(backend, tensor, 0);

        // Assert
        Assert.NotNull(handle);
        Assert.IsType<PendingOperationHandle>(handle);
    }

    [Fact]
    public void BroadcastTensorAsync_WaitForCompletion_Succeeds()
    {
        // Arrange
        var backend = new MockAsyncCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });
        var handle = BroadcastAsync.BroadcastTensorAsync(backend, tensor, 0);

        // Act
        handle.Wait();

        // Assert
        Assert.True(handle.IsCompleted);
        var result = handle.GetResult();
        Assert.NotNull(result);
    }
}
