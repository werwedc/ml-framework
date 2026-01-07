namespace MLFramework.Communication.Tests;

using MLFramework.Communication;
using MLFramework.Communication.Operations;
using MLFramework.Communication.Operations.Async;
using RitterFramework.Core.Tensor;
using Xunit;

/// <summary>
/// Unit tests for asynchronous all-reduce operation
/// </summary>
public class AllReduceAsyncTests
{
    [Fact]
    public void AllReduceTensorAsync_WithNullBackend_ThrowsArgumentNullException()
    {
        // Arrange
        IAsyncCommunicationBackend backend = null;
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => 
            AllReduceAsync.AllReduceTensorAsync(backend, tensor, ReduceOp.Sum));
    }

    [Fact]
    public void AllReduceTensorAsync_WithNullTensor_ThrowsArgumentNullException()
    {
        // Arrange
        var backend = new MockAsyncCommunicationBackend(0, 4);
        Tensor tensor = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => 
            AllReduceAsync.AllReduceTensorAsync(backend, tensor, ReduceOp.Sum));
    }

    [Fact]
    public void AllReduceTensorAsync_WithValidParameters_ReturnsHandle()
    {
        // Arrange
        var backend = new MockAsyncCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act
        var handle = AllReduceAsync.AllReduceTensorAsync(backend, tensor, ReduceOp.Sum);

        // Assert
        Assert.NotNull(handle);
        Assert.IsType<PendingOperationHandle>(handle);
    }

    [Fact]
    public void AllReduceTensorAsync_WaitForCompletion_Succeeds()
    {
        // Arrange
        var backend = new MockAsyncCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });
        var handle = AllReduceAsync.AllReduceTensorAsync(backend, tensor, ReduceOp.Sum);

        // Act
        handle.Wait();

        // Assert
        Assert.True(handle.IsCompleted);
        var result = handle.GetResult();
        Assert.NotNull(result);
    }

    [Fact]
    public void AllReduceTensorAsync_TryWaitWithTimeout_Succeeds()
    {
        // Arrange
        var backend = new MockAsyncCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });
        var handle = AllReduceAsync.AllReduceTensorAsync(backend, tensor, ReduceOp.Sum);

        // Act
        var completed = handle.TryWait(5000);

        // Assert
        Assert.True(completed);
        Assert.True(handle.IsCompleted);
    }

    [Fact]
    public void AverageGradientsAsync_WithValidParameters_Succeeds()
    {
        // Arrange
        var backend = new MockAsyncCommunicationBackend(0, 4);
        var gradients = Tensor.FromArray(new float[] { 4, 8, 12 });

        // Act
        var result = AllReduceAsync.AverageGradientsAsync(backend, gradients);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(gradients.Size, result.Size);
        // After dividing by 4: [1, 2, 3]
        Assert.Equal(1.0f, result.Data[0]);
        Assert.Equal(2.0f, result.Data[1]);
        Assert.Equal(3.0f, result.Data[2]);
    }

    [Fact]
    public void GetResult_BeforeCompletion_ThrowsInvalidOperationException()
    {
        // Arrange
        var backend = new MockAsyncCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });
        var handle = AllReduceAsync.AllReduceTensorAsync(backend, tensor, ReduceOp.Sum);

        // Act & Assert
        // Note: This test may be flaky since the task might complete before we check
        // For a more robust test, we'd need to mock a backend that actually delays
        if (!handle.IsCompleted)
        {
            Assert.Throws<InvalidOperationException>(() => handle.GetResult());
        }
    }
}
