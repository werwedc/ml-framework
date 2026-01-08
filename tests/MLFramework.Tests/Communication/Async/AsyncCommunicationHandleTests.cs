using MLFramework.Communication.Async;
using MLFramework.Communication;
using MLFramework.Communication.Operations;
using RitterFramework.Core.Tensor;
using System;
using System.Threading.Tasks;
using Xunit;
using Moq;

namespace MLFramework.Tests.Communication.Async;

public class AsyncCommunicationHandleTests
{
    [Fact]
    public async Task Constructor_WithValidTask_CreatesHandle()
    {
        // Arrange
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f });
        var task = Task.FromResult(tensor);

        // Act
        var handle = new AsyncCommunicationHandle(task);

        // Assert
        Assert.False(handle.IsCompleted);
        Assert.NotNull(handle);
    }

    [Fact]
    public void Wait_WithCompletedTask_Succeeds()
    {
        // Arrange
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f });
        var task = Task.FromResult(tensor);
        var handle = new AsyncCommunicationHandle(task);

        // Act
        handle.Wait();

        // Assert
        Assert.True(handle.IsCompleted);
    }

    [Fact]
    public void GetResult_AfterCompletion_ReturnsTensor()
    {
        // Arrange
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f });
        var task = Task.FromResult(tensor);
        var handle = new AsyncCommunicationHandle(task);
        handle.Wait();

        // Act
        var result = handle.GetResult();

        // Assert
        Assert.NotNull(result);
        Assert.Equal(3, result.Size);
    }

    [Fact]
    public void GetResult_BeforeCompletion_ThrowsInvalidOperationException()
    {
        // Arrange
        var task = new Task<Tensor>(() => Tensor.FromArray(new float[] { 1.0f, 2.0f }));
        var handle = new AsyncCommunicationHandle(task);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => handle.GetResult());
    }

    [Fact]
    public void TryWait_WithCompletedTask_ReturnsTrue()
    {
        // Arrange
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        var task = Task.FromResult(tensor);
        var handle = new AsyncCommunicationHandle(task);

        // Act
        var result = handle.TryWait(1000);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void Cancel_MarksHandleAsCancelled()
    {
        // Arrange
        var task = new Task<Tensor>(() => Tensor.FromArray(new float[] { 1.0f, 2.0f }));
        var handle = new AsyncCommunicationHandle(task);

        // Act
        handle.Cancel();

        // Assert
        Assert.True(handle.IsCancelled);
    }

    [Fact]
    public void AsTask_ReturnsUnderlyingTask()
    {
        // Arrange
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        var task = Task.FromResult(tensor);
        var handle = new AsyncCommunicationHandle(task);

        // Act
        var result = handle.AsTask();

        // Assert
        Assert.NotNull(result);
        Assert.Same(task, result);
    }
}
