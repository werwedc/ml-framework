using MLFramework.Communication.Async;
using RitterFramework.Core.Tensor;
using System;
using System.Threading.Tasks;
using Xunit;
using Moq;

namespace MLFramework.Tests.Communication.Async;

public class CommunicationOperationQueueTests
{
    [Fact]
    public void Enqueue_AddsOperationToQueue()
    {
        // Arrange
        var queue = new CommunicationOperationQueue();
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        var task = Task.FromResult(tensor);
        var handle = new AsyncCommunicationHandle(task);

        // Act
        queue.Enqueue(handle);

        // Assert
        Assert.Equal(1, queue.PendingOperationsCount);
    }

    [Fact]
    public void WaitForAll_WaitsForAllOperations()
    {
        // Arrange
        var queue = new CommunicationOperationQueue();
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        var task1 = Task.FromResult(tensor);
        var task2 = Task.FromResult(tensor);
        var handle1 = new AsyncCommunicationHandle(task1);
        var handle2 = new AsyncCommunicationHandle(task2);
        queue.Enqueue(handle1);
        queue.Enqueue(handle2);

        // Act
        queue.WaitForAll();

        // Assert
        Assert.Equal(0, queue.PendingOperationsCount);
    }

    [Fact]
    public void WaitForAny_ReturnsIndexOfFirstCompleted()
    {
        // Arrange
        var queue = new CommunicationOperationQueue();
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        var task1 = Task.FromResult(tensor);
        var task2 = Task.FromResult(tensor);
        var handle1 = new AsyncCommunicationHandle(task1);
        var handle2 = new AsyncCommunicationHandle(task2);
        queue.Enqueue(handle1);
        queue.Enqueue(handle2);

        // Act
        var index = queue.WaitForAny();

        // Assert
        Assert.True(index >= 0 && index < 2);
    }

    [Fact]
    public void ClearCompleted_RemovesCompletedOperations()
    {
        // Arrange
        var queue = new CommunicationOperationQueue();
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        var task = Task.FromResult(tensor);
        var handle = new AsyncCommunicationHandle(task);
        queue.Enqueue(handle);
        handle.Wait();

        // Act
        queue.ClearCompleted();

        // Assert
        Assert.Equal(0, queue.PendingOperationsCount);
    }

    [Fact]
    public void CancelAll_CancelsAllAsyncHandles()
    {
        // Arrange
        var queue = new CommunicationOperationQueue();
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        var task1 = new Task<Tensor>(() => tensor);
        var task2 = new Task<Tensor>(() => tensor);
        var handle1 = new AsyncCommunicationHandle(task1);
        var handle2 = new AsyncCommunicationHandle(task2);
        queue.Enqueue(handle1);
        queue.Enqueue(handle2);

        // Act
        queue.CancelAll();

        // Assert
        Assert.True(handle1.IsCancelled);
        Assert.True(handle2.IsCancelled);
    }

    [Fact]
    public void TryWaitForAll_WithTimeout_ReturnsFalse()
    {
        // Arrange
        var queue = new CommunicationOperationQueue();
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        var task = new Task<Tensor>(() =>
        {
            Task.Delay(5000).Wait();
            return tensor;
        });
        var handle = new AsyncCommunicationHandle(task);
        queue.Enqueue(handle);

        // Act
        var result = queue.TryWaitForAll(100);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void Dispose_CancelsPendingOperations()
    {
        // Arrange
        var queue = new CommunicationOperationQueue();
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        var task = new Task<Tensor>(() => tensor);
        var handle = new AsyncCommunicationHandle(task);
        queue.Enqueue(handle);

        // Act
        queue.Dispose();

        // Assert
        Assert.True(handle.IsCancelled);
    }
}
