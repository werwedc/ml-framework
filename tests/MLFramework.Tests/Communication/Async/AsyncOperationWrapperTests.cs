using MLFramework.Communication.Async;
using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;
using System;
using System.Threading;
using System.Threading.Tasks;
using Xunit;

namespace MLFramework.Tests.Communication.Async;

public class AsyncOperationWrapperTests
{
    [Fact]
    public async Task ExecuteAsync_WithValidOperation_ReturnsResult()
    {
        // Arrange
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f });
        Func<Task<Tensor>> operation = () => Task.FromResult(tensor);

        // Act
        var result = await AsyncOperationWrapper.ExecuteAsync(operation, 1000);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(3, result.Size);
    }

    [Fact]
    public async Task ExecuteAsync_WithTimeout_ThrowsCommunicationTimeoutException()
    {
        // Arrange
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        Func<Task<Tensor>> operation = async () =>
        {
            await Task.Delay(2000);
            return tensor;
        };

        // Act & Assert
        await Assert.ThrowsAsync<CommunicationTimeoutException>(
            () => AsyncOperationWrapper.ExecuteAsync(operation, 100));
    }

    [Fact]
    public async Task ExecuteAllAsync_WithMultipleOperations_ReturnsAllResults()
    {
        // Arrange
        var tensor1 = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        var tensor2 = Tensor.FromArray(new float[] { 3.0f, 4.0f });
        var operations = new Func<Task<Tensor>>[]
        {
            () => Task.FromResult(tensor1),
            () => Task.FromResult(tensor2)
        };

        // Act
        var results = await AsyncOperationWrapper.ExecuteAllAsync(operations);

        // Assert
        Assert.Equal(2, results.Count);
        Assert.Equal(2, results[0].Size);
        Assert.Equal(2, results[1].Size);
    }

    [Fact]
    public async Task ExecuteAllAsync_WithTimeout_ThrowsCommunicationTimeoutException()
    {
        // Arrange
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        var operations = new Func<Task<Tensor>>[]
        {
            async () => { await Task.Delay(2000); return tensor; },
            async () => { await Task.Delay(2000); return tensor; }
        };

        // Act & Assert
        await Assert.ThrowsAsync<CommunicationTimeoutException>(
            () => AsyncOperationWrapper.ExecuteAllAsync(operations, 100));
    }
}

public class TaskExtensionsTests
{
    [Fact]
    public async Task WithCancellation_WithoutCancellation_Completes()
    {
        // Arrange
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        var task = Task.FromResult(tensor);
        var cts = new CancellationTokenSource();

        // Act
        var result = await task.WithCancellation(cts.Token);

        // Assert
        Assert.NotNull(result);
    }

    [Fact]
    public async Task WithCancellation_WithCancellation_ThrowsOperationCanceledException()
    {
        // Arrange
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        var task = new Task<Tensor>(() => tensor);
        var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(
            () => task.WithCancellation(cts.Token));
    }
}
