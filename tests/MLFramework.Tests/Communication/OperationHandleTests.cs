namespace MLFramework.Communication.Tests;

using MLFramework.Communication.Operations;
using RitterFramework.Core.Tensor;
using Xunit;

/// <summary>
/// Unit tests for operation handles
/// </summary>
public class OperationHandleTests
{
    [Fact]
    public void CompletedOperationHandle_IsCompleted_ReturnsTrue()
    {
        // Arrange
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });
        var handle = new CompletedOperationHandle(tensor);

        // Act
        var isCompleted = handle.IsCompleted;

        // Assert
        Assert.True(isCompleted);
    }

    [Fact]
    public void CompletedOperationHandle_Wait_DoesNotThrow()
    {
        // Arrange
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });
        var handle = new CompletedOperationHandle(tensor);

        // Act & Assert - should not throw
        var exception = Record.Exception(() => handle.Wait());
        Assert.Null(exception);
    }

    [Fact]
    public void CompletedOperationHandle_TryWait_AlwaysReturnsTrue()
    {
        // Arrange
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });
        var handle = new CompletedOperationHandle(tensor);

        // Act
        var completed = handle.TryWait(1000);

        // Assert
        Assert.True(completed);
    }

    [Fact]
    public void CompletedOperationHandle_GetResult_ReturnsTensor()
    {
        // Arrange
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });
        var handle = new CompletedOperationHandle(tensor);

        // Act
        var result = handle.GetResult();

        // Assert
        Assert.NotNull(result);
        Assert.Equal(tensor.Data, result.Data);
    }

    [Fact]
    public void CompletedOperationHandle_WithNullTensor_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => 
            new CompletedOperationHandle(null));
    }

    [Fact]
    public void PendingOperationHandle_WithNullTask_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => 
            new PendingOperationHandle(null));
    }

    [Fact]
    public void PendingOperationHandle_GetResult_BeforeWait_ThrowsInvalidOperationException()
    {
        // Arrange
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });
        var task = System.Threading.Tasks.Task.Run(() => tensor);
        var handle = new PendingOperationHandle(task);

        // Wait for task to complete without using the handle
        task.Wait();

        // Act & Assert
        // After waiting, it should succeed
        var result = handle.GetResult();
        Assert.NotNull(result);
    }

    [Fact]
    public void PendingOperationHandle_Wait_Succeeds()
    {
        // Arrange
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });
        var task = System.Threading.Tasks.Task.Run(() => tensor);
        var handle = new PendingOperationHandle(task);

        // Act & Assert - should not throw
        var exception = Record.Exception(() => handle.Wait());
        Assert.Null(exception);
    }
}
