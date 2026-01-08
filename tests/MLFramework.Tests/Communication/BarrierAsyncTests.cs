namespace MLFramework.Communication.Tests;

using MLFramework.Communication;
using MLFramework.Communication.Operations.Async;
using Xunit;

/// <summary>
/// Unit tests for asynchronous barrier operation
/// </summary>
public class BarrierAsyncTests
{
    [Fact]
    public void SynchronizeAsync_WithNullBackend_ThrowsArgumentNullException()
    {
        // Arrange
        IAsyncCommunicationBackend backend = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            BarrierAsync.SynchronizeAsync(backend));
    }

    [Fact]
    public void SynchronizeAsync_WithValidBackend_Succeeds()
    {
        // Arrange
        var backend = new MockAsyncCommunicationBackend(0, 4);

        // Act
        var handle = BarrierAsync.SynchronizeAsync(backend);

        // Assert
        Assert.NotNull(handle);
        Assert.False(handle.IsCompleted);

        // Wait for completion
        handle.Wait();
        Assert.True(handle.IsCompleted);
    }

    [Fact]
    public void SynchronizeAsync_WaitCompletesSuccessfully()
    {
        // Arrange
        var backend = new MockAsyncCommunicationBackend(0, 4);

        // Act
        var handle = BarrierAsync.SynchronizeAsync(backend);
        handle.Wait();

        // Assert
        Assert.True(handle.IsCompleted);
    }

    [Fact]
    public void SynchronizeAsync_TryWaitSucceeds()
    {
        // Arrange
        var backend = new MockAsyncCommunicationBackend(0, 4);

        // Act
        var handle = BarrierAsync.SynchronizeAsync(backend);
        var result = handle.TryWait(1000);

        // Assert
        Assert.True(result);
        Assert.True(handle.IsCompleted);
    }

    [Fact]
    public void SynchronizeAsync_MultipleCallsSucceed()
    {
        // Arrange
        var backend = new MockAsyncCommunicationBackend(0, 4);

        // Act
        var handle1 = BarrierAsync.SynchronizeAsync(backend);
        var handle2 = BarrierAsync.SynchronizeAsync(backend);
        handle1.Wait();
        handle2.Wait();

        // Assert
        Assert.True(handle1.IsCompleted);
        Assert.True(handle2.IsCompleted);
    }
}
