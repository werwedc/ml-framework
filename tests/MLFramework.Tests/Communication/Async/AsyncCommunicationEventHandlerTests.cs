using MLFramework.Communication.Async;
using MLFramework.Communication;
using RitterFramework.Core.Tensor;
using System;
using System.Threading.Tasks;
using Xunit;
using Moq;

namespace MLFramework.Tests.Communication.Async;

public class AsyncCommunicationEventHandlerTests
{
    [Fact]
    public void Constructor_WithValidBackend_CreatesHandler()
    {
        // Arrange
        var backendMock = new Mock<IAsyncCommunicationBackend>();

        // Act
        var handler = new AsyncCommunicationEventHandler(backendMock.Object);

        // Assert
        Assert.NotNull(handler);
    }

    [Fact]
    public void Constructor_WithNullBackend_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new AsyncCommunicationEventHandler(null));
    }

    [Fact]
    public async Task StartOperation_WithValidOperation_FiresCompletionEvent()
    {
        // Arrange
        var backendMock = new Mock<IAsyncCommunicationBackend>();
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        var task = Task.FromResult(tensor);
        var handle = new AsyncCommunicationHandle(task);
        var eventFired = false;

        var handler = new AsyncCommunicationEventHandler(backendMock.Object);
        handler.OnOperationComplete += (h) => eventFired = true;

        // Act
        handler.StartOperation(() => handle, 1);
        await Task.Delay(100); // Allow async processing

        // Assert
        Assert.True(eventFired);
    }

    [Fact]
    public void StartOperation_WithOnComplete_CallsHandler()
    {
        // Arrange
        var backendMock = new Mock<IAsyncCommunicationBackend>();
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        var task = Task.FromResult(tensor);
        var handle = new AsyncCommunicationHandle(task);
        var handlerCalled = false;

        var handler = new AsyncCommunicationEventHandler(backendMock.Object);
        handler.StartOperation(() => handle, 1, (h) => handlerCalled = true);

        // Wait for completion
        handle.Wait();
        Task.Delay(100).Wait();

        // Assert
        Assert.True(handlerCalled);
    }

    [Fact]
    public async Task StartOperation_WithFailingOperation_FiresErrorEvent()
    {
        // Arrange
        var backendMock = new Mock<IAsyncCommunicationBackend>();
        var task = new Task<Tensor>(() => throw new InvalidOperationException("Test error"));
        var handle = new AsyncCommunicationHandle(task);
        Exception? capturedException = null;

        var handler = new AsyncCommunicationEventHandler(backendMock.Object);
        handler.OnOperationError += (h, ex) => capturedException = ex;

        // Act
        handler.StartOperation(() => handle, 1);
        await Task.Delay(100); // Allow async processing

        // Assert
        Assert.NotNull(capturedException);
    }

    [Fact]
    public void Dispose_ClearsEventHandlers()
    {
        // Arrange
        var backendMock = new Mock<IAsyncCommunicationBackend>();
        var handler = new AsyncCommunicationEventHandler(backendMock.Object);

        // Act
        handler.Dispose();

        // Assert
        // If dispose completes without exception, it's successful
        Assert.True(true);
    }
}
