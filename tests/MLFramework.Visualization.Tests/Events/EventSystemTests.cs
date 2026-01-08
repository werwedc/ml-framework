using Xunit;
using MachineLearning.Visualization.Events;

namespace MLFramework.Visualization.Tests.Events;

/// <summary>
/// Unit tests for EventSystem
/// </summary>
public class EventSystemTests
{
    [Fact]
    public void Constructor_WithDefaults_CreatesEventSystem()
    {
        // Act
        var eventSystem = new EventSystem();

        // Assert
        Assert.True(eventSystem.IsRunning);
        eventSystem.Dispose();
    }

    [Fact]
    public void Constructor_WithAsyncDisabled_CreatesEventSystem()
    {
        // Act
        var eventSystem = new EventSystem(enableAsync: false);

        // Assert
        Assert.True(eventSystem.IsRunning);
        eventSystem.Dispose();
    }

    [Fact]
    public void Subscribe_WithValidHandler_AddsSubscriber()
    {
        // Arrange
        var eventSystem = new EventSystem();
        var callCount = 0;
        Action<ScalarMetricEvent> handler = (e) => callCount++;

        // Act
        eventSystem.Subscribe(handler);

        // Assert
        Assert.True(eventSystem.IsRunning);
        eventSystem.Dispose();
    }

    [Fact]
    public void Publish_WithSubscribers_NotifiesAllSubscribers()
    {
        // Arrange
        var eventSystem = new EventSystem();
        var callCount1 = 0;
        var callCount2 = 0;
        Action<ScalarMetricEvent> handler1 = (e) => callCount1++;
        Action<ScalarMetricEvent> handler2 = (e) => callCount2++;
        eventSystem.Subscribe(handler1);
        eventSystem.Subscribe(handler2);

        // Act
        var testEvent = new ScalarMetricEvent("test", 1.0f);
        eventSystem.Publish(testEvent);

        // Assert
        Assert.Equal(1, callCount1);
        Assert.Equal(1, callCount2);
        eventSystem.Dispose();
    }

    [Fact]
    public async Task PublishAsync_WithSubscribers_NotifiesAllSubscribers()
    {
        // Arrange
        var eventSystem = new EventSystem();
        var callCount1 = 0;
        var callCount2 = 0;
        Action<ScalarMetricEvent> handler1 = (e) => callCount1++;
        Action<ScalarMetricEvent> handler2 = (e) => callCount2++;
        eventSystem.Subscribe(handler1);
        eventSystem.Subscribe(handler2);

        // Act
        var testEvent = new ScalarMetricEvent("test", 1.0f);
        await eventSystem.PublishAsync(testEvent);

        // Assert
        Assert.Equal(1, callCount1);
        Assert.Equal(1, callCount2);
        eventSystem.Dispose();
    }

    [Fact]
    public void Publish_WithMultipleSubscribersOfDifferentTypes_OnlyNotifiesRelevantSubscribers()
    {
        // Arrange
        var eventSystem = new EventSystem();
        var scalarCount = 0;
        var histogramCount = 0;
        Action<ScalarMetricEvent> scalarHandler = (e) => scalarCount++;
        Action<HistogramEvent> histogramHandler = (e) => histogramCount++;
        eventSystem.Subscribe(scalarHandler);
        eventSystem.Subscribe(histogramHandler);

        // Act
        var scalarEvent = new ScalarMetricEvent("test", 1.0f);
        eventSystem.Publish(scalarEvent);

        // Assert
        Assert.Equal(1, scalarCount);
        Assert.Equal(0, histogramCount);
        eventSystem.Dispose();
    }

    [Fact]
    public void Unsubscribe_WithExistingHandler_RemovesSubscriber()
    {
        // Arrange
        var eventSystem = new EventSystem();
        var callCount = 0;
        Action<ScalarMetricEvent> handler = (e) => callCount++;
        eventSystem.Subscribe(handler);

        // Act
        eventSystem.Unsubscribe(handler);
        var testEvent = new ScalarMetricEvent("test", 1.0f);
        eventSystem.Publish(testEvent);

        // Assert
        Assert.Equal(0, callCount);
        eventSystem.Dispose();
    }

    [Fact]
    public void SubscribeAll_WithValidHandler_ReceivesAllEventTypes()
    {
        // Arrange
        var eventSystem = new EventSystem();
        var eventCount = 0;
        Action<Event> handler = (e) => eventCount++;
        eventSystem.SubscribeAll(handler);

        // Act
        eventSystem.Publish(new ScalarMetricEvent("test1", 1.0f));
        eventSystem.Publish(new HistogramEvent("test2", new float[] { 1, 2, 3 }));
        eventSystem.Publish(new ProfilingStartEvent("test3"));

        // Assert
        Assert.Equal(3, eventCount);
        eventSystem.Dispose();
    }

    [Fact]
    public void Publish_WithHandlerThatThrowsException_DoesNotAffectOtherHandlers()
    {
        // Arrange
        var eventSystem = new EventSystem();
        var callCount = 0;
        Action<ScalarMetricEvent> throwingHandler = (e) => throw new Exception("Test exception");
        Action<ScalarMetricEvent> normalHandler = (e) => callCount++;
        eventSystem.Subscribe(throwingHandler);
        eventSystem.Subscribe(normalHandler);

        // Act
        var testEvent = new ScalarMetricEvent("test", 1.0f);
        eventSystem.Publish(testEvent);

        // Assert
        Assert.Equal(1, callCount);
        eventSystem.Dispose();
    }

    [Fact]
    public void Dispose_WhenCalled_PreventsFurtherPublishing()
    {
        // Arrange
        var eventSystem = new EventSystem();
        var callCount = 0;
        Action<ScalarMetricEvent> handler = (e) => callCount++;
        eventSystem.Subscribe(handler);

        // Act
        eventSystem.Dispose();
        var testEvent = new ScalarMetricEvent("test", 1.0f);
        eventSystem.Publish(testEvent);

        // Assert
        Assert.Equal(0, callCount);
    }

    [Fact]
    public void Subscribe_WhenSystemDisposed_ThrowsObjectDisposedException()
    {
        // Arrange
        var eventSystem = new EventSystem();
        eventSystem.Dispose();

        // Act & Assert
        Assert.Throws<ObjectDisposedException>(() =>
        {
            Action<ScalarMetricEvent> handler = (e) => { };
            eventSystem.Subscribe(handler);
        });
    }

    [Fact]
    public void Shutdown_WhenCalled_DisposesEventSystem()
    {
        // Arrange
        var eventSystem = new EventSystem();

        // Act
        eventSystem.Shutdown();

        // Assert
        Assert.False(eventSystem.IsRunning);
    }

    [Fact]
    public void Publish_AfterShutdown_DoesNotNotifySubscribers()
    {
        // Arrange
        var eventSystem = new EventSystem();
        var callCount = 0;
        Action<ScalarMetricEvent> handler = (e) => callCount++;
        eventSystem.Subscribe(handler);

        // Act
        eventSystem.Shutdown();
        var testEvent = new ScalarMetricEvent("test", 1.0f);
        eventSystem.Publish(testEvent);

        // Assert
        Assert.Equal(0, callCount);
    }
}
