using Xunit;
using MachineLearning.Visualization.Events;

namespace MLFramework.Visualization.Tests.Events;

/// <summary>
/// Unit tests for EventFactory
/// </summary>
public class EventFactoryTests
{
    [Fact]
    public void GlobalStep_InitiallyZero()
    {
        // Act & Assert
        Assert.Equal(0, EventFactory.GlobalStep);
    }

    [Fact]
    public void GlobalStep_WhenSet_ReturnsSetValue()
    {
        // Arrange
        EventFactory.GlobalStep = 10;

        // Act & Assert
        Assert.Equal(10, EventFactory.GlobalStep);
        EventFactory.GlobalStep = 0; // Reset
    }

    [Fact]
    public void NextStep_WhenCalled_IncrementsStep()
    {
        // Arrange
        EventFactory.GlobalStep = 0;

        // Act
        var step1 = EventFactory.NextStep();
        var step2 = EventFactory.NextStep();

        // Assert
        Assert.Equal(1, step1);
        Assert.Equal(2, step2);
        EventFactory.GlobalStep = 0; // Reset
    }

    [Fact]
    public void CreateScalar_WithExplicitStep_UsesProvidedStep()
    {
        // Arrange
        EventFactory.GlobalStep = 0;

        // Act
        var evt = EventFactory.CreateScalar("test", 1.0f, step: 100);

        // Assert
        Assert.Equal(100, evt.Step);
        EventFactory.GlobalStep = 0; // Reset
    }

    [Fact]
    public void CreateScalar_WithoutStep_UsesAutoIncrement()
    {
        // Arrange
        EventFactory.GlobalStep = 0;

        // Act
        var evt1 = EventFactory.CreateScalar("test", 1.0f);
        var evt2 = EventFactory.CreateScalar("test", 1.0f);

        // Assert
        Assert.Equal(1, evt1.Step);
        Assert.Equal(2, evt2.Step);
        EventFactory.GlobalStep = 0; // Reset
    }

    [Fact]
    public void CreateScalar_WithParameters_CreatesCorrectEvent()
    {
        // Act
        var evt = EventFactory.CreateScalar("test_metric", 2.5f, step: 5);

        // Assert
        Assert.Equal("test_metric", evt.Name);
        Assert.Equal(2.5f, evt.Value);
        Assert.Equal(5, evt.Step);
        EventFactory.GlobalStep = 0; // Reset
    }

    [Fact]
    public void CreateHistogram_WithParameters_CreatesCorrectEvent()
    {
        // Arrange
        var values = new float[] { 1.0f, 2.0f, 3.0f };

        // Act
        var evt = EventFactory.CreateHistogram("test_hist", values, step: 10, binCount: 50, useLogScale: true);

        // Assert
        Assert.Equal("test_hist", evt.Name);
        Assert.Equal(values, evt.Values);
        Assert.Equal(10, evt.Step);
        Assert.Equal(50, evt.BinCount);
        Assert.True(evt.UseLogScale);
        EventFactory.GlobalStep = 0; // Reset
    }

    [Fact]
    public void CreateProfilingStart_WithParameters_CreatesCorrectEvent()
    {
        // Act
        var evt = EventFactory.CreateProfilingStart("test_op", step: 15);

        // Assert
        Assert.Equal("test_op", evt.Name);
        Assert.Equal(15, evt.Step);
        Assert.Equal(Environment.CurrentManagedThreadId, evt.ThreadId);
        EventFactory.GlobalStep = 0; // Reset
    }

    [Fact]
    public void CreateProfilingEnd_WithParameters_CreatesCorrectEvent()
    {
        // Act
        var evt = EventFactory.CreateProfilingEnd("test_op", step: 20, durationNanoseconds: 1000000);

        // Assert
        Assert.Equal("test_op", evt.Name);
        Assert.Equal(20, evt.Step);
        Assert.Equal(1000000, evt.DurationNanoseconds);
        EventFactory.GlobalStep = 0; // Reset
    }

    [Fact]
    public void CreateProfilingPair_WithAction_CreatesBothEvents()
    {
        // Arrange
        var actionExecuted = false;

        // Act
        var (startEvent, endEvent) = EventFactory.CreateProfilingPair(
            "test_op",
            () => actionExecuted = true,
            step: 25);

        // Assert
        Assert.NotNull(startEvent);
        Assert.NotNull(endEvent);
        Assert.Equal("test_op", startEvent.Name);
        Assert.Equal("test_op", endEvent.Name);
        Assert.Equal(25, startEvent.Step);
        Assert.Equal(25, endEvent.Step);
        Assert.True(actionExecuted);
        Assert.True(endEvent.DurationNanoseconds > 0);
        EventFactory.GlobalStep = 0; // Reset
    }

    [Fact]
    public async Task CreateProfilingPairAsync_WithAsyncAction_CreatesBothEvents()
    {
        // Arrange
        var actionExecuted = false;

        // Act
        var (startEvent, endEvent) = await EventFactory.CreateProfilingPairAsync(
            "test_op",
            async () =>
            {
                await Task.Delay(10);
                actionExecuted = true;
            },
            step: 30);

        // Assert
        Assert.NotNull(startEvent);
        Assert.NotNull(endEvent);
        Assert.Equal("test_op", startEvent.Name);
        Assert.Equal("test_op", endEvent.Name);
        Assert.Equal(30, startEvent.Step);
        Assert.Equal(30, endEvent.Step);
        Assert.True(actionExecuted);
        Assert.True(endEvent.DurationNanoseconds > 0);
        EventFactory.GlobalStep = 0; // Reset
    }

    [Fact]
    public void CreateProfilingPair_WithThrowingAction_StillCreatesBothEvents()
    {
        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
        {
            var (startEvent, endEvent) = EventFactory.CreateProfilingPair(
                "test_op",
                () => throw new InvalidOperationException("Test exception"),
                step: 35);

            Assert.NotNull(startEvent);
            Assert.NotNull(endEvent);
            Assert.Equal("test_op", startEvent.Name);
            Assert.Equal("test_op", endEvent.Name);
            Assert.True(endEvent.DurationNanoseconds >= 0);
        });
        EventFactory.GlobalStep = 0; // Reset
    }

    [Fact]
    public void CreateMemoryAllocation_WithParameters_CreatesCorrectEvent()
    {
        // Act
        var evt = EventFactory.CreateMemoryAllocation("tensor1", 1024, step: 40, location: "GPU");

        // Assert
        Assert.Equal("tensor1", evt.Name);
        Assert.Equal(1024, evt.SizeBytes);
        Assert.Equal(40, evt.Step);
        Assert.Equal("GPU", evt.Location);
        EventFactory.GlobalStep = 0; // Reset
    }

    [Fact]
    public void CreateTensorOperation_WithParameters_CreatesCorrectEvent()
    {
        // Arrange
        var inputShapes = new int[][] { new int[] { 3, 224, 224 } };
        var outputShape = new int[] { 512 };

        // Act
        var evt = EventFactory.CreateTensorOperation(
            "Conv2D",
            inputShapes,
            outputShape,
            step: 45,
            durationNanoseconds: 5000000);

        // Assert
        Assert.Equal("Conv2D", evt.OperationName);
        Assert.Equal(inputShapes, evt.InputShapes);
        Assert.Equal(outputShape, evt.OutputShape);
        Assert.Equal(45, evt.Step);
        Assert.Equal(5000000, evt.DurationNanoseconds);
        EventFactory.GlobalStep = 0; // Reset
    }

    [Fact]
    public void CreateScalar_WithNullName_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => EventFactory.CreateScalar(null!, 1.0f));
    }

    [Fact]
    public void CreateHistogram_WithNullName_ThrowsArgumentNullException()
    {
        // Arrange
        var values = new float[] { 1.0f, 2.0f };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => EventFactory.CreateHistogram(null!, values));
    }

    [Fact]
    public void CreateProfilingPair_WithNullAction_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => EventFactory.CreateProfilingPair("test", null!));
    }
}
