namespace MLFramework.Tests.Distributed.TensorParallel;

using MLFramework.Distributed.TensorParallel;
using System;
using Xunit;

/// <summary>
/// Tests for TensorParallel static API wrapper.
/// </summary>
public class TensorParallelTests : IDisposable
{
    public TensorParallelTests()
    {
        // Initialize TP context before each test
        TensorParallelContext.Initialize(worldSize: 4, rank: 2, backend: "mock");
    }

    public void Dispose()
    {
        // Clean up context after each test
        var context = TensorParallelContext.Current;
        context?.Dispose();
    }

    [Fact]
    public void Initialize_WithParameters_ReturnsValidContext()
    {
        // Clean up existing context
        var existing = TensorParallelContext.Current;
        existing?.Dispose();

        // Act
        var context = TensorParallel.Initialize(worldSize: 8, rank: 3, backend: "mock");

        // Assert
        Assert.NotNull(context);
        Assert.Equal(8, context.WorldSize);
        Assert.Equal(3, context.Rank);

        // Cleanup
        context.Dispose();
    }

    [Fact]
    public void GetContext_WhenInitialized_ReturnsContext()
    {
        // Act
        var context = TensorParallel.GetContext();

        // Assert
        Assert.NotNull(context);
        Assert.Equal(4, context.WorldSize);
        Assert.Equal(2, context.Rank);
    }

    [Fact]
    public void GetContext_WhenNotInitialized_ThrowsException()
    {
        // Arrange
        var context = TensorParallelContext.Current;
        context?.Dispose();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => TensorParallel.GetContext());

        // Reinitialize for cleanup
        TensorParallel.Initialize(4, 2, "mock");
    }

    [Fact]
    public void TryGetContext_WhenInitialized_ReturnsContext()
    {
        // Act
        var context = TensorParallel.TryGetContext();

        // Assert
        Assert.NotNull(context);
        Assert.Equal(4, context.WorldSize);
        Assert.Equal(2, context.Rank);
    }

    [Fact]
    public void TryGetContext_WhenNotInitialized_ReturnsNull()
    {
        // Arrange
        var context = TensorParallelContext.Current;
        context?.Dispose();

        // Act
        var result = TensorParallel.TryGetContext();

        // Assert
        Assert.Null(result);

        // Reinitialize for cleanup
        TensorParallel.Initialize(4, 2, "mock");
    }

    [Fact]
    public void IsInitialized_WhenContextExists_ReturnsTrue()
    {
        // Act
        var result = TensorParallel.IsInitialized;

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void IsInitialized_WhenNoContext_ReturnsFalse()
    {
        // Arrange
        var context = TensorParallelContext.Current;
        context?.Dispose();

        // Act
        var result = TensorParallel.IsInitialized;

        // Assert
        Assert.False(result);

        // Reinitialize for cleanup
        TensorParallel.Initialize(4, 2, "mock");
    }

    [Fact]
    public void GetWorldSize_ReturnsCorrectValue()
    {
        // Act
        var worldSize = TensorParallel.GetWorldSize();

        // Assert
        Assert.Equal(4, worldSize);
    }

    [Fact]
    public void GetRank_ReturnsCorrectValue()
    {
        // Act
        var rank = TensorParallel.GetRank();

        // Assert
        Assert.Equal(2, rank);
    }

    [Fact]
    public void GetCommunicator_ReturnsCommunicator()
    {
        // Act
        var communicator = TensorParallel.GetCommunicator();

        // Assert
        Assert.NotNull(communicator);
        Assert.Equal(4, communicator.WorldSize);
        Assert.Equal(2, communicator.Rank);
    }

    [Fact]
    public void GetContext_WhenNotInitialized_ThrowsCorrectMessage()
    {
        // Arrange
        var context = TensorParallelContext.Current;
        context?.Dispose();

        // Act & Assert
        var ex = Assert.Throws<InvalidOperationException>(() => TensorParallel.GetContext());
        Assert.Contains("TensorParallel context not initialized", ex.Message);

        // Reinitialize for cleanup
        TensorParallel.Initialize(4, 2, "mock");
    }

    [Fact]
    public void MultipleGetContextCalls_ReturnSameInstance()
    {
        // Act
        var context1 = TensorParallel.GetContext();
        var context2 = TensorParallel.GetContext();

        // Assert
        Assert.Same(context1, context2);
    }

    [Fact]
    public void GetWorldSize_AfterReinitialization_ReturnsNewValue()
    {
        // Arrange
        var context1 = TensorParallelContext.Current;
        context1?.Dispose();

        // Act
        TensorParallel.Initialize(worldSize: 8, rank: 1, backend: "mock");
        var worldSize = TensorParallel.GetWorldSize();

        // Assert
        Assert.Equal(8, worldSize);
    }

    [Fact]
    public void GetRank_AfterReinitialization_ReturnsNewValue()
    {
        // Arrange
        var context1 = TensorParallelContext.Current;
        context1?.Dispose();

        // Act
        TensorParallel.Initialize(worldSize: 8, rank: 5, backend: "mock");
        var rank = TensorParallel.GetRank();

        // Assert
        Assert.Equal(5, rank);
    }
}
