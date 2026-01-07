namespace MLFramework.Tests.Distributed.TensorParallel;

using MLFramework.Distributed.Communication;
using MLFramework.Distributed.TensorParallel;
using System;
using Xunit;

/// <summary>
/// Tests for TensorParallelContext class.
/// </summary>
public class TensorParallelContextTests
{
    [Fact]
    public void Initialize_WithParameters_CreatesValidContext()
    {
        // Arrange & Act
        using var context = TensorParallelContext.Initialize(worldSize: 4, rank: 2, backend: "mock");

        // Assert
        Assert.NotNull(context);
        Assert.Equal(4, context.WorldSize);
        Assert.Equal(2, context.Rank);
        Assert.NotNull(context.Communicator);
    }

    [Fact]
    public void Initialize_WithCommunicator_CreatesValidContext()
    {
        // Arrange
        var config = new System.Collections.Generic.Dictionary<string, object>
        {
            ["world_size"] = 2,
            ["rank"] = 1
        };
        var communicator = CommunicatorFactory.Create("mock", config);

        // Act
        using var context = TensorParallelContext.Initialize(communicator);

        // Assert
        Assert.NotNull(context);
        Assert.Equal(2, context.WorldSize);
        Assert.Equal(1, context.Rank);
        Assert.Same(communicator, context.Communicator);
    }

    [Fact]
    public void GetCurrent_ReturnsActiveContext()
    {
        // Arrange
        using var context1 = TensorParallelContext.Initialize(worldSize: 4, rank: 0);

        // Act
        var current = TensorParallelContext.Current;

        // Assert
        Assert.NotNull(current);
        Assert.Same(context1, current);
    }

    [Fact]
    public void GetCurrent_ReturnsNullAfterDispose()
    {
        // Arrange
        var context = TensorParallelContext.Initialize(worldSize: 4, rank: 0);

        // Act
        context.Dispose();
        var current = TensorParallelContext.Current;

        // Assert
        Assert.Null(current);
    }

    [Fact]
    public void MultipleContexts_SecondReplacesFirst()
    {
        // Arrange
        var context1 = TensorParallelContext.Initialize(worldSize: 4, rank: 0);
        var context2 = TensorParallelContext.Initialize(worldSize: 8, rank: 2);

        // Act
        var current = TensorParallelContext.Current;

        // Assert
        Assert.NotNull(current);
        Assert.Same(context2, current);
        Assert.Equal(8, current.WorldSize);
        Assert.Equal(2, current.Rank);

        // Cleanup
        context2.Dispose();
        context1.Dispose();
    }

    [Fact]
    public void CreateProcessGroup_CreatesValidGroup()
    {
        // Arrange
        using var context = TensorParallelContext.Initialize(worldSize: 4, rank: 1);

        // Act
        var ranks = new System.Collections.Generic.List<int> { 0, 1 };
        var group = context.CreateProcessGroup(ranks);

        // Assert
        Assert.NotNull(group);
        Assert.True(group.InGroup);
        Assert.Equal(2, group.WorldSize);
        Assert.Equal(1, group.LocalRank);
    }

    [Fact]
    public void CreateProcessGroup_RankNotInGroup()
    {
        // Arrange
        using var context = TensorParallelContext.Initialize(worldSize: 4, rank: 3);

        // Act
        var ranks = new System.Collections.Generic.List<int> { 0, 1 };
        var group = context.CreateProcessGroup(ranks);

        // Assert
        Assert.NotNull(group);
        Assert.False(group.InGroup);
        Assert.Equal(0, group.WorldSize);
        Assert.Equal(-1, group.LocalRank);
    }

    [Fact]
    public void DefaultGroup_AlwaysInGroup()
    {
        // Arrange
        using var context = TensorParallelContext.Initialize(worldSize: 4, rank: 2);

        // Act
        var defaultGroup = context.DefaultGroup;

        // Assert
        Assert.NotNull(defaultGroup);
        Assert.True(defaultGroup.InGroup);
        Assert.Equal(4, defaultGroup.WorldSize);
        Assert.Equal(2, defaultGroup.LocalRank);
    }

    [Fact]
    public void Dispose_CleansUpOwnedCommunicator()
    {
        // Arrange
        var context = TensorParallelContext.Initialize(worldSize: 4, rank: 0);
        var communicator = context.Communicator;

        // Act
        context.Dispose();

        // Assert - Communicator should be disposed if owned
        // Note: We can't directly test if communicator is disposed,
        // but we can verify context is cleaned up
        Assert.Null(TensorParallelContext.Current);
    }

    [Fact]
    public void Dispose_DoesNotDisposeUnownedCommunicator()
    {
        // Arrange
        var config = new System.Collections.Generic.Dictionary<string, object>
        {
            ["world_size"] = 2,
            ["rank"] = 0
        };
        var communicator = CommunicatorFactory.Create("mock", config);
        var context = TensorParallelContext.Initialize(communicator);

        // Act
        context.Dispose();

        // Assert - Context should be cleaned up but communicator should not be disposed
        Assert.Null(TensorParallelContext.Current);
        // Note: We can't directly test if communicator is NOT disposed,
        // but the context should not own it
    }

    [Fact]
    public void SequentialContexts_EachValid()
    {
        // Arrange & Act
        using (var context1 = TensorParallelContext.Initialize(worldSize: 2, rank: 0))
        {
            Assert.Equal(2, context1.WorldSize);
            Assert.Equal(0, context1.Rank);
        }

        using (var context2 = TensorParallelContext.Initialize(worldSize: 4, rank: 2))
        {
            Assert.Equal(4, context2.WorldSize);
            Assert.Equal(2, context2.Rank);
        }

        // Assert
        Assert.Null(TensorParallelContext.Current);
    }

    [Fact]
    public void BarrierAsync_WorksOnContext()
    {
        // Arrange
        using var context = TensorParallelContext.Initialize(worldSize: 2, rank: 0);

        // Act & Assert - Should not throw
        var task = context.Communicator.BarrierAsync();
        task.Wait();
    }
}
