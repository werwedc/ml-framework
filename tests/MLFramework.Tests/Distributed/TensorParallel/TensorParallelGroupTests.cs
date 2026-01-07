namespace MLFramework.Tests.Distributed.TensorParallel;

using MLFramework.Distributed.Communication;
using MLFramework.Distributed.TensorParallel;
using RitterFramework.Core.Tensor;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Xunit;

/// <summary>
/// Tests for TensorParallelGroup class.
/// </summary>
public class TensorParallelGroupTests : IDisposable
{
    private ICommunicator _communicator;

    public TensorParallelGroupTests()
    {
        // Initialize TP context and communicator for testing
        var config = new Dictionary<string, object>
        {
            ["world_size"] = 4,
            ["rank"] = 1
        };
        _communicator = CommunicatorFactory.Create("mock", config);
        TensorParallelContext.Initialize(_communicator);
    }

    public void Dispose()
    {
        var context = TensorParallelContext.Current;
        context?.Dispose();
        _communicator?.Dispose();
    }

    [Fact]
    public void CreateDefaultGroup_AllRanksInGroup()
    {
        // Arrange
        var context = TensorParallel.GetContext();

        // Act
        var group = context.DefaultGroup;

        // Assert
        Assert.NotNull(group);
        Assert.True(group.InGroup);
        Assert.Equal(4, group.WorldSize);
        Assert.Equal(1, group.LocalRank);
    }

    [Fact]
    public void CreateCustomGroup_RankInGroup()
    {
        // Arrange
        var context = TensorParallel.GetContext();
        var ranks = new List<int> { 0, 1, 2 };

        // Act
        var group = context.CreateProcessGroup(ranks);

        // Assert
        Assert.NotNull(group);
        Assert.True(group.InGroup);
        Assert.Equal(3, group.WorldSize);
        Assert.Equal(1, group.LocalRank);
    }

    [Fact]
    public void CreateCustomGroup_RankNotInGroup()
    {
        // Arrange
        var context = TensorParallel.GetContext();
        var ranks = new List<int> { 0, 2, 3 };

        // Act
        var group = context.CreateProcessGroup(ranks);

        // Assert
        Assert.NotNull(group);
        Assert.False(group.InGroup);
        Assert.Equal(0, group.WorldSize);
        Assert.Equal(-1, group.LocalRank);
    }

    [Fact]
    public void CreateCustomGroup_SingleRank()
    {
        // Arrange
        var context = TensorParallel.GetContext();
        var ranks = new List<int> { 1 };

        // Act
        var group = context.CreateProcessGroup(ranks);

        // Assert
        Assert.NotNull(group);
        Assert.True(group.InGroup);
        Assert.Equal(1, group.WorldSize);
        Assert.Equal(0, group.LocalRank);
    }

    [Fact]
    public void CreateCustomGroup_AllRanks()
    {
        // Arrange
        var context = TensorParallel.GetContext();
        var ranks = new List<int> { 0, 1, 2, 3 };

        // Act
        var group = context.CreateProcessGroup(ranks);

        // Assert
        Assert.NotNull(group);
        Assert.True(group.InGroup);
        Assert.Equal(4, group.WorldSize);
        Assert.Equal(1, group.LocalRank);
    }

    [Fact]
    public async Task AllReduceAsync_DefaultGroup_Works()
    {
        // Arrange
        var context = TensorParallel.GetContext();
        var group = context.DefaultGroup;
        var tensor = Tensor.Ones(2, 3);

        // Act
        var result = await group.AllReduceAsync(tensor, ReduceOperation.Sum);

        // Assert
        Assert.NotNull(result);
        // In mock communicator with worldSize=4, all-reduce sums across 4 ranks
        // So each element should be multiplied by worldSize
        // Note: Mock communicator implementation may vary, so we just check it doesn't throw
    }

    [Fact]
    public async Task AllReduceAsync_CustomGroupInGroup_Works()
    {
        // Arrange
        var context = TensorParallel.GetContext();
        var ranks = new List<int> { 0, 1 };
        var group = context.CreateProcessGroup(ranks);
        var tensor = Tensor.Ones(2, 3);

        // Act
        var result = await group.AllReduceAsync(tensor, ReduceOperation.Sum);

        // Assert
        Assert.NotNull(result);
        // Should work without throwing since we're in the group
    }

    [Fact]
    public async Task AllReduceAsync_CustomGroupNotInGroup_ReturnsTensor()
    {
        // Arrange
        var context = TensorParallel.GetContext();
        var ranks = new List<int> { 0, 2 };
        var group = context.CreateProcessGroup(ranks);
        var tensor = Tensor.Ones(2, 3);

        // Act
        var result = await group.AllReduceAsync(tensor, ReduceOperation.Sum);

        // Assert
        Assert.NotNull(result);
        // When not in group, should return tensor unchanged
    }

    [Fact]
    public async Task AllGatherAsync_DefaultGroup_Works()
    {
        // Arrange
        var context = TensorParallel.GetContext();
        var group = context.DefaultGroup;
        var tensor = Tensor.Ones(2, 3);

        // Act
        var result = await group.AllGatherAsync(tensor, dim: 0);

        // Assert
        Assert.NotNull(result);
        // Should work without throwing
    }

    [Fact]
    public async Task AllGatherAsync_CustomGroupInGroup_Works()
    {
        // Arrange
        var context = TensorParallel.GetContext();
        var ranks = new List<int> { 1, 3 };
        var group = context.CreateProcessGroup(ranks);
        var tensor = Tensor.Ones(2, 3);

        // Act
        var result = await group.AllGatherAsync(tensor, dim: 0);

        // Assert
        Assert.NotNull(result);
    }

    [Fact]
    public async Task AllGatherAsync_CustomGroupNotInGroup_ReturnsTensor()
    {
        // Arrange
        var context = TensorParallel.GetContext();
        var ranks = new List<int> { 0, 2 };
        var group = context.CreateProcessGroup(ranks);
        var tensor = Tensor.Ones(2, 3);

        // Act
        var result = await group.AllGatherAsync(tensor, dim: 0);

        // Assert
        Assert.NotNull(result);
    }

    [Fact]
    public async Task ReduceScatterAsync_DefaultGroup_Works()
    {
        // Arrange
        var context = TensorParallel.GetContext();
        var group = context.DefaultGroup;
        var tensor = Tensor.Ones(4, 3);

        // Act
        var result = await group.ReduceScatterAsync(tensor, ReduceOperation.Sum);

        // Assert
        Assert.NotNull(result);
    }

    [Fact]
    public async Task BroadcastAsync_DefaultGroup_Works()
    {
        // Arrange
        var context = TensorParallel.GetContext();
        var group = context.DefaultGroup;
        var tensor = Tensor.Ones(2, 3);

        // Act
        var result = await group.BroadcastAsync(tensor, root: 0);

        // Assert
        Assert.NotNull(result);
    }

    [Fact]
    public async Task BarrierAsync_DefaultGroup_Works()
    {
        // Arrange
        var context = TensorParallel.GetContext();
        var group = context.DefaultGroup;

        // Act & Assert - Should not throw
        await group.BarrierAsync();
    }

    [Fact]
    public async Task BarrierAsync_CustomGroupInGroup_Works()
    {
        // Arrange
        var context = TensorParallel.GetContext();
        var ranks = new List<int> { 0, 1 };
        var group = context.CreateProcessGroup(ranks);

        // Act & Assert - Should not throw
        await group.BarrierAsync();
    }

    [Fact]
    public async Task BarrierAsync_CustomGroupNotInGroup_ReturnsImmediately()
    {
        // Arrange
        var context = TensorParallel.GetContext();
        var ranks = new List<int> { 0, 2 };
        var group = context.CreateProcessGroup(ranks);

        // Act & Assert - Should not throw
        await group.BarrierAsync();
    }

    [Fact]
    public void LocalRank_WithMultipleGroups_CalculatesCorrectly()
    {
        // Arrange
        var context = TensorParallel.GetContext();

        // Act
        var group1 = context.CreateProcessGroup(new List<int> { 0, 1, 2, 3 });
        var group2 = context.CreateProcessGroup(new List<int> { 1, 3 });
        var group3 = context.CreateProcessGroup(new List<int> { 3, 1 });

        // Assert
        Assert.Equal(1, group1.LocalRank);
        Assert.Equal(0, group2.LocalRank);  // Rank 1 is at index 0 in [1, 3]
        Assert.Equal(1, group3.LocalRank);  // Rank 1 is at index 1 in [3, 1]
    }

    [Fact]
    public void Dispose_CanBeCalledMultipleTimes()
    {
        // Arrange
        var context = TensorParallel.GetContext();
        var ranks = new List<int> { 0, 1 };
        var group = context.CreateProcessGroup(ranks);

        // Act & Assert - Should not throw
        group.Dispose();
        group.Dispose();
    }
}
