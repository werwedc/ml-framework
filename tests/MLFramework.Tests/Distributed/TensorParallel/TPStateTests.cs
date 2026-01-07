namespace MLFramework.Tests.Distributed.TensorParallel;

using MLFramework.Distributed.TensorParallel;
using System.Threading.Tasks;
using Xunit;

/// <summary>
/// Tests for TPState helper class.
/// </summary>
public class TPStateTests : IDisposable
{
    private bool _rank0Executed;
    private bool _rank1Executed;
    private bool _rank2Executed;
    private bool _rank3Executed;

    public TPStateTests()
    {
        // Initialize TP context before each test
        TensorParallelContext.Initialize(worldSize: 4, rank: 2, backend: "mock");
    }

    public void Dispose()
    {
        // Clean up context after each test
        var context = TensorParallelContext.Current;
        context?.Dispose();

        // Reset flags
        _rank0Executed = false;
        _rank1Executed = false;
        _rank2Executed = false;
        _rank3Executed = false;
    }

    [Fact]
    public async Task ExecuteOnRankAsync_TargetRankMatch_Executes()
    {
        // Arrange
        var executed = false;
        async Task Action()
        {
            executed = true;
        }

        // Act - Rank is 2, so this should execute
        await TPState.ExecuteOnRankAsync(2, Action);

        // Assert
        Assert.True(executed);
    }

    [Fact]
    public async Task ExecuteOnRankAsync_TargetRankMismatch_DoesNotExecute()
    {
        // Arrange
        var executed = false;
        async Task Action()
        {
            executed = true;
        }

        // Act - Rank is 2, so this should NOT execute
        await TPState.ExecuteOnRankAsync(1, Action);

        // Assert
        Assert.False(executed);
    }

    [Fact]
    public void ExecuteOnRank_TargetRankMatch_Executes()
    {
        // Arrange
        var executed = false;
        void Action()
        {
            executed = true;
        }

        // Act - Rank is 2, so this should execute
        TPState.ExecuteOnRank(2, Action);

        // Assert
        Assert.True(executed);
    }

    [Fact]
    public void ExecuteOnRank_TargetRankMismatch_DoesNotExecute()
    {
        // Arrange
        var executed = false;
        void Action()
        {
            executed = true;
        }

        // Act - Rank is 2, so this should NOT execute
        TPState.ExecuteOnRank(1, Action);

        // Assert
        Assert.False(executed);
    }

    [Fact]
    public async Task ExecuteOnMasterAsync_OnRank2_DoesNotExecute()
    {
        // Arrange
        var executed = false;
        async Task Action()
        {
            executed = true;
        }

        // Act - Rank is 2, not master (0), so should NOT execute
        await TPState.ExecuteOnMasterAsync(Action);

        // Assert
        Assert.False(executed);
    }

    [Fact]
    public async Task ExecuteOnMasterAsync_OnRank0_Executes()
    {
        // Arrange - Reinitialize with rank 0
        var context = TensorParallelContext.Current;
        context?.Dispose();
        TensorParallelContext.Initialize(worldSize: 4, rank: 0, backend: "mock");

        var executed = false;
        async Task Action()
        {
            executed = true;
        }

        // Act - Rank is 0 (master), so should execute
        await TPState.ExecuteOnMasterAsync(Action);

        // Assert
        Assert.True(executed);
    }

    [Fact]
    public void ExecuteOnMaster_OnRank2_DoesNotExecute()
    {
        // Arrange
        var executed = false;
        void Action()
        {
            executed = true;
        }

        // Act - Rank is 2, not master (0), so should NOT execute
        TPState.ExecuteOnMaster(Action);

        // Assert
        Assert.False(executed);
    }

    [Fact]
    public void ExecuteOnMaster_OnRank0_Executes()
    {
        // Arrange - Reinitialize with rank 0
        var context = TensorParallelContext.Current;
        context?.Dispose();
        TensorParallelContext.Initialize(worldSize: 4, rank: 0, backend: "mock");

        var executed = false;
        void Action()
        {
            executed = true;
        }

        // Act - Rank is 0 (master), so should execute
        TPState.ExecuteOnMaster(Action);

        // Assert
        Assert.True(executed);
    }

    [Fact]
    public async Task ExecuteOnAllAsync_ExecutesOnCurrentRank()
    {
        // Arrange
        var executed = false;
        async Task Action()
        {
            executed = true;
        }

        // Act
        await TPState.ExecuteOnAllAsync(Action);

        // Assert - Should execute on current rank (2)
        Assert.True(executed);
    }

    [Fact]
    public void ExecuteOnAll_ExecutesOnCurrentRank()
    {
        // Arrange
        var executed = false;
        void Action()
        {
            executed = true;
        }

        // Act
        TPState.ExecuteOnAll(Action);

        // Assert - Should execute on current rank (2)
        Assert.True(executed);
    }

    [Fact]
    public void IsMaster_OnRank2_ReturnsFalse()
    {
        // Act
        var result = TPState.IsMaster;

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void IsMaster_OnRank0_ReturnsTrue()
    {
        // Arrange - Reinitialize with rank 0
        var context = TensorParallelContext.Current;
        context?.Dispose();
        TensorParallelContext.Initialize(worldSize: 4, rank: 0, backend: "mock");

        // Act
        var result = TPState.IsMaster;

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void IsLastRank_OnRank2_ReturnsFalse()
    {
        // Act
        var result = TPState.IsLastRank;

        // Assert - World size is 4, last rank is 3
        Assert.False(result);
    }

    [Fact]
    public void IsLastRank_OnRank3_ReturnsTrue()
    {
        // Arrange - Reinitialize with rank 3
        var context = TensorParallelContext.Current;
        context?.Dispose();
        TensorParallelContext.Initialize(worldSize: 4, rank: 3, backend: "mock");

        // Act
        var result = TPState.IsLastRank;

        // Assert
        Assert.True(result);
    }

    [Fact]
    public async Task ExecuteOnRankAsync_DifferentRanks()
    {
        // Arrange
        var executionLog = new System.Collections.Generic.List<int>();

        // Act - Only rank 2 should execute
        for (int i = 0; i < 4; i++)
        {
            await TPState.ExecuteOnRankAsync(i, async () =>
            {
                executionLog.Add(TensorParallel.GetRank());
            });
        }

        // Assert
        Assert.Single(executionLog);
        Assert.Contains(2, executionLog);
    }

    [Fact]
    public async Task ExecuteOnMasterAsync_MultipleCalls()
    {
        // Arrange
        var executionCount = 0;
        async Task Action()
        {
            executionCount++;
        }

        // Act - Should only execute on master (not on rank 2)
        await TPState.ExecuteOnMasterAsync(Action);
        await TPState.ExecuteOnMasterAsync(Action);
        await TPState.ExecuteOnMasterAsync(Action);

        // Assert
        Assert.Equal(0, executionCount);
    }

    [Fact]
    public async Task ExecuteOnAllAsync_MultipleCalls()
    {
        // Arrange
        var executionCount = 0;
        async Task Action()
        {
            executionCount++;
        }

        // Act - Should execute on current rank
        await TPState.ExecuteOnAllAsync(Action);
        await TPState.ExecuteOnAllAsync(Action);
        await TPState.ExecuteOnAllAsync(Action);

        // Assert
        Assert.Equal(3, executionCount);
    }
}
