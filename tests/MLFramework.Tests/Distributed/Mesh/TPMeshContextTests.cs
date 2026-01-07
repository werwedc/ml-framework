using MLFramework.Distributed.Mesh;
using System;
using System.Linq;
using Xunit;

namespace MLFramework.Tests.Distributed.Mesh;

/// <summary>
/// Unit tests for TensorParallelMeshContext functionality.
/// </summary>
public class TPMeshContextTests
{
    [Fact]
    public void InitializeWithMesh_ValidParameters_CreatesContext()
    {
        // Arrange
        var meshShape = new[] { 4, 2 }; // 4 DP, 2 TP
        var rank = 3;

        // Act
        var context = TensorParallelMeshContext.InitializeWithMesh(meshShape, rank);

        // Assert
        Assert.NotNull(context);
        Assert.Equal(2, context.TPWorldSize);
        Assert.Equal(4, context.DPWorldSize);
        Assert.NotNull(context.Mesh);
    }

    [Fact]
    public void InitializeWithMesh_Rank0_CorrectCoordinates()
    {
        // Arrange
        var meshShape = new[] { 2, 3 }; // 2 DP, 3 TP
        var rank = 0;

        // Act
        var context = TensorParallelMeshContext.InitializeWithMesh(meshShape, rank);

        // Assert
        Assert.Equal(0, context.TPRank);
        Assert.Equal(0, context.DPRank);
    }

    [Fact]
    public void InitializeWithMesh_Rank5_CorrectCoordinates()
    {
        // Arrange
        var meshShape = new[] { 2, 3 }; // 2 DP, 3 TP
        var rank = 5;

        // Act
        var context = TensorParallelMeshContext.InitializeWithMesh(meshShape, rank);

        // Assert
        // Rank 5 in [2, 3]: dp_idx = 5 / 3 = 1, tp_idx = 5 % 3 = 2
        Assert.Equal(2, context.TPRank);
        Assert.Equal(1, context.DPRank);
    }

    [Fact]
    public void GetTPProcessGroupRanks_ReturnsCorrectRanks()
    {
        // Arrange
        var meshShape = new[] { 2, 4 }; // 2 DP, 4 TP
        var rank = 2; // Coordinates: [0, 2]
        var context = TensorParallelMeshContext.InitializeWithMesh(meshShape, rank);

        // Act
        var tpRanks = context.GetTPProcessGroupRanks();

        // Assert
        // For DP index 0, TP group should be ranks 0, 1, 2, 3
        Assert.Equal(4, tpRanks.Count);
        Assert.Contains(0, tpRanks);
        Assert.Contains(1, tpRanks);
        Assert.Contains(2, tpRanks);
        Assert.Contains(3, tpRanks);
    }

    [Fact]
    public void GetDPProcessGroupRanks_ReturnsCorrectRanks()
    {
        // Arrange
        var meshShape = new[] { 2, 4 }; // 2 DP, 4 TP
        var rank = 2; // Coordinates: [0, 2]
        var context = TensorParallelMeshContext.InitializeWithMesh(meshShape, rank);

        // Act
        var dpRanks = context.GetDPProcessGroupRanks();

        // Assert
        // For TP index 2, DP group should be ranks 2, 6
        Assert.Equal(2, dpRanks.Count);
        Assert.Contains(2, dpRanks);
        Assert.Contains(6, dpRanks);
    }

    [Fact]
    public void TPBarrierAsync_CompletesSuccessfully()
    {
        // Arrange
        var meshShape = new[] { 2, 2 };
        var rank = 1;
        var context = TensorParallelMeshContext.InitializeWithMesh(meshShape, rank);

        // Act
        var task = context.TPBarrierAsync();

        // Assert
        Assert.NotNull(task);
        Assert.True(task.IsCompleted);
    }

    [Fact]
    public void DPBarrierAsync_CompletesSuccessfully()
    {
        // Arrange
        var meshShape = new[] { 2, 2 };
        var rank = 1;
        var context = TensorParallelMeshContext.InitializeWithMesh(meshShape, rank);

        // Act
        var task = context.DPBarrierAsync();

        // Assert
        Assert.NotNull(task);
        Assert.True(task.IsCompleted);
    }

    [Fact]
    public void MeshShape_AccessibleFromContext()
    {
        // Arrange
        var meshShape = new[] { 4, 2, 2 };
        var rank = 0;
        var context = TensorParallelMeshContext.InitializeWithMesh(meshShape, rank);

        // Act
        var shape = context.Mesh.Shape;

        // Assert
        Assert.Equal(3, shape.Count);
        Assert.Equal(4, shape[0]);
        Assert.Equal(2, shape[1]);
        Assert.Equal(2, shape[2]);
    }

    [Fact]
    public void Dispose_CanBeCalledMultipleTimes()
    {
        // Arrange
        var meshShape = new[] { 2, 2 };
        var rank = 0;
        var context = TensorParallelMeshContext.InitializeWithMesh(meshShape, rank);

        // Act & Assert (should not throw)
        context.Dispose();
        context.Dispose();
    }

    [Fact]
    public void Context_With3DMesh_WorksCorrectly()
    {
        // Arrange
        var meshShape = new[] { 2, 2, 2 }; // DP, TP, PP
        var rank = 5;

        // Act
        var context = TensorParallelMeshContext.InitializeWithMesh(meshShape, rank);

        // Assert
        // Rank 5 in [2, 2, 2]: dp=1, tp=0, pp=1
        Assert.Equal(0, context.TPRank);
        Assert.Equal(1, context.DPRank);
        Assert.NotNull(context.Mesh);
    }

    [Fact]
    public void InitializeWithMesh_MeshAggregatesToWorldSize()
    {
        // Arrange
        var meshShape = new[] { 3, 4, 2 }; // 3*4*2 = 24
        var rank = 10;

        // Act
        var context = TensorParallelMeshContext.InitializeWithMesh(meshShape, rank);

        // Assert
        Assert.Equal(24, context.Mesh.TotalDevices);
        Assert.Equal(24, context.GlobalProcessGroup.WorldSize);
    }
}
