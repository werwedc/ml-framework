using MLFramework.Distributed;
using MLFramework.Distributed.Mesh;
using System;
using System.Linq;
using Xunit;

namespace MLFramework.Tests.Distributed.Mesh;

/// <summary>
/// Unit tests for DeviceMesh functionality.
/// </summary>
public class DeviceMeshTests
{
    private MockProcessGroup CreateMockProcessGroup(int worldSize)
    {
        return new MockProcessGroup(worldSize, 0);
    }

    [Fact]
    public void Constructor_ValidShapeAndCoord_CreatesMesh()
    {
        // Arrange
        var shape = new[] { 2, 4 };
        var myCoord = new DeviceMeshCoord(0, 1);
        var processGroup = CreateMockProcessGroup(8);

        // Act
        var mesh = new DeviceMesh(shape, myCoord, processGroup);

        // Assert
        Assert.Equal(2, mesh.Shape.Count);
        Assert.Equal(2, mesh.Shape[0]);
        Assert.Equal(4, mesh.Shape[1]);
        Assert.Equal(8, mesh.TotalDevices);
        Assert.Equal(0, mesh.MyCoordinate[0]);
        Assert.Equal(1, mesh.MyCoordinate[1]);
    }

    [Fact]
    public void Constructor_MismatchedDimensions_ThrowsArgumentException()
    {
        // Arrange
        var shape = new[] { 2, 4 };
        var myCoord = new DeviceMeshCoord(0, 1, 2); // 3D coordinate
        var processGroup = CreateMockProcessGroup(8);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new DeviceMesh(shape, myCoord, processGroup));
    }

    [Fact]
    public void CreateFromRank_ValidRank_CreatesCorrectCoordinate()
    {
        // Arrange
        var shape = new[] { 2, 4 };
        var rank = 5;
        var processGroup = CreateMockProcessGroup(8);

        // Act
        var mesh = DeviceMesh.CreateFromRank(rank, shape, processGroup);

        // Assert
        Assert.Equal(5, rank);
        // Rank 5 in shape [2, 4] should map to coordinate [1, 1]
        // Since 5 / 4 = 1 (row), 5 % 4 = 1 (col)
        Assert.Equal(1, mesh.MyCoordinate[0]);
        Assert.Equal(1, mesh.MyCoordinate[1]);
    }

    [Fact]
    public void RankToCoord_VariousRanks_CorrectlyMapsToCoordinates()
    {
        // Arrange
        var shape = new[] { 2, 3 };

        // Act & Assert
        var coord0 = DeviceMesh.CreateFromRank(0, shape, CreateMockProcessGroup(6)).MyCoordinate;
        Assert.Equal(0, coord0[0]);
        Assert.Equal(0, coord0[1]);

        var coord1 = DeviceMesh.CreateFromRank(1, shape, CreateMockProcessGroup(6)).MyCoordinate;
        Assert.Equal(0, coord0[0]); // Same row
        Assert.Equal(1, coord1[1]); // Next column

        var coord3 = DeviceMesh.CreateFromRank(3, shape, CreateMockProcessGroup(6)).MyCoordinate;
        Assert.Equal(1, coord3[0]); // Next row
        Assert.Equal(0, coord3[1]); // First column
    }

    [Fact]
    public void CoordToRank_VariousCoordinates_CorrectlyMapsToRanks()
    {
        // Arrange
        var shape = new[] { 2, 3 };
        var mesh = new DeviceMesh(shape, new DeviceMeshCoord(0, 0), CreateMockProcessGroup(6));

        // Act & Assert
        Assert.Equal(0, mesh.CoordToRank(new DeviceMeshCoord(0, 0)));
        Assert.Equal(1, mesh.CoordToRank(new DeviceMeshCoord(0, 1)));
        Assert.Equal(2, mesh.CoordToRank(new DeviceMeshCoord(0, 2)));
        Assert.Equal(3, mesh.CoordToRank(new DeviceMeshCoord(1, 0)));
        Assert.Equal(4, mesh.CoordToRank(new DeviceMeshCoord(1, 1)));
        Assert.Equal(5, mesh.CoordToRank(new DeviceMeshCoord(1, 2)));
    }

    [Fact]
    public void CoordToRank_MismatchedDimensions_ThrowsArgumentException()
    {
        // Arrange
        var shape = new[] { 2, 3 };
        var mesh = new DeviceMesh(shape, new DeviceMeshCoord(0, 0), CreateMockProcessGroup(6));
        var coord3d = new DeviceMeshCoord(0, 1, 2);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => mesh.CoordToRank(coord3d));
    }

    [Fact]
    public void GetTPGroupRanks_CorrectlyIdentifiesTPGroup()
    {
        // Arrange
        var shape = new[] { 2, 4 }; // 2 DP groups, 4 TP ranks
        var mesh = new DeviceMesh(shape, new DeviceMeshCoord(0, 1), CreateMockProcessGroup(8));

        // Act
        var tpRanks = mesh.GetTPGroupRanks();

        // Assert
        // For rank at [0, 1], TP group should include all ranks with dp_idx=0
        // These are ranks 0, 1, 2, 3
        Assert.Equal(4, tpRanks.Count);
        Assert.Contains(0, tpRanks);
        Assert.Contains(1, tpRanks);
        Assert.Contains(2, tpRanks);
        Assert.Contains(3, tpRanks);
    }

    [Fact]
    public void GetDPGroupRanks_CorrectlyIdentifiesDPGroup()
    {
        // Arrange
        var shape = new[] { 2, 4 }; // 2 DP groups, 4 TP ranks
        var mesh = new DeviceMesh(shape, new DeviceMeshCoord(0, 1), CreateMockProcessGroup(8));

        // Act
        var dpRanks = mesh.GetDPGroupRanks();

        // Assert
        // For rank at [0, 1], DP group should include all ranks with tp_idx=1
        // These are ranks 1, 5
        Assert.Equal(2, dpRanks.Count);
        Assert.Contains(1, dpRanks);
        Assert.Contains(5, dpRanks);
    }

    [Fact]
    public void GetGroupRanks_InvalidDimension_ThrowsArgumentException()
    {
        // Arrange
        var shape = new[] { 2, 3 };
        var mesh = new DeviceMesh(shape, new DeviceMeshCoord(0, 0), CreateMockProcessGroup(6));

        // Act & Assert
        Assert.Throws<ArgumentException>(() => mesh.GetGroupRanks(ParallelismDimension.Pipeline, 0));
    }

    [Fact]
    public void GetGroupRanks_InvalidIndex_ThrowsArgumentException()
    {
        // Arrange
        var shape = new[] { 2, 3 };
        var mesh = new DeviceMesh(shape, new DeviceMeshCoord(0, 0), CreateMockProcessGroup(6));

        // Act & Assert
        Assert.Throws<ArgumentException>(() => mesh.GetGroupRanks(ParallelismDimension.Data, 5));
    }

    [Fact]
    public void BarrierAsync_CompletesSuccessfully()
    {
        // Arrange
        var shape = new[] { 2, 2 };
        var mesh = new DeviceMesh(shape, new DeviceMeshCoord(0, 0), CreateMockProcessGroup(4));

        // Act & Assert (should not throw)
        var task = mesh.BarrierAsync();
        Assert.True(task.IsCompleted);
    }

    [Fact]
    public void BarrierAsync_WithDimension_CompletesSuccessfully()
    {
        // Arrange
        var shape = new[] { 2, 2 };
        var mesh = new DeviceMesh(shape, new DeviceMeshCoord(0, 0), CreateMockProcessGroup(4));

        // Act & Assert (should not throw)
        var task = mesh.BarrierAsync(ParallelismDimension.Tensor);
        Assert.True(task.IsCompleted);
    }

    [Fact]
    public void Mesh_1DShape_WorksCorrectly()
    {
        // Arrange
        var shape = new[] { 4 };
        var rank = 2;
        var processGroup = CreateMockProcessGroup(4);

        // Act
        var mesh = DeviceMesh.CreateFromRank(rank, shape, processGroup);

        // Assert
        Assert.Single(mesh.Shape);
        Assert.Equal(4, mesh.Shape[0]);
        Assert.Equal(4, mesh.TotalDevices);
        Assert.Equal(2, mesh.MyCoordinate[0]);
    }

    [Fact]
    public void Mesh_3DShape_WorksCorrectly()
    {
        // Arrange
        var shape = new[] { 2, 2, 2 };
        var rank = 5;
        var processGroup = CreateMockProcessGroup(8);

        // Act
        var mesh = DeviceMesh.CreateFromRank(rank, shape, processGroup);

        // Assert
        Assert.Equal(3, mesh.Shape.Count);
        Assert.Equal(2, mesh.Shape[0]);
        Assert.Equal(2, mesh.Shape[1]);
        Assert.Equal(2, mesh.Shape[2]);
        Assert.Equal(8, mesh.TotalDevices);
        // Rank 5 in [2, 2, 2]: 5 / (2*2) = 1, (5 % 4) / 2 = 0, 5 % 2 = 1
        // Coordinate should be [1, 0, 1]
        Assert.Equal(1, mesh.MyCoordinate[0]);
        Assert.Equal(0, mesh.MyCoordinate[1]);
        Assert.Equal(1, mesh.MyCoordinate[2]);
    }

    [Fact]
    public void GetGlobalProcessGroup_ReturnsCorrectGroup()
    {
        // Arrange
        var shape = new[] { 2, 2 };
        var mesh = new DeviceMesh(shape, new DeviceMeshCoord(0, 0), CreateMockProcessGroup(4));

        // Act
        var globalGroup = mesh.GetGlobalProcessGroup();

        // Assert
        Assert.NotNull(globalGroup);
        Assert.Equal(4, globalGroup.WorldSize);
    }
}
