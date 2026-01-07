using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Tests.Integration;
using MLFramework.Distributed;
using System.Linq;

namespace MLFramework.Tests.Integration
{
    [TestClass]
    public class TPDeviceMeshTests
    {
        [TestMethod]
        public void DeviceMesh_2DMesh_CreatesCorrectProcessGroups()
        {
            // Arrange
            int[] meshShape = new[] { 2, 2 }; // 2 DP groups x 2 TP ranks
            int totalDevices = meshShape.Aggregate(1, (a, b) => a * b);

            for (int rank = 0; rank < totalDevices; rank++)
            {
                // Act
                var processGroup = MockProcessGroup.Create(totalDevices, rank);
                var mesh = DeviceMesh.CreateFromRank(rank, meshShape, processGroup);

                var coord = mesh.MyCoordinate;

                // Assert
                Assert.IsNotNull(coord);
                Assert.AreEqual(2, coord.Dimensions);
                Assert.IsTrue(coord[0] < 2, "DP coordinate should be < 2");
                Assert.IsTrue(coord[1] < 2, "TP coordinate should be < 2");

                processGroup.Destroy();
            }
        }

        [TestMethod]
        public void DeviceMesh_TPGroups_AllRanksInCorrectGroups()
        {
            // Arrange
            int[] meshShape = new[] { 2, 2 }; // 2 DP groups x 2 TP ranks
            var meshes = new DeviceMesh[4];
            var processGroups = new ProcessGroup[4];

            for (int rank = 0; rank < 4; rank++)
            {
                processGroups[rank] = MockProcessGroup.Create(4, rank);
                meshes[rank] = DeviceMesh.CreateFromRank(rank, meshShape, processGroups[rank]);
            }

            // Act & Assert
            // All ranks with same DP coordinate should be in same TP group
            var tpGroups = meshes.Select(m => m.GetTPGroup()).ToList();
            Assert.AreEqual(2, tpGroups.Distinct().Count(), "Should have 2 TP groups");

            // Cleanup
            foreach (var pg in processGroups)
            {
                pg.Destroy();
            }
        }

        [TestMethod]
        public void DeviceMesh_1DMesh_SingleDimension()
        {
            // Arrange
            int[] meshShape = new[] { 4 }; // Single dimension, 4 devices
            int totalDevices = meshShape.Aggregate(1, (a, b) => a * b);

            for (int rank = 0; rank < totalDevices; rank++)
            {
                // Act
                var processGroup = MockProcessGroup.Create(totalDevices, rank);
                var mesh = DeviceMesh.CreateFromRank(rank, meshShape, processGroup);

                var coord = mesh.MyCoordinate;

                // Assert
                Assert.IsNotNull(coord);
                Assert.AreEqual(1, coord.Dimensions);
                Assert.AreEqual(rank, coord[0], "Rank should equal coordinate in 1D mesh");

                processGroup.Destroy();
            }
        }

        [TestMethod]
        public void DeviceMesh_3DMesh_CreatesCorrectCoordinates()
        {
            // Arrange
            int[] meshShape = new[] { 2, 2, 2 }; // 2x2x2 mesh
            int totalDevices = meshShape.Aggregate(1, (a, b) => a * b);

            for (int rank = 0; rank < totalDevices; rank++)
            {
                // Act
                var processGroup = MockProcessGroup.Create(totalDevices, rank);
                var mesh = DeviceMesh.CreateFromRank(rank, meshShape, processGroup);

                var coord = mesh.MyCoordinate;

                // Assert
                Assert.IsNotNull(coord);
                Assert.AreEqual(3, coord.Dimensions);
                Assert.IsTrue(coord[0] < 2, "First coordinate should be < 2");
                Assert.IsTrue(coord[1] < 2, "Second coordinate should be < 2");
                Assert.IsTrue(coord[2] < 2, "Third coordinate should be < 2");

                processGroup.Destroy();
            }
        }

        [TestMethod]
        public void DeviceMesh_CoordinateUniqueness_AllRanksHaveUniqueCoords()
        {
            // Arrange
            int[] meshShape = new[] { 2, 2 };
            var meshes = new DeviceMesh[4];
            var processGroups = new ProcessGroup[4];

            for (int rank = 0; rank < 4; rank++)
            {
                processGroups[rank] = MockProcessGroup.Create(4, rank);
                meshes[rank] = DeviceMesh.CreateFromRank(rank, meshShape, processGroups[rank]);
            }

            // Act & Assert
            // All ranks should have unique coordinates
            var coordinates = meshes.Select(m => $"{m.MyCoordinate[0]},{m.MyCoordinate[1]}").ToList();
            var uniqueCoords = coordinates.Distinct().ToList();
            Assert.AreEqual(4, uniqueCoords.Count, "All ranks should have unique coordinates");

            // Cleanup
            foreach (var pg in processGroups)
            {
                pg.Destroy();
            }
        }

        [TestMethod]
        public void DeviceMesh_GetDPGroup_ReturnsProcessGroup()
        {
            // Arrange
            int[] meshShape = new[] { 2, 2 };
            var processGroup = MockProcessGroup.Create(4, 0);
            var mesh = DeviceMesh.CreateFromRank(0, meshShape, processGroup);

            // Act
            var dpGroup = mesh.GetDPGroup();

            // Assert
            Assert.IsNotNull(dpGroup);

            processGroup.Destroy();
        }

        [TestMethod]
        public void DeviceMesh_GetTPGroup_ReturnsProcessGroup()
        {
            // Arrange
            int[] meshShape = new[] { 2, 2 };
            var processGroup = MockProcessGroup.Create(4, 0);
            var mesh = DeviceMesh.CreateFromRank(0, meshShape, processGroup);

            // Act
            var tpGroup = mesh.GetTPGroup();

            // Assert
            Assert.IsNotNull(tpGroup);

            processGroup.Destroy();
        }

        [TestMethod]
        public void DeviceMesh_LargeMesh_HandlesMultipleDimensions()
        {
            // Arrange
            int[] meshShape = new[] { 4, 3, 2 }; // 24 devices
            int totalDevices = meshShape.Aggregate(1, (a, b) => a * b);

            for (int rank = 0; rank < totalDevices; rank++)
            {
                // Act
                var processGroup = MockProcessGroup.Create(totalDevices, rank);
                var mesh = DeviceMesh.CreateFromRank(rank, meshShape, processGroup);

                var coord = mesh.MyCoordinate;

                // Assert
                Assert.IsNotNull(coord);
                Assert.AreEqual(3, coord.Dimensions);
                Assert.IsTrue(coord[0] < 4, "First coordinate should be < 4");
                Assert.IsTrue(coord[1] < 3, "Second coordinate should be < 3");
                Assert.IsTrue(coord[2] < 2, "Third coordinate should be < 2");

                processGroup.Destroy();
            }
        }

        [TestMethod]
        public void DeviceMesh_ZeroRank_CreatesCorrectly()
        {
            // Arrange & Act
            int[] meshShape = new[] { 2, 2 };
            var processGroup = MockProcessGroup.Create(4, 0);
            var mesh = DeviceMesh.CreateFromRank(0, meshShape, processGroup);

            var coord = mesh.MyCoordinate;

            // Assert
            Assert.IsNotNull(coord);
            Assert.AreEqual(0, coord[0], "Rank 0 should have first coordinate 0");
            Assert.AreEqual(0, coord[1], "Rank 0 should have second coordinate 0");

            processGroup.Destroy();
        }

        [TestMethod]
        public void DeviceMesh_LastRank_CreatesCorrectly()
        {
            // Arrange & Act
            int[] meshShape = new[] { 2, 2 };
            var processGroup = MockProcessGroup.Create(4, 3); // Last rank in 2x2 mesh
            var mesh = DeviceMesh.CreateFromRank(3, meshShape, processGroup);

            var coord = mesh.MyCoordinate;

            // Assert
            Assert.IsNotNull(coord);
            Assert.IsTrue(coord[0] < 2, "Last rank should have valid first coordinate");
            Assert.IsTrue(coord[1] < 2, "Last rank should have valid second coordinate");

            processGroup.Destroy();
        }
    }
}
