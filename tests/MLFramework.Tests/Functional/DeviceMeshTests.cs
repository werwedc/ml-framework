using Xunit;
using MLFramework.Functional.Distributed;

namespace MLFramework.Tests.Functional
{
    public class DeviceTests
    {
        [Fact]
        public void Constructor_ShouldInitializeCorrectly()
        {
            // Act
            var device = new Device(0, DeviceType.GPU, "gpu:0");

            // Assert
            Assert.Equal(0, device.Id);
            Assert.Equal(DeviceType.GPU, device.Type);
            Assert.Equal("gpu:0", device.Name);
            Assert.True(device.IsAvailable);
        }

        [Fact]
        public void CPU_ShouldCreateCPUDevice()
        {
            // Act
            var device = Device.CPU(5);

            // Assert
            Assert.Equal(5, device.Id);
            Assert.Equal(DeviceType.CPU, device.Type);
            Assert.Equal("cpu:5", device.Name);
        }

        [Fact]
        public void GPU_ShouldCreateGPUDevice()
        {
            // Act
            var device = Device.GPU(3);

            // Assert
            Assert.Equal(3, device.Id);
            Assert.Equal(DeviceType.GPU, device.Type);
            Assert.Equal("gpu:3", device.Name);
        }

        [Fact]
        public void ToString_ShouldReturnCorrectFormat()
        {
            // Arrange
            var device = Device.GPU(0);

            // Act
            var str = device.ToString();

            // Assert
            Assert.Equal("Device(GPU:0)", str);
        }
    }

    public class DeviceMeshTests
    {
        [Fact]
        public void Create1D_ShouldCreateCorrectMesh()
        {
            // Arrange
            var devices = new[]
            {
                Device.CPU(0), Device.CPU(1), Device.CPU(2), Device.CPU(3)
            };

            // Act
            var mesh = new DeviceMesh(devices);

            // Assert
            Assert.Equal(4, mesh.DeviceCount);
            Assert.Single(mesh.Shape);
            Assert.Equal(4, mesh.Shape[0]);
            Assert.Equal(1, mesh.Rank);
        }

        [Fact]
        public void Create2D_ShouldCreateCorrectMesh()
        {
            // Arrange
            var devices = new Device[6];
            for (int i = 0; i < 6; i++)
                devices[i] = Device.CPU(i);

            // Act
            var mesh = new DeviceMesh(new int[] { 2, 3 }, devices);

            // Assert
            Assert.Equal(6, mesh.DeviceCount);
            Assert.Equal(2, mesh.Rank);
            Assert.Equal(2, mesh.Shape[0]);
            Assert.Equal(3, mesh.Shape[1]);
        }

        [Fact]
        public void Create2D_ShouldThrowForMismatchedDeviceCount()
        {
            // Arrange
            var devices = new Device[4];  // 4 devices but need 6 (2x3)

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new DeviceMesh(new int[] { 2, 3 }, devices));
        }

        [Fact]
        public void GetDevice_1D_ShouldReturnCorrectDevice()
        {
            // Arrange
            var devices = new[] { Device.CPU(0), Device.CPU(1), Device.CPU(2) };
            var mesh = new DeviceMesh(devices);

            // Act
            var device = mesh.GetDevice(1);

            // Assert
            Assert.Same(devices[1], device);
        }

        [Fact]
        public void GetDevice_2D_ShouldReturnCorrectDevice()
        {
            // Arrange
            var devices = new Device[4];
            for (int i = 0; i < 4; i++)
                devices[i] = Device.CPU(i);

            var mesh = new DeviceMesh(new int[] { 2, 2 }, devices);

            // Act
            var device = mesh.GetDevice(0, 1);

            // Assert
            Assert.Same(devices[1], device);
        }

        [Fact]
        public void GetDevice_ShouldThrowForInvalidIndexCount()
        {
            // Arrange
            var devices = new[] { Device.CPU(0), Device.CPU(1) };
            var mesh = new DeviceMesh(devices);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => mesh.GetDevice(0, 0));
        }

        [Fact]
        public void GetDevice_ShouldThrowForOutOfRangeIndex()
        {
            // Arrange
            var devices = new[] { Device.CPU(0), Device.CPU(1) };
            var mesh = new DeviceMesh(devices);

            // Act & Assert
            Assert.Throws<IndexOutOfRangeException>(() => mesh.GetDevice(5));
        }

        [Fact]
        public void ShardingAxis_ShouldReturnCorrectAxis()
        {
            // Arrange
            var devices = new[] { Device.CPU(0), Device.CPU(1) };
            var mesh = new DeviceMesh(devices);

            // Act
            var axis = mesh["data"];

            // Assert
            Assert.NotNull(axis);
            Assert.Equal("data", axis.Name);
        }

        [Fact]
        public void ShardingAxis_ShouldThrowForUnknownAxis()
        {
            // Arrange
            var devices = new[] { Device.CPU(0) };
            var mesh = new DeviceMesh(devices);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
            {
                var _ = mesh["unknown"];
            });
        }
    }

    public class DeviceMeshFactoryTests
    {
        [Fact]
        public void Create1D_ShouldCreateCPUMesh()
        {
            // Act
            var mesh = DeviceMeshFactory.Create1D(4, DeviceType.CPU);

            // Assert
            Assert.Equal(4, mesh.DeviceCount);
            Assert.Single(mesh.Shape);
            Assert.Equal(4, mesh.Shape[0]);
        }

        [Fact]
        public void Create1D_ShouldCreateGPUMesh()
        {
            // Act
            var mesh = DeviceMeshFactory.Create1D(8, DeviceType.GPU);

            // Assert
            Assert.Equal(8, mesh.DeviceCount);

            // Verify all are GPU devices
            var device = mesh.GetDevice(0);
            Assert.Equal(DeviceType.GPU, device.Type);
        }

        [Fact]
        public void Create2D_ShouldCreateCorrectTopology()
        {
            // Act
            var mesh = DeviceMeshFactory.Create2D(2, 4, DeviceType.CPU);

            // Assert
            Assert.Equal(8, mesh.DeviceCount);
            Assert.Equal(2, mesh.Rank);
            Assert.Equal(2, mesh.Shape[0]);
            Assert.Equal(4, mesh.Shape[1]);
        }

        [Fact]
        public void Default_ShouldReturnSingleCPUMesh()
        {
            // Act
            var mesh = DeviceMeshFactory.Default();

            // Assert
            Assert.Equal(1, mesh.DeviceCount);
            var device = mesh.GetDevice(0);
            Assert.Equal(DeviceType.CPU, device.Type);
        }
    }
}
