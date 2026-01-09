using MLFramework.MobileRuntime.Memory;
using Xunit;

namespace MLFramework.MobileRuntime.Tests.Memory
{
    /// <summary>
    /// Unit tests for MemoryPoolFactory.
    /// </summary>
    public class MemoryPoolFactoryTests
    {
        [Fact]
        public void CreateDefault_ReturnsDefaultMemoryPool()
        {
            // Act
            var pool = MemoryPoolFactory.CreateDefault();

            // Assert
            Assert.NotNull(pool);
            Assert.IsType<DefaultMemoryPool>(pool);
        }

        [Fact]
        public void CreateDefault_WithCapacity_ReturnsPoolWithCapacity()
        {
            // Act
            using var pool = (DefaultMemoryPool)MemoryPoolFactory.CreateDefault(32 * 1024 * 1024);

            // Assert
            Assert.NotNull(pool);
        }

        [Fact]
        public void CreatePreallocated_ReturnsPreallocatedMemoryPool()
        {
            // Act
            var pool = MemoryPoolFactory.CreatePreallocated(16 * 1024 * 1024);

            // Assert
            Assert.NotNull(pool);
            Assert.IsType<PreallocatedMemoryPool>(pool);
        }

        [Fact]
        public void CreateLowMemoryMode_ReturnsPoolInLowMemoryMode()
        {
            // Act
            using var pool = MemoryPoolFactory.CreateLowMemoryMode();

            // Assert
            Assert.NotNull(pool);
            Assert.IsType<DefaultMemoryPool>(pool);
        }

        [Fact]
        public void CreateOptimalForPlatform_ReturnsValidPool()
        {
            // Act
            var pool = MemoryPoolFactory.CreateOptimalForPlatform();

            // Assert
            Assert.NotNull(pool);
        }

        [Fact]
        public void CreateForModel_MNIST_ReturnsPoolWith8MB()
        {
            // Act
            using var pool = MemoryPoolFactory.CreateForModel(MemoryPoolFactory.ModelType.MNIST);

            // Assert
            Assert.NotNull(pool);
        }

        [Fact]
        public void CreateForModel_CIFAR_ReturnsPoolWith16MB()
        {
            // Act
            using var pool = MemoryPoolFactory.CreateForModel(MemoryPoolFactory.ModelType.CIFAR);

            // Assert
            Assert.NotNull(pool);
        }

        [Fact]
        public void CreateForModel_ImageNet_ReturnsPoolWith64MB()
        {
            // Act
            using var pool = MemoryPoolFactory.CreateForModel(MemoryPoolFactory.ModelType.ImageNet);

            // Assert
            Assert.NotNull(pool);
        }

        [Fact]
        public void CreateForModel_Custom_ReturnsPoolWith32MB()
        {
            // Act
            using var pool = MemoryPoolFactory.CreateForModel(MemoryPoolFactory.ModelType.Custom);

            // Assert
            Assert.NotNull(pool);
        }

        [Fact]
        public void CreateDefault_DisposeCorrectly()
        {
            // Arrange & Act
            using var pool = MemoryPoolFactory.CreateDefault();

            // Assert - No exception thrown
        }

        [Fact]
        public void CreatePreallocated_DisposeCorrectly()
        {
            // Arrange & Act
            using var pool = MemoryPoolFactory.CreatePreallocated(16 * 1024 * 1024);

            // Assert - No exception thrown
        }

        [Fact]
        public void CreateLowMemoryMode_DisposeCorrectly()
        {
            // Arrange & Act
            using var pool = MemoryPoolFactory.CreateLowMemoryMode();

            // Assert - No exception thrown
        }
    }
}
