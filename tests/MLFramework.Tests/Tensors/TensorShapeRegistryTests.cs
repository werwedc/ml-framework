using Xunit;
using MLFramework.Tensors;

namespace MLFramework.Tests.Tensors
{
    /// <summary>
    /// Unit tests for TensorShapeRegistry class.
    /// </summary>
    public class TensorShapeRegistryTests
    {
        [Fact]
        public void Constructor_CreatesEmptyRegistry()
        {
            // Act
            var registry = new TensorShapeRegistry();

            // Assert
            Assert.Equal(0, registry.Count);
        }

        [Fact]
        public void RegisterBinding_WithValidNameAndValue_AddsBinding()
        {
            // Arrange
            var registry = new TensorShapeRegistry();

            // Act
            registry.RegisterBinding("batch_size", 32);

            // Assert
            Assert.Equal(1, registry.Count);
            Assert.Equal(32, registry.GetBinding("batch_size"));
        }

        [Fact]
        public void RegisterBinding_WithNullName_ThrowsArgumentNullException()
        {
            // Arrange
            var registry = new TensorShapeRegistry();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => registry.RegisterBinding(null!, 32));
        }

        [Fact]
        public void RegisterBinding_WithEmptyName_ThrowsArgumentNullException()
        {
            // Arrange
            var registry = new TensorShapeRegistry();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => registry.RegisterBinding("", 32));
        }

        [Fact]
        public void RegisterBinding_WithWhitespaceName_ThrowsArgumentNullException()
        {
            // Arrange
            var registry = new TensorShapeRegistry();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => registry.RegisterBinding("   ", 32));
        }

        [Fact]
        public void RegisterBinding_WithNegativeValue_ThrowsArgumentException()
        {
            // Arrange
            var registry = new TensorShapeRegistry();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => registry.RegisterBinding("batch_size", -1));
        }

        [Fact]
        public void RegisterBinding_WithZeroValue_AddsBinding()
        {
            // Arrange
            var registry = new TensorShapeRegistry();

            // Act
            registry.RegisterBinding("scalar", 0);

            // Assert
            Assert.Equal(0, registry.GetBinding("scalar"));
        }

        [Fact]
        public void RegisterBinding_UpdatesExistingBinding()
        {
            // Arrange
            var registry = new TensorShapeRegistry();
            registry.RegisterBinding("batch_size", 32);

            // Act
            registry.RegisterBinding("batch_size", 64);

            // Assert
            Assert.Equal(1, registry.Count);
            Assert.Equal(64, registry.GetBinding("batch_size"));
        }

        [Fact]
        public void GetBinding_WithExistingBinding_ReturnsValue()
        {
            // Arrange
            var registry = new TensorShapeRegistry();
            registry.RegisterBinding("batch_size", 32);

            // Act
            var result = registry.GetBinding("batch_size");

            // Assert
            Assert.Equal(32, result);
        }

        [Fact]
        public void GetBinding_WithNonExistentBinding_ReturnsNull()
        {
            // Arrange
            var registry = new TensorShapeRegistry();

            // Act
            var result = registry.GetBinding("unknown_dim");

            // Assert
            Assert.Null(result);
        }

        [Fact]
        public void GetBinding_WithNullName_ThrowsArgumentNullException()
        {
            // Arrange
            var registry = new TensorShapeRegistry();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => registry.GetBinding(null!));
        }

        [Fact]
        public void HasBinding_WithExistingBinding_ReturnsTrue()
        {
            // Arrange
            var registry = new TensorShapeRegistry();
            registry.RegisterBinding("batch_size", 32);

            // Act
            var result = registry.HasBinding("batch_size");

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void HasBinding_WithNonExistentBinding_ReturnsFalse()
        {
            // Arrange
            var registry = new TensorShapeRegistry();

            // Act
            var result = registry.HasBinding("unknown_dim");

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void HasBinding_WithNullName_ThrowsArgumentNullException()
        {
            // Arrange
            var registry = new TensorShapeRegistry();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => registry.HasBinding(null!));
        }

        [Fact]
        public void RemoveBinding_WithExistingBinding_RemovesBinding()
        {
            // Arrange
            var registry = new TensorShapeRegistry();
            registry.RegisterBinding("batch_size", 32);

            // Act
            var result = registry.RemoveBinding("batch_size");

            // Assert
            Assert.True(result);
            Assert.Equal(0, registry.Count);
            Assert.False(registry.HasBinding("batch_size"));
        }

        [Fact]
        public void RemoveBinding_WithNonExistentBinding_ReturnsFalse()
        {
            // Arrange
            var registry = new TensorShapeRegistry();

            // Act
            var result = registry.RemoveBinding("unknown_dim");

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void RemoveBinding_WithNullName_ThrowsArgumentNullException()
        {
            // Arrange
            var registry = new TensorShapeRegistry();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => registry.RemoveBinding(null!));
        }

        [Fact]
        public void ClearBindings_RemovesAllBindings()
        {
            // Arrange
            var registry = new TensorShapeRegistry();
            registry.RegisterBinding("batch_size", 32);
            registry.RegisterBinding("seq_len", 128);
            registry.RegisterBinding("hidden", 512);

            // Act
            registry.ClearBindings();

            // Assert
            Assert.Equal(0, registry.Count);
        }

        [Fact]
        public void Clone_CreatesIndependentCopy()
        {
            // Arrange
            var registry = new TensorShapeRegistry();
            registry.RegisterBinding("batch_size", 32);
            registry.RegisterBinding("seq_len", 128);

            // Act
            var clone = registry.Clone();
            clone.RegisterBinding("hidden", 512);
            clone.RegisterBinding("batch_size", 64);

            // Assert
            Assert.Equal(2, registry.Count);
            Assert.Equal(32, registry.GetBinding("batch_size"));
            Assert.Equal(3, clone.Count);
            Assert.Equal(64, clone.GetBinding("batch_size"));
        }

        [Fact]
        public void Clone_WithEmptyRegistry_CreatesEmptyCopy()
        {
            // Arrange
            var registry = new TensorShapeRegistry();

            // Act
            var clone = registry.Clone();

            // Assert
            Assert.Equal(0, clone.Count);
            Assert.NotSame(registry, clone);
        }

        [Fact]
        public void MergeFrom_MergesBindings()
        {
            // Arrange
            var registry1 = new TensorShapeRegistry();
            registry1.RegisterBinding("batch_size", 32);
            registry1.RegisterBinding("seq_len", 128);

            var registry2 = new TensorShapeRegistry();
            registry2.RegisterBinding("hidden", 512);
            registry2.RegisterBinding("seq_len", 256);

            // Act
            registry1.MergeFrom(registry2);

            // Assert
            Assert.Equal(3, registry1.Count);
            Assert.Equal(32, registry1.GetBinding("batch_size"));
            Assert.Equal(256, registry1.GetBinding("seq_len")); // Should be overwritten
            Assert.Equal(512, registry1.GetBinding("hidden"));
        }

        [Fact]
        public void MergeFrom_WithNullRegistry_ThrowsArgumentNullException()
        {
            // Arrange
            var registry = new TensorShapeRegistry();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => registry.MergeFrom(null!));
        }

        [Fact]
        public void GetBoundDimensionNames_ReturnsAllBoundNames()
        {
            // Arrange
            var registry = new TensorShapeRegistry();
            registry.RegisterBinding("batch_size", 32);
            registry.RegisterBinding("seq_len", 128);
            registry.RegisterBinding("hidden", 512);

            // Act
            var names = registry.GetBoundDimensionNames();

            // Assert
            Assert.Equal(3, names.Count());
            Assert.Contains("batch_size", names);
            Assert.Contains("seq_len", names);
            Assert.Contains("hidden", names);
        }

        [Fact]
        public void GetBoundDimensionNames_WithEmptyRegistry_ReturnsEmpty()
        {
            // Arrange
            var registry = new TensorShapeRegistry();

            // Act
            var names = registry.GetBoundDimensionNames();

            // Assert
            Assert.Empty(names);
        }

        [Fact]
        public void ToString_WithBindings_ReturnsDescriptiveString()
        {
            // Arrange
            var registry = new TensorShapeRegistry();
            registry.RegisterBinding("batch_size", 32);
            registry.RegisterBinding("seq_len", 128);

            // Act
            var result = registry.ToString();

            // Assert
            Assert.Contains("TensorShapeRegistry", result);
            Assert.Contains("batch_size=32", result);
            Assert.Contains("seq_len=128", result);
        }

        [Fact]
        public void ToString_WithEmptyRegistry_ReturnsEmptyMessage()
        {
            // Arrange
            var registry = new TensorShapeRegistry();

            // Act
            var result = registry.ToString();

            // Assert
            Assert.Equal("TensorShapeRegistry: No bindings", result);
        }

        [Fact]
        public void CaseSensitiveBinding_NameComparisonIsCaseSensitive()
        {
            // Arrange
            var registry = new TensorShapeRegistry();

            // Act
            registry.RegisterBinding("BatchSize", 32);

            // Assert
            Assert.True(registry.HasBinding("BatchSize"));
            Assert.False(registry.HasBinding("batch_size"));
            Assert.False(registry.HasBinding("BATCHSIZE"));
        }
    }
}
