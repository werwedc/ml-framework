using Xunit;
using MLFramework.Data;

namespace MLFramework.Tests.Data
{
    public class PrefetchBufferTests
    {
        [Fact]
        public void Constructor_ValidCapacity_CreatesBuffer()
        {
            // Arrange & Act
            var buffer = new PrefetchBuffer<int>(10);

            // Assert
            Assert.Equal(10, buffer.Capacity);
            Assert.Equal(0, buffer.Count);
            Assert.True(buffer.IsEmpty);
            Assert.False(buffer.IsFull);
        }

        [Fact]
        public void Constructor_ZeroCapacity_ThrowsArgumentOutOfRangeException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => new PrefetchBuffer<int>(0));
        }

        [Fact]
        public void Constructor_NegativeCapacity_ThrowsArgumentOutOfRangeException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => new PrefetchBuffer<int>(-1));
        }

        [Fact]
        public void Add_Item_BufferCountIncreases()
        {
            // Arrange
            var buffer = new PrefetchBuffer<int>(10);

            // Act
            buffer.Add(1);
            buffer.Add(2);
            buffer.Add(3);

            // Assert
            Assert.Equal(3, buffer.Count);
            Assert.False(buffer.IsEmpty);
        }

        [Fact]
        public void Add_FullBuffer_ThrowsInvalidOperationException()
        {
            // Arrange
            var buffer = new PrefetchBuffer<int>(2);
            buffer.Add(1);
            buffer.Add(2);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => buffer.Add(3));
        }

        [Fact]
        public void GetNext_Item_ReturnsCorrectItem()
        {
            // Arrange
            var buffer = new PrefetchBuffer<int>(10);
            buffer.Add(1);
            buffer.Add(2);
            buffer.Add(3);

            // Act
            var item = buffer.GetNext();

            // Assert
            Assert.Equal(1, item);
            Assert.Equal(2, buffer.Count);
        }

        [Fact]
        public void GetNext_MultipleItems_ReturnsInFIFOOrder()
        {
            // Arrange
            var buffer = new PrefetchBuffer<int>(10);
            buffer.Add(1);
            buffer.Add(2);
            buffer.Add(3);

            // Act
            var item1 = buffer.GetNext();
            var item2 = buffer.GetNext();
            var item3 = buffer.GetNext();

            // Assert
            Assert.Equal(1, item1);
            Assert.Equal(2, item2);
            Assert.Equal(3, item3);
            Assert.Equal(0, buffer.Count);
            Assert.True(buffer.IsEmpty);
        }

        [Fact]
        public void GetNext_EmptyBuffer_ThrowsInvalidOperationException()
        {
            // Arrange
            var buffer = new PrefetchBuffer<int>(10);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => buffer.GetNext());
        }

        [Fact]
        public void Peek_Item_ReturnsItemWithoutRemoving()
        {
            // Arrange
            var buffer = new PrefetchBuffer<int>(10);
            buffer.Add(1);
            buffer.Add(2);

            // Act
            var item = buffer.Peek();

            // Assert
            Assert.Equal(1, item);
            Assert.Equal(2, buffer.Count); // Count unchanged
        }

        [Fact]
        public void Peek_EmptyBuffer_ThrowsInvalidOperationException()
        {
            // Arrange
            var buffer = new PrefetchBuffer<int>(10);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => buffer.Peek());
        }

        [Fact]
        public void TryGet_Item_ReturnsTrueAndItem()
        {
            // Arrange
            var buffer = new PrefetchBuffer<int>(10);
            buffer.Add(1);

            // Act
            bool result = buffer.TryGet(out int item);

            // Assert
            Assert.True(result);
            Assert.Equal(1, item);
            Assert.Equal(0, buffer.Count);
        }

        [Fact]
        public void TryGet_EmptyBuffer_ReturnsFalse()
        {
            // Arrange
            var buffer = new PrefetchBuffer<int>(10);

            // Act
            bool result = buffer.TryGet(out int item);

            // Assert
            Assert.False(result);
            Assert.Equal(0, item);
        }

        [Fact]
        public void Clear_BufferWithItems_EmptiesBuffer()
        {
            // Arrange
            var buffer = new PrefetchBuffer<int>(10);
            buffer.Add(1);
            buffer.Add(2);
            buffer.Add(3);

            // Act
            buffer.Clear();

            // Assert
            Assert.Equal(0, buffer.Count);
            Assert.True(buffer.IsEmpty);
            Assert.False(buffer.IsFull);
        }

        [Fact]
        public void IsFull_BufferAtCapacity_ReturnsTrue()
        {
            // Arrange
            var buffer = new PrefetchBuffer<int>(3);
            buffer.Add(1);
            buffer.Add(2);
            buffer.Add(3);

            // Act & Assert
            Assert.True(buffer.IsFull);
        }

        [Fact]
        public void IsFull_BufferNotAtCapacity_ReturnsFalse()
        {
            // Arrange
            var buffer = new PrefetchBuffer<int>(3);
            buffer.Add(1);
            buffer.Add(2);

            // Act & Assert
            Assert.False(buffer.IsFull);
        }

        [Fact]
        public void MultipleOperations_WorkCorrectly()
        {
            // Arrange
            var buffer = new PrefetchBuffer<int>(5);

            // Act
            buffer.Add(1);
            buffer.Add(2);
            buffer.Add(3);
            var item1 = buffer.GetNext();
            buffer.Add(4);
            buffer.Add(5);
            var item2 = buffer.GetNext();
            var item3 = buffer.GetNext();
            var item4 = buffer.GetNext();
            var item5 = buffer.GetNext();

            // Assert
            Assert.Equal(1, item1);
            Assert.Equal(2, item2);
            Assert.Equal(3, item3);
            Assert.Equal(4, item4);
            Assert.Equal(5, item5);
            Assert.Equal(0, buffer.Count);
            Assert.True(buffer.IsEmpty);
        }
    }
}
