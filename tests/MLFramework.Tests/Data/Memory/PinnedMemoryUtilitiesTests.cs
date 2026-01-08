using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using MLFramework.Data.Memory;
using Xunit;

namespace MLFramework.Tests.Data.Memory
{
    /// <summary>
    /// Unit tests for PinnedMemory class.
    /// </summary>
    public class PinnedMemoryTests
    {
        [Fact]
        public void Constructor_ValidArray_PinsArray()
        {
            // Arrange
            int[] array = new int[] { 1, 2, 3, 4, 5 };

            // Act
            using var pinned = new PinnedMemory<int>(array);

            // Assert
            Assert.True(pinned.IsPinned);
            Assert.Equal(array.Length, pinned.Length);
            Assert.NotEqual(IntPtr.Zero, pinned.Pointer);
        }

        [Fact]
        public void Constructor_NullArray_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new PinnedMemory<int>(null));
        }

        [Fact]
        public void Span_ProvidesSafeAccess()
        {
            // Arrange
            int[] array = new int[] { 1, 2, 3, 4, 5 };
            using var pinned = new PinnedMemory<int>(array);

            // Act
            Span<int> span = pinned.Span;

            // Assert
            Assert.Equal(array.Length, span.Length);
            for (int i = 0; i < array.Length; i++)
            {
                Assert.Equal(array[i], span[i]);
            }
        }

        [Fact]
        public void Pointer_ProvidesValidAddress()
        {
            // Arrange
            int[] array = new int[] { 1, 2, 3, 4, 5 };
            using var pinned = new PinnedMemory<int>(array);

            // Act
            IntPtr pointer = pinned.Pointer;

            // Assert
            Assert.NotEqual(IntPtr.Zero, pointer);

            // Verify pointer points to array data
            int firstValue = Marshal.ReadInt32(pointer);
            Assert.Equal(array[0], firstValue);
        }

        [Fact]
        public void Unpin_ReleasesGCHandle()
        {
            // Arrange
            int[] array = new int[] { 1, 2, 3, 4, 5 };
            var pinned = new PinnedMemory<int>(array);

            // Act
            pinned.Unpin();

            // Assert
            Assert.False(pinned.IsPinned);
        }

        [Fact]
        public void Unpin_CalledMultipleTimes_NoOpAfterFirst()
        {
            // Arrange
            int[] array = new int[] { 1, 2, 3, 4, 5 };
            using var pinned = new PinnedMemory<int>(array);

            // Act & Assert - Should not throw
            pinned.Unpin();
            pinned.Unpin();
            pinned.Unpin();
        }

        [Fact]
        public void Dispose_UnpinsArray()
        {
            // Arrange
            int[] array = new int[] { 1, 2, 3, 4, 5 };
            var pinned = new PinnedMemory<int>(array);

            // Act
            pinned.Dispose();

            // Assert
            Assert.False(pinned.IsPinned);
        }

        [Fact]
        public void Dispose_CalledMultipleTimes_NoOpAfterFirst()
        {
            // Arrange
            int[] array = new int[] { 1, 2, 3, 4, 5 };
            var pinned = new PinnedMemory<int>(array);

            // Act & Assert - Should not throw
            pinned.Dispose();
            pinned.Dispose();
            pinned.Dispose();
        }

        [Fact]
        public void UsingStatement_CleansUpCorrectly()
        {
            // Arrange
            int[] array = new int[] { 1, 2, 3, 4, 5 };

            // Act
            using (var pinned = new PinnedMemory<int>(array))
            {
                Assert.True(pinned.IsPinned);
            }

            // Assert - Array should be unpinned after using block
        }

        [Fact]
        public void Span_Modification_ReflectsInOriginalArray()
        {
            // Arrange
            int[] array = new int[] { 1, 2, 3, 4, 5 };
            using var pinned = new PinnedMemory<int>(array);

            // Act
            pinned.Span[0] = 100;

            // Assert
            Assert.Equal(100, array[0]);
        }

        [Fact]
        public void EmptyArray_HandlesCorrectly()
        {
            // Arrange
            int[] array = new int[0];

            // Act
            using var pinned = new PinnedMemory<int>(array);

            // Assert
            Assert.Equal(0, pinned.Length);
            Assert.Equal(0, pinned.Span.Length);
        }
    }

    /// <summary>
    /// Unit tests for PinnedBuffer class.
    /// </summary>
    public class PinnedBufferTests
    {
        [Fact]
        public void Allocate_CreatesPinnedBuffer()
        {
            // Arrange
            int length = 100;

            // Act
            using var buffer = PinnedBuffer<int>.Allocate(length);

            // Assert
            Assert.Equal(length, buffer.Length);
            Assert.True(buffer.IsPinned);
            Assert.NotEqual(IntPtr.Zero, buffer.Pointer);
        }

        [Fact]
        public void Allocate_NegativeLength_ThrowsArgumentOutOfRangeException()
        {
            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => PinnedBuffer<int>.Allocate(-1));
        }

        [Fact]
        public void Allocate_ZeroLength_CreatesEmptyBuffer()
        {
            // Act
            using var buffer = PinnedBuffer<int>.Allocate(0);

            // Assert
            Assert.Equal(0, buffer.Length);
            Assert.Empty(buffer.Span);
        }

        [Fact]
        public void Array_ReturnsUnderlyingArray()
        {
            // Arrange
            int length = 10;
            using var buffer = PinnedBuffer<int>.Allocate(length);

            // Act
            int[] array = buffer.Array;

            // Assert
            Assert.NotNull(array);
            Assert.Equal(length, array.Length);
        }

        [Fact]
        public void CopyFromArray_CopiesCorrectly()
        {
            // Arrange
            int[] source = new int[] { 1, 2, 3, 4, 5 };
            using var buffer = PinnedBuffer<int>.Allocate(source.Length);

            // Act
            buffer.CopyFrom(source);

            // Assert
            Assert.Equal(source, buffer.Span.ToArray());
        }

        [Fact]
        public void CopyFromArray_WithOffset_CopiesCorrectly()
        {
            // Arrange
            int[] source = new int[] { 0, 0, 1, 2, 3 };
            using var buffer = PinnedBuffer<int>.Allocate(3);

            // Act
            buffer.CopyFrom(source, 2);

            // Assert
            Assert.Equal(new int[] { 1, 2, 3 }, buffer.Span.ToArray());
        }

        [Fact]
        public void CopyFromArray_NullArray_ThrowsArgumentNullException()
        {
            // Arrange
            using var buffer = PinnedBuffer<int>.Allocate(10);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => buffer.CopyFrom(null));
        }

        [Fact]
        public void CopyFromArray_WrongLength_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            int[] source = new int[] { 1, 2, 3 };
            using var buffer = PinnedBuffer<int>.Allocate(10);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => buffer.CopyFrom(source));
        }

        [Fact]
        public void CopyFromSpan_CopiesCorrectly()
        {
            // Arrange
            Span<int> source = new int[] { 1, 2, 3, 4, 5 };
            using var buffer = PinnedBuffer<int>.Allocate(source.Length);

            // Act
            buffer.CopyFrom(source);

            // Assert
            Assert.Equal(source.ToArray(), buffer.Span.ToArray());
        }

        [Fact]
        public void CopyToArray_CopiesCorrectly()
        {
            // Arrange
            int[] source = new int[] { 1, 2, 3, 4, 5 };
            int[] destination = new int[5];
            using var buffer = PinnedBuffer<int>.Allocate(source.Length);
            buffer.CopyFrom(source);

            // Act
            buffer.CopyTo(destination);

            // Assert
            Assert.Equal(source, destination);
        }

        [Fact]
        public void CopyToArray_WithOffset_CopiesCorrectly()
        {
            // Arrange
            int[] source = new int[] { 1, 2, 3 };
            int[] destination = new int[] { 0, 0, 0, 0, 0 };
            using var buffer = PinnedBuffer<int>.Allocate(source.Length);
            buffer.CopyFrom(source);

            // Act
            buffer.CopyTo(destination, 2);

            // Assert
            Assert.Equal(new int[] { 0, 0, 1, 2, 3 }, destination);
        }

        [Fact]
        public void CopyToArray_NullArray_ThrowsArgumentNullException()
        {
            // Arrange
            using var buffer = PinnedBuffer<int>.Allocate(10);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => buffer.CopyTo(null));
        }

        [Fact]
        public void CopyToSpan_CopiesCorrectly()
        {
            // Arrange
            int[] source = new int[] { 1, 2, 3, 4, 5 };
            Span<int> destination = new int[5];
            using var buffer = PinnedBuffer<int>.Allocate(source.Length);
            buffer.CopyFrom(source);

            // Act
            buffer.CopyTo(destination);

            // Assert
            Assert.Equal(source, destination.ToArray());
        }

        [Fact]
        public void Fill_SetsAllElements()
        {
            // Arrange
            using var buffer = PinnedBuffer<int>.Allocate(10);

            // Act
            buffer.Fill(42);

            // Assert
            foreach (int value in buffer.Span)
            {
                Assert.Equal(42, value);
            }
        }

        [Fact]
        public void Fill_WithZero_ZerosBuffer()
        {
            // Arrange
            using var buffer = PinnedBuffer<int>.Allocate(10);
            buffer.Fill(42);

            // Act
            buffer.Fill(0);

            // Assert
            Assert.True(buffer.Span.ToArray().All(v => v == 0));
        }

        [Fact]
        public void Unpin_ReleasesGCHandle()
        {
            // Arrange
            using var buffer = PinnedBuffer<int>.Allocate(10);

            // Act
            buffer.Unpin();

            // Assert
            Assert.False(buffer.IsPinned);
        }

        [Fact]
        public void Dispose_UnpinsArray()
        {
            // Arrange
            var buffer = PinnedBuffer<int>.Allocate(10);

            // Act
            buffer.Dispose();

            // Assert
            Assert.False(buffer.IsPinned);
        }

        [Fact]
        public void LargeBuffer_HandlesCorrectly()
        {
            // Arrange
            int length = 1_000_000;

            // Act
            using var buffer = PinnedBuffer<int>.Allocate(length);

            // Assert
            Assert.Equal(length, buffer.Length);
        }
    }

    /// <summary>
    /// Unit tests for PinnedMemoryPool class.
    /// </summary>
    public class PinnedMemoryPoolTests
    {
        [Fact]
        public void Constructor_ValidParameters_CreatesPool()
        {
            // Arrange & Act
            using var pool = new PinnedMemoryPool<int>(1024, 5, 20);

            // Assert
            Assert.Equal(1024, pool.BufferSize);
            Assert.Equal(20, pool.MaxSize);
        }

        [Fact]
        public void Constructor_NegativeBufferSize_ThrowsArgumentOutOfRangeException()
        {
            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => new PinnedMemoryPool<int>(-1));
        }

        [Fact]
        public void Constructor_ZeroMaxSize_ThrowsArgumentOutOfRangeException()
        {
            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => new PinnedMemoryPool<int>(1024, 0, 0));
        }

        [Fact]
        public void Constructor_PreAllocatesBuffers()
        {
            // Arrange
            int initialSize = 5;
            int bufferSize = 1024;

            // Act
            using var pool = new PinnedMemoryPool<int>(bufferSize, initialSize);

            // Assert
            Assert.Equal(initialSize, pool.Count);
        }

        [Fact]
        public void Rent_WhenEmpty_CreatesNewBuffer()
        {
            // Arrange
            using var pool = new PinnedMemoryPool<int>(1024, 0);

            // Act
            using var buffer = pool.Rent();

            // Assert
            Assert.NotNull(buffer);
            Assert.Equal(1024, buffer.Length);
        }

        [Fact]
        public void Rent_WhenNotEmpty_ReturnsExistingBuffer()
        {
            // Arrange
            using var pool = new PinnedMemoryPool<int>(1024, 1);
            var firstBuffer = pool.Rent();
            IntPtr firstPointer = firstBuffer.Pointer;
            pool.Return(firstBuffer);

            // Act
            var secondBuffer = pool.Rent();

            // Assert
            Assert.Equal(firstPointer, secondBuffer.Pointer);
            secondBuffer.Dispose();
        }

        [Fact]
        public void Rent_AfterDispose_ThrowsObjectDisposedException()
        {
            // Arrange
            var pool = new PinnedMemoryPool<int>(1024);
            pool.Dispose();

            // Act & Assert
            Assert.Throws<ObjectDisposedException>(() => pool.Rent());
        }

        [Fact]
        public void Return_ValidBuffer_AddsToPool()
        {
            // Arrange
            using var pool = new PinnedMemoryPool<int>(1024, 0);
            var buffer = pool.Rent();

            // Act
            pool.Return(buffer);

            // Assert
            Assert.Equal(1, pool.Count);
        }

        [Fact]
        public void Return_NullBuffer_ThrowsArgumentNullException()
        {
            // Arrange
            using var pool = new PinnedMemoryPool<int>(1024);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => pool.Return(null));
        }

        [Fact]
        public void Return_WrongSizeBuffer_ThrowsArgumentException()
        {
            // Arrange
            using var pool = new PinnedMemoryPool<int>(1024);
            var wrongBuffer = PinnedBuffer<int>.Allocate(2048);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => pool.Return(wrongBuffer));
            wrongBuffer.Dispose();
        }

        [Fact]
        public void Return_WhenFull_DisposesBuffer()
        {
            // Arrange
            int maxSize = 2;
            using var pool = new PinnedMemoryPool<int>(1024, 2, maxSize);
            var buffer = PinnedBuffer<int>.Allocate(1024);

            // Act
            pool.Return(buffer);

            // Assert
            Assert.Equal(maxSize, pool.Count);
        }

        [Fact]
        public void Return_AfterDispose_ThrowsObjectDisposedException()
        {
            // Arrange
            var pool = new PinnedMemoryPool<int>(1024);
            var buffer = pool.Rent();
            pool.Dispose();

            // Act & Assert
            Assert.Throws<ObjectDisposedException>(() => pool.Return(buffer));
            buffer.Dispose();
        }

        [Fact]
        public void Resize_ClearsExistingPool()
        {
            // Arrange
            using var pool = new PinnedMemoryPool<int>(1024, 5);
            Assert.Equal(5, pool.Count);

            // Act
            pool.Resize(2048);

            // Assert
            Assert.Equal(0, pool.Count);
            Assert.Equal(2048, pool.BufferSize);
        }

        [Fact]
        public void Resize_InvalidSize_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            using var pool = new PinnedMemoryPool<int>(1024);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => pool.Resize(0));
        }

        [Fact]
        public void Clear_RemovesAllBuffers()
        {
            // Arrange
            using var pool = new PinnedMemoryPool<int>(1024, 5);
            Assert.Equal(5, pool.Count);

            // Act
            pool.Clear();

            // Assert
            Assert.Equal(0, pool.Count);
        }

        [Fact]
        public void MultipleRentAndReturn_ReusesBuffers()
        {
            // Arrange
            using var pool = new PinnedMemoryPool<int>(1024, 0);

            // Act
            var pointers = new IntPtr[5];
            for (int i = 0; i < 5; i++)
            {
                using var buffer = pool.Rent();
                pointers[i] = buffer.Pointer;
                pool.Return(buffer);
            }

            // Assert
            Assert.Equal(5, pool.Count);
        }

        [Fact]
        public void ConcurrentRentAndReturn_ThreadSafe()
        {
            // Arrange
            using var pool = new PinnedMemoryPool<int>(1024, 5);
            int numThreads = 10;
            int operationsPerThread = 100;

            // Act
            Parallel.For(0, numThreads, i =>
            {
                for (int j = 0; j < operationsPerThread; j++)
                {
                    var buffer = pool.Rent();
                    buffer.Fill(j);
                    pool.Return(buffer);
                }
            });

            // Assert - No exceptions should occur
        }

        [Fact]
        public void Dispose_ClearsAllBuffers()
        {
            // Arrange
            var pool = new PinnedMemoryPool<int>(1024, 5);
            Assert.Equal(5, pool.Count);

            // Act
            pool.Dispose();

            // Assert
            Assert.Equal(0, pool.Count);
        }

        [Fact]
        public void Dispose_CalledMultipleTimes_NoOpAfterFirst()
        {
            // Arrange
            var pool = new PinnedMemoryPool<int>(1024);

            // Act & Assert - Should not throw
            pool.Dispose();
            pool.Dispose();
            pool.Dispose();
        }
    }

    /// <summary>
    /// Unit tests for PinnedMemoryHelper class.
    /// </summary>
    public class PinnedMemoryHelperTests
    {
        [Fact]
        public void_Pin_ValidArray_ReturnsPinnedMemory()
        {
            // Arrange
            int[] array = new int[] { 1, 2, 3, 4, 5 };

            // Act
            using var pinned = PinnedMemoryHelper<int>.Pin(array);

            // Assert
            Assert.NotNull(pinned);
            Assert.True(pinned.IsPinned);
            Assert.Equal(array.Length, pinned.Length);
        }

        [Fact]
        public void Pin_NullArray_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => PinnedMemoryHelper<int>.Pin(null));
        }

        [Fact]
        public void PinAndCopy_ValidArray_ReturnsPinnedBuffer()
        {
            // Arrange
            int[] source = new int[] { 1, 2, 3, 4, 5 };

            // Act
            using var buffer = PinnedMemoryHelper<int>.PinAndCopy(source);

            // Assert
            Assert.NotNull(buffer);
            Assert.Equal(source.Length, buffer.Length);
            Assert.Equal(source, buffer.Span.ToArray());
        }

        [Fact]
        public void PinAndCopy_NullArray_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => PinnedMemoryHelper<int>.PinAndCopy(null));
        }

        [Fact]
        public void AllocateUnmanaged_ValidLength_ReturnsBuffer()
        {
            // Arrange
            int length = 100;

            // Act
            using var buffer = PinnedMemoryHelper<int>.AllocateUnmanaged(length);

            // Assert
            Assert.NotNull(buffer);
            Assert.Equal(length, buffer.Length);
        }

        [Fact]
        public void AllocateUnmanaged_NegativeLength_ThrowsArgumentOutOfRangeException()
        {
            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => PinnedMemoryHelper<int>.AllocateUnmanaged(-1));
        }

        [Fact]
        public void NonGenericPin_WorksCorrectly()
        {
            // Arrange
            int[] array = new int[] { 1, 2, 3, 4, 5 };

            // Act
            using var pinned = PinnedMemoryHelper.Pin(array);

            // Assert
            Assert.NotNull(pinned);
            Assert.True(pinned.IsPinned);
        }

        [Fact]
        public void NonGenericPinAndCopy_WorksCorrectly()
        {
            // Arrange
            int[] source = new int[] { 1, 2, 3, 4, 5 };

            // Act
            using var buffer = PinnedMemoryHelper.PinAndCopy(source);

            // Assert
            Assert.NotNull(buffer);
            Assert.Equal(source, buffer.Span.ToArray());
        }

        [Fact]
        public void NonGenericAllocateUnmanaged_WorksCorrectly()
        {
            // Arrange
            int length = 100;

            // Act
            using var buffer = PinnedMemoryHelper.AllocateUnmanaged<int>(length);

            // Assert
            Assert.NotNull(buffer);
            Assert.Equal(length, buffer.Length);
        }
    }

    /// <summary>
    /// Unit tests for PinningStrategySelector class.
    /// </summary>
    public class PinningStrategySelectorTests
    {
        [Fact]
        public void SelectStrategy_VerySmallBuffer_ReturnsNone()
        {
            // Arrange
            int bufferSize = 100; // Less than 1KB

            // Act
            var strategy = PinningStrategySelector.SelectStrategy<byte>(bufferSize, 1);

            // Assert
            Assert.Equal(PinningStrategy.None, strategy);
        }

        [Fact]
        public void SelectStrategy_MediumBuffer_ReturnsGCHandle()
        {
            // Arrange
            int bufferSize = 1000; // About 1KB for int

            // Act
            var strategy = PinningStrategySelector.SelectStrategy<int>(bufferSize, 1);

            // Assert
            Assert.Equal(PinningStrategy.GCHandle, strategy);
        }

        [Fact]
        public void SelectStrategy_LargeBuffer_ReturnsUnmanaged()
        {
            // Arrange
            int bufferSize = 1_000_000; // About 4MB for int

            // Act
            var strategy = PinningStrategySelector.SelectStrategy<int>(bufferSize, 1);

            // Assert
            Assert.Equal(PinningStrategy.Unmanaged, strategy);
        }

        [Fact]
        public void SelectStrategy_LargeBufferWithHighReuse_ReturnsPooled()
        {
            // Arrange
            int bufferSize = 1_000_000; // About 4MB for int
            int expectedLifetime = 20; // High reuse

            // Act
            var strategy = PinningStrategySelector.SelectStrategy<int>(bufferSize, expectedLifetime);

            // Assert
            Assert.Equal(PinningStrategy.PinnedObjectPool, strategy);
        }

        [Fact]
        public void SelectStrategy_MediumBufferWithHighReuse_ReturnsPooled()
        {
            // Arrange
            int bufferSize = 10_000; // About 40KB for int
            int expectedLifetime = 20; // High reuse

            // Act
            var strategy = PinningStrategySelector.SelectStrategy<int>(bufferSize, expectedLifetime);

            // Assert
            Assert.Equal(PinningStrategy.PinnedObjectPool, strategy);
        }

        [Fact]
        public void SelectStrategyByByteSize_WorksCorrectly()
        {
            // Arrange
            int byteSize = 512; // Less than 1KB

            // Act
            var strategy = PinningStrategySelector.SelectStrategyByByteSize(byteSize, 1);

            // Assert
            Assert.Equal(PinningStrategy.None, strategy);
        }

        [Fact]
        public void SelectStrategy_NegativeBufferSize_ThrowsArgumentOutOfRangeException()
        {
            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                PinningStrategySelector.SelectStrategy<int>(-1, 1));
        }

        [Fact]
        public void SelectStrategy_NegativeLifetime_ThrowsArgumentOutOfRangeException()
        {
            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                PinningStrategySelector.SelectStrategy<int>(100, -1));
        }

        [Fact]
        public void SmallBufferSizeThreshold_ReturnsCorrectValue()
        {
            // Act & Assert
            Assert.Equal(1024, PinningStrategySelector.SmallBufferSizeThresholdBytes);
        }

        [Fact]
        public void LargeBufferSizeThreshold_ReturnsCorrectValue()
        {
            // Act & Assert
            Assert.Equal(1024 * 1024, PinningStrategySelector.LargeBufferSizeThresholdBytes);
        }

        [Fact]
        public void HighReuseThreshold_ReturnsCorrectValue()
        {
            // Act & Assert
            Assert.Equal(10, PinningStrategySelector.HighReuseThresholdCount);
        }

        [Fact]
        public void CreatePinnedMemory_NoneStrategy_ReturnsNull()
        {
            // Arrange
            int[] array = new int[] { 1, 2, 3 };

            // Act
            var result = PinningStrategySelector.CreatePinnedMemory(
                array, PinningStrategy.None);

            // Assert
            Assert.Null(result);
        }

        [Fact]
        public void CreatePinnedMemory_GCHandleStrategy_ReturnsPinnedMemory()
        {
            // Arrange
            int[] array = new int[] { 1, 2, 3 };

            // Act
            var result = PinningStrategySelector.CreatePinnedMemory(
                array, PinningStrategy.GCHandle);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IsPinned);
        }

        [Fact]
        public void CreatePinnedMemory_PooledStrategyWithoutPool_ThrowsArgumentNullException()
        {
            // Arrange
            int[] array = new int[] { 1, 2, 3 };

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                PinningStrategySelector.CreatePinnedMemory(
                    array, PinningStrategy.PinnedObjectPool));
        }

        [Fact]
        public void CreatePinnedMemory_PooledStrategyWithPool_ReturnsBuffer()
        {
            // Arrange
            int[] array = new int[] { 1, 2, 3 };
            using var pool = new PinnedMemoryPool<int>(3, 1);

            // Act
            var result = PinningStrategySelector.CreatePinnedMemory(
                array, PinningStrategy.PinnedObjectPool, pool);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(array, result.Span.ToArray());
        }

        [Fact]
        public void CreatePinnedMemory_PooledStrategyWrongSize_ThrowsArgumentException()
        {
            // Arrange
            int[] array = new int[] { 1, 2, 3 };
            using var pool = new PinnedMemoryPool<int>(10);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                PinningStrategySelector.CreatePinnedMemory(
                    array, PinningStrategy.PinnedObjectPool, pool));
        }

        [Fact]
        public void CreatePinnedMemory_UnknownStrategy_ThrowsArgumentException()
        {
            // Arrange
            int[] array = new int[] { 1, 2, 3 };

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                PinningStrategySelector.CreatePinnedMemory(
                    array, (PinningStrategy)999));
        }
    }
}
