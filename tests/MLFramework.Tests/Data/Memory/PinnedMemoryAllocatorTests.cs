using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using MLFramework.Data.Memory;
using Xunit;

namespace MLFramework.Tests.Data.Memory
{
    /// <summary>
    /// Unit tests for PinnedMemoryAllocator class.
    /// </summary>
    public class PinnedMemoryAllocatorTests
    {
        [Fact]
        public void Constructor_InitializesCorrectly()
        {
            // Arrange & Act
            using var allocator = new PinnedMemoryAllocator();

            // Assert
            Assert.True(allocator.IsPinnedMemorySupported);
        }

        [Fact]
        public void Constructor_WithForcePinnedFalse_InitializesWithDetection()
        {
            // Arrange & Act
            using var allocator = new PinnedMemoryAllocator(forcePinned: false);

            // Assert
            // Should detect CUDA support (currently returns true in placeholder)
            Assert.True(allocator.IsPinnedMemorySupported);
        }

        [Fact]
        public void Allocate_ValidSize_ReturnsValidPointer()
        {
            // Arrange
            using var allocator = new PinnedMemoryAllocator();
            int size = 1024;

            // Act
            IntPtr pointer = allocator.Allocate(size);

            // Assert
            Assert.NotEqual(IntPtr.Zero, pointer);
            Assert.NotEqual(-1, pointer.ToInt64());

            // Cleanup
            allocator.Free(pointer);
        }

        [Fact]
        public void Allocate_AlreadyDisposed_ThrowsObjectDisposedException()
        {
            // Arrange
            var allocator = new PinnedMemoryAllocator();
            allocator.Dispose();

            // Act & Assert
            Assert.Throws<ObjectDisposedException>(() => allocator.Allocate(1024));
        }

        [Fact]
        public void Allocate_NegativeSize_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            using var allocator = new PinnedMemoryAllocator();

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => allocator.Allocate(-1));
        }

        [Fact]
        public void Allocate_ZeroSize_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            using var allocator = new PinnedMemoryAllocator();

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => allocator.Allocate(0));
        }

        [Fact]
        public void Free_ValidPointer_FreesMemorySuccessfully()
        {
            // Arrange
            using var allocator = new PinnedMemoryAllocator();
            IntPtr pointer = allocator.Allocate(1024);

            // Act - Should not throw
            allocator.Free(pointer);

            // Assert - No exception thrown
        }

        [Fact]
        public void Free_AlreadyDisposed_ThrowsObjectDisposedException()
        {
            // Arrange
            var allocator = new PinnedMemoryAllocator();
            IntPtr pointer = allocator.Allocate(1024);
            allocator.Dispose();

            // Act & Assert
            Assert.Throws<ObjectDisposedException>(() => allocator.Free(pointer));
        }

        [Fact]
        public void Free_NullPointer_ThrowsArgumentException()
        {
            // Arrange
            using var allocator = new PinnedMemoryAllocator();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => allocator.Free(IntPtr.Zero));
        }

        [Fact]
        public void Free_PointerNotAllocatedByThisAllocator_ThrowsArgumentException()
        {
            // Arrange
            using var allocator = new PinnedMemoryAllocator();
            IntPtr externalPointer = Marshal.AllocHGlobal(1024);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => allocator.Free(externalPointer));

            // Cleanup
            Marshal.FreeHGlobal(externalPointer);
        }

        [Fact]
        public void Free_SamePointerTwice_ThrowsArgumentException()
        {
            // Arrange
            using var allocator = new PinnedMemoryAllocator();
            IntPtr pointer = allocator.Allocate(1024);
            allocator.Free(pointer);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => allocator.Free(pointer));
        }

        [Fact]
        public void MultipleAllocateAndFree_TracksAllocationsCorrectly()
        {
            // Arrange
            using var allocator = new PinnedMemoryAllocator();
            int numAllocations = 10;
            IntPtr[] pointers = new IntPtr[numAllocations];

            // Act
            for (int i = 0; i < numAllocations; i++)
            {
                pointers[i] = allocator.Allocate(1024);
            }

            // Assert - All pointers should be valid
            foreach (var ptr in pointers)
            {
                Assert.NotEqual(IntPtr.Zero, ptr);
            }

            // Cleanup
            for (int i = 0; i < numAllocations; i++)
            {
                allocator.Free(pointers[i]);
            }
        }

        [Fact]
        public void Dispose_FreesAllAllocatedMemory()
        {
            // Arrange
            var allocator = new PinnedMemoryAllocator();
            IntPtr pointer1 = allocator.Allocate(1024);
            IntPtr pointer2 = allocator.Allocate(2048);
            IntPtr pointer3 = allocator.Allocate(4096);

            // Act - Should not throw
            allocator.Dispose();

            // Assert - All memory should be freed
            // Note: We can't directly verify memory is freed, but the allocator
            // should have cleared its internal tracking
        }

        [Fact]
        public void Dispose_AlreadyDisposed_DoesNotThrow()
        {
            // Arrange
            var allocator = new PinnedMemoryAllocator();
            allocator.Dispose();

            // Act & Assert - Should not throw
            allocator.Dispose();
        }

        [Fact]
        public void CopyToPinnedMemory_ValidData_CopiesSuccessfully()
        {
            // Arrange
            using var allocator = new PinnedMemoryAllocator();
            IntPtr pointer = allocator.Allocate(1024);
            byte[] data = new byte[] { 1, 2, 3, 4, 5 };

            // Act - Should not throw
            PinnedMemoryAllocator.CopyToPinnedMemory(pointer, data);

            // Assert
            byte[] result = new byte[data.Length];
            Marshal.Copy(pointer, result, 0, data.Length);
            Assert.Equal(data, result);

            // Cleanup
            allocator.Free(pointer);
        }

        [Fact]
        public void CopyToPinnedMemory_WithOffset_CopiesSuccessfully()
        {
            // Arrange
            using var allocator = new PinnedMemoryAllocator();
            IntPtr pointer = allocator.Allocate(1024);
            byte[] data = new byte[] { 0, 0, 1, 2, 3 };
            byte[] toCopy = new byte[] { 1, 2, 3 };

            // Act
            PinnedMemoryAllocator.CopyToPinnedMemory(pointer, toCopy, 2);

            // Assert
            byte[] result = new byte[data.Length];
            Marshal.Copy(pointer, result, 0, data.Length);
            Assert.Equal(data, result);

            // Cleanup
            allocator.Free(pointer);
        }

        [Fact]
        public void CopyToPinnedMemory_WithLength_CopiesSuccessfully()
        {
            // Arrange
            using var allocator = new PinnedMemoryAllocator();
            IntPtr pointer = allocator.Allocate(1024);
            byte[] data = new byte[] { 1, 2, 3, 0, 0 };
            byte[] toCopy = new byte[] { 1, 2, 3, 4, 5 };

            // Act
            PinnedMemoryAllocator.CopyToPinnedMemory(pointer, toCopy, 0, 3);

            // Assert
            byte[] result = new byte[data.Length];
            Marshal.Copy(pointer, result, 0, data.Length);
            Assert.Equal(data, result);

            // Cleanup
            allocator.Free(pointer);
        }

        [Fact]
        public void CopyToPinnedMemory_NullPointer_ThrowsArgumentException()
        {
            // Arrange
            byte[] data = new byte[] { 1, 2, 3 };

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                PinnedMemoryAllocator.CopyToPinnedMemory(IntPtr.Zero, data));
        }

        [Fact]
        public void CopyToPinnedMemory_NullData_ThrowsArgumentNullException()
        {
            // Arrange
            using var allocator = new PinnedMemoryAllocator();
            IntPtr pointer = allocator.Allocate(1024);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                PinnedMemoryAllocator.CopyToPinnedMemory(pointer, null));

            // Cleanup
            allocator.Free(pointer);
        }

        [Fact]
        public void CopyToPinnedMemory_NegativeOffset_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            using var allocator = new PinnedMemoryAllocator();
            IntPtr pointer = allocator.Allocate(1024);
            byte[] data = new byte[] { 1, 2, 3 };

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                PinnedMemoryAllocator.CopyToPinnedMemory(pointer, data, -1));

            // Cleanup
            allocator.Free(pointer);
        }

        [Fact]
        public void CopyToPinnedMemory_OffsetOutOfRange_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            using var allocator = new PinnedMemoryAllocator();
            IntPtr pointer = allocator.Allocate(1024);
            byte[] data = new byte[] { 1, 2, 3 };

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                PinnedMemoryAllocator.CopyToPinnedMemory(pointer, data, 10));

            // Cleanup
            allocator.Free(pointer);
        }

        [Fact]
        public void CopyToPinnedMemory_LengthOutOfRange_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            using var allocator = new PinnedMemoryAllocator();
            IntPtr pointer = allocator.Allocate(1024);
            byte[] data = new byte[] { 1, 2, 3 };

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                PinnedMemoryAllocator.CopyToPinnedMemory(pointer, data, 0, 10));

            // Cleanup
            allocator.Free(pointer);
        }

        [Fact]
        public void CopyFromPinnedMemory_ValidData_CopiesSuccessfully()
        {
            // Arrange
            using var allocator = new PinnedMemoryAllocator();
            IntPtr pointer = allocator.Allocate(1024);
            byte[] sourceData = new byte[] { 1, 2, 3, 4, 5 };
            PinnedMemoryAllocator.CopyToPinnedMemory(pointer, sourceData);
            byte[] destData = new byte[sourceData.Length];

            // Act
            PinnedMemoryAllocator.CopyFromPinnedMemory(pointer, destData);

            // Assert
            Assert.Equal(sourceData, destData);

            // Cleanup
            allocator.Free(pointer);
        }

        [Fact]
        public void CopyFromPinnedMemory_WithOffsetAndLength_CopiesSuccessfully()
        {
            // Arrange
            using var allocator = new PinnedMemoryAllocator();
            IntPtr pointer = allocator.Allocate(1024);
            byte[] sourceData = new byte[] { 0, 1, 2, 3, 4, 5 };
            PinnedMemoryAllocator.CopyToPinnedMemory(pointer, sourceData);
            byte[] destData = new byte[] { 99, 99, 0, 0, 0 };

            // Act
            PinnedMemoryAllocator.CopyFromPinnedMemory(pointer, destData, 2, 3);

            // Assert
            Assert.Equal(new byte[] { 99, 99, 1, 2, 3 }, destData);

            // Cleanup
            allocator.Free(pointer);
        }

        [Fact]
        public void CopyFromPinnedMemory_NullPointer_ThrowsArgumentException()
        {
            // Arrange
            byte[] data = new byte[10];

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                PinnedMemoryAllocator.CopyFromPinnedMemory(IntPtr.Zero, data));
        }

        [Fact]
        public void CopyFromPinnedMemory_NullData_ThrowsArgumentNullException()
        {
            // Arrange
            using var allocator = new PinnedMemoryAllocator();
            IntPtr pointer = allocator.Allocate(1024);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                PinnedMemoryAllocator.CopyFromPinnedMemory(pointer, null));

            // Cleanup
            allocator.Free(pointer);
        }

        [Fact]
        public void LargeAllocation_Succeeds()
        {
            // Arrange
            using var allocator = new PinnedMemoryAllocator();
            int size = 10 * 1024 * 1024; // 10 MB

            // Act
            IntPtr pointer = allocator.Allocate(size);

            // Assert
            Assert.NotEqual(IntPtr.Zero, pointer);

            // Cleanup
            allocator.Free(pointer);
        }

        [Fact]
        public void ConcurrentAllocateAndFree_ThreadSafe()
        {
            // Arrange
            using var allocator = new PinnedMemoryAllocator();
            int numThreads = 10;
            int operationsPerThread = 100;
            var exceptions = new System.Collections.Concurrent.ConcurrentBag<Exception>();

            // Act
            Parallel.For(0, numThreads, threadIndex =>
            {
                try
                {
                    IntPtr[] pointers = new IntPtr[operationsPerThread];
                    for (int i = 0; i < operationsPerThread; i++)
                    {
                        pointers[i] = allocator.Allocate(1024);
                    }

                    for (int i = 0; i < operationsPerThread; i++)
                    {
                        allocator.Free(pointers[i]);
                    }
                }
                catch (Exception ex)
                {
                    exceptions.Add(ex);
                }
            });

            // Assert
            Assert.Empty(exceptions);
        }

        [Fact]
        public void UsingStatement_CleansUpCorrectly()
        {
            // Arrange & Act
            IntPtr pointer;
            using (var allocator = new PinnedMemoryAllocator())
            {
                pointer = allocator.Allocate(1024);
                Assert.NotEqual(IntPtr.Zero, pointer);
            }

            // Assert - Allocator should be disposed and memory freed
            // (We can't directly verify, but no exception should occur)
        }
    }
}
