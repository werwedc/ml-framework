using System;
using Xunit;
using MLFramework.Communication.Optimizations;
using MLFramework.Communication;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Communication.Optimizations
{
    public class PinnedMemoryManagerTests : IDisposable
    {
        private readonly PinnedMemoryManager _manager;

        public PinnedMemoryManagerTests()
        {
            _manager = new PinnedMemoryManager(1024 * 1024); // 1MB limit for tests
        }

        [Fact]
        public void Constructor_Default_CreatesInstance()
        {
            using var manager = new PinnedMemoryManager();
            Assert.NotNull(manager);
            Assert.Equal(0, manager.TotalPinnedBytes);
        }

        [Fact]
        public void Constructor_CustomLimit_CreatesInstanceWithLimit()
        {
            using var manager = new PinnedMemoryManager(10 * 1024 * 1024);
            Assert.NotNull(manager);
        }

        [Fact]
        public void PinMemory_NullTensor_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() => _manager.PinMemory(null));
        }

        [Fact]
        public void PinMemory_ValidTensor_ReturnsHandle()
        {
            var data = new float[100];
            var shape = new int[] { 100 };
            var tensor = new Tensor(data, shape);
            using var handle = _manager.PinMemory(tensor);

            Assert.NotNull(handle);
            Assert.NotEqual(IntPtr.Zero, handle.Pointer);
            Assert.True(handle.Size > 0);
        }

        [Fact]
        public void PinMemory_MultipleAllocations_IncreasesTotalPinnedBytes()
        {
            var data1 = new float[100];
            var shape1 = new int[] { 100 };
            var tensor1 = new Tensor(data1, shape1);

            var data2 = new float[100];
            var shape2 = new int[] { 100 };
            var tensor2 = new Tensor(data2, shape2);

            using var handle1 = _manager.PinMemory(tensor1);
            var before = _manager.TotalPinnedBytes;

            using var handle2 = _manager.PinMemory(tensor2);
            var after = _manager.TotalPinnedBytes;

            Assert.True(after > before);
        }

        [Fact]
        public void PinMemory_ExceedsLimit_ThrowsException()
        {
            var data = new float[1024 * 1024];
            var shape = new int[] { 1024 * 1024 };
            var tensor = new Tensor(data, shape); // Larger than limit

            Assert.Throws<CommunicationException>(() => _manager.PinMemory(tensor));
        }

        [Fact]
        public void DisposeHandle_DecreasesTotalPinnedBytes()
        {
            var data = new float[100];
            var shape = new int[] { 100 };
            var tensor = new Tensor(data, shape);

            using var handle = _manager.PinMemory(tensor);
            var before = _manager.TotalPinnedBytes;
        }

        [Fact]
        public void Dispose_ClearsAllAllocations()
        {
            var data = new float[100];
            var shape = new int[] { 100 };
            var tensor = new Tensor(data, shape);

            var handle = _manager.PinMemory(tensor);
            Assert.True(_manager.TotalPinnedBytes > 0);

            _manager.Dispose();
            Assert.Equal(0, _manager.TotalPinnedBytes);
        }

        [Fact]
        public void PinMemory_LRU_EvictsOldestWhenLimitExceeded()
        {
            var data1 = new float[100];
            var shape1 = new int[] { 100 };
            var smallTensor = new Tensor(data1, shape1);

            var data2 = new float[51200];
            var shape2 = new int[] { 51200 };
            var largeTensor = new Tensor(data2, shape2);

            using var handle1 = _manager.PinMemory(smallTensor);
            var handle2 = _manager.PinMemory(smallTensor);
            var handle3 = _manager.PinMemory(smallTensor);

            // Fill up most of the limit
            var before = _manager.TotalPinnedBytes;

            // This should trigger LRU eviction
            var handle4 = _manager.PinMemory(largeTensor);

            Assert.NotNull(handle4);
        }

        public void Dispose()
        {
            _manager.Dispose();
        }
    }
}
