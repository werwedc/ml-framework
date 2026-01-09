using MLFramework.MobileRuntime;
using MLFramework.MobileRuntime.Backends.Cpu;
using MLFramework.MobileRuntime.Backends.Cpu.Interfaces;
using MLFramework.MobileRuntime.Backends.Cpu.Models;
using MLFramework.MobileRuntime.Memory;

namespace MobileRuntime.Tests.Backends.Cpu
{
    using System;
    using Xunit;
    using System.Collections.Generic;

    /// <summary>
    /// Tests for the CPU backend implementation.
    /// </summary>
    public class CpuBackendTests
    {
        private readonly IMemoryPool _memoryPool;
        private readonly ITensorFactory _tensorFactory;

        public CpuBackendTests()
        {
            // Create a simple memory pool for testing
            _memoryPool = MemoryPoolFactory.CreateDefault();
            _tensorFactory = new MockTensorFactory();
        }

        [Fact]
        public void CreateDefault_BackendIsNotNull()
        {
            // Arrange & Act
            var backend = CpuBackendFactory.CreateDefault(_memoryPool, _tensorFactory);

            // Assert
            Assert.NotNull(backend);
            Assert.Equal("CPU", backend.Name);
        }

        [Fact]
        public void GetCpuInfo_ReturnsValidInfo()
        {
            // Arrange
            var backend = CpuBackendFactory.CreateDefault(_memoryPool, _tensorFactory);

            // Act
            var cpuInfo = backend.GetCpuInfo();

            // Assert
            Assert.NotNull(cpuInfo);
            Assert.NotEmpty(cpuInfo.Vendor);
            Assert.True(cpuInfo.CoreCount > 0);
            Assert.NotNull(cpuInfo.Capabilities);
            Assert.NotNull(cpuInfo.Capabilities.MaxThreads);
        }

        [Fact]
        public void EnableVectorization_DoesNotThrow()
        {
            // Arrange
            var backend = CpuBackendFactory.CreateDefault(_memoryPool, _tensorFactory);

            // Act & Assert - Should not throw
            backend.EnableVectorization(false);
            backend.EnableVectorization(true);
        }

        [Fact]
        public void EnableMultiThreading_DoesNotThrow()
        {
            // Arrange
            var backend = CpuBackendFactory.CreateDefault(_memoryPool, _tensorFactory);

            // Act & Assert - Should not throw
            backend.EnableMultiThreading(false);
            backend.EnableMultiThreading(true, 4);
        }

        [Fact]
        public void Execute_UnsupportedOperator_ThrowsException()
        {
            // Arrange
            var backend = CpuBackendFactory.CreateDefault(_memoryPool, _tensorFactory);
            var op = new OperatorDescriptor
            {
                Type = OperatorType.Sigmoid, // Not implemented yet
                Id = 1
            };

            // Act & Assert
            Assert.Throws<NotSupportedException>(() =>
                backend.Execute(op, Array.Empty<ITensor>(), new Dictionary<string, object>()));
        }

        [Fact]
        public void Capabilities_ContainsValidData()
        {
            // Arrange
            var backend = CpuBackendFactory.CreateDefault(_memoryPool, _tensorFactory);

            // Act
            var caps = backend.Capabilities;

            // Assert
            Assert.NotNull(caps);
            Assert.True(caps.MaxThreads > 0);
            Assert.True(caps.CacheLineSize > 0);
        }

        [Fact]
        public void CreateWithNeonOptimization_CreatesValidBackend()
        {
            // Arrange & Act
            var backend = CpuBackendFactory.CreateWithNeonOptimization(_memoryPool, _tensorFactory);

            // Assert
            Assert.NotNull(backend);
            Assert.Equal("CPU", backend.Name);
        }

        [Fact]
        public void CreateForX86_CreatesValidBackend()
        {
            // Arrange & Act
            var backend = CpuBackendFactory.CreateForX86(_memoryPool, _tensorFactory);

            // Assert
            Assert.NotNull(backend);
            Assert.Equal("CPU", backend.Name);
        }

        [Fact]
        public void CreateCustom_WithAllParameters_CreatesValidBackend()
        {
            // Arrange & Act
            var backend = CpuBackendFactory.CreateCustom(
                _memoryPool,
                _tensorFactory,
                enableVectorization: false,
                enableMultiThreading: false,
                maxThreads: 2);

            // Assert
            Assert.NotNull(backend);
            Assert.Equal("CPU", backend.Name);
        }
    }

    /// <summary>
    /// Mock tensor factory for testing purposes.
    /// </summary>
    internal class MockTensorFactory : ITensorFactory
    {
        public ITensor Create(int[] shape, DataType dataType)
        {
            // Create a simple mock tensor
            return new MockTensor(shape, dataType);
        }

        public ITensor FromArray<T>(T[] data, int[] shape)
        {
            return new MockTensor(shape, DataType.Float32);
        }
    }

    /// <summary>
    /// Mock tensor for testing purposes.
    /// </summary>
    internal class MockTensor : ITensor
    {
        private readonly float[] _data;

        public MockTensor(int[] shape, DataType dataType)
        {
            Shape = shape;
            DataType = dataType;
            Size = 1;
            foreach (int dim in shape)
            {
                Size *= dim;
            }
            ByteCount = Size * sizeof(float);
            _data = new float[Size];
        }

        public int[] Shape { get; }
        public DataType DataType { get; }
        public long Size { get; }
        public long ByteCount { get; }

        public T GetData<T>(params int[] indices)
        {
            int index = 0;
            for (int i = 0; i < indices.Length; i++)
            {
                index = index * Shape[i] + indices[i];
            }
            return (T)(object)_data[index];
        }

        public T[] ToArray<T>()
        {
            if (typeof(T) == typeof(float))
            {
                return (T[])(object)_data;
            }
            throw new NotSupportedException($"Type {typeof(T)} is not supported.");
        }

        public void Dispose()
        {
            // No-op for mock
        }
    }
}
