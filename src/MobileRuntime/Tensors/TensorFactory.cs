using System;
using MobileRuntime.Interfaces;
using MobileRuntime.Memory;

namespace MobileRuntime
{
    /// <summary>
    /// Factory class for creating tensors with memory pool support
    /// </summary>
    public class TensorFactory
    {
        private readonly IMemoryPool _memoryPool;

        /// <summary>
        /// Creates a new TensorFactory with a default memory pool
        /// </summary>
        public TensorFactory() : this(new DefaultMemoryPool())
        {
        }

        /// <summary>
        /// Creates a new TensorFactory with the specified memory pool
        /// </summary>
        public TensorFactory(IMemoryPool memoryPool)
        {
            _memoryPool = memoryPool ?? throw new ArgumentNullException(nameof(memoryPool));
        }

        /// <summary>
        /// Creates an empty tensor with the given shape and data type
        /// </summary>
        public Tensor CreateTensor(int[] shape, DataType dataType)
        {
            long size = CalculateSize(shape) * GetDataTypeSize(dataType);
            IntPtr data = _memoryPool.Allocate(size, dataType);
            return new Tensor(shape, dataType, data, true);
        }

        /// <summary>
        /// Creates a tensor from an array
        /// </summary>
        public Tensor CreateTensor<T>(T[] data, int[] shape) where T : struct
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));
            if (shape == null)
                throw new ArgumentNullException(nameof(shape));

            long expectedSize = CalculateSize(shape);
            if (data.Length != expectedSize)
                throw new ArgumentException($"Data length {data.Length} does not match shape {expectedSize}");

            DataType dataType = GetDataType<T>();
            long byteSize = expectedSize * GetDataTypeSize(dataType);
            IntPtr dataPtr = _memoryPool.Allocate(byteSize, dataType);

            // Copy data to allocated memory
            unsafe
            {
                if (typeof(T) == typeof(float))
                {
                    float* floatPtr = (float*)dataPtr;
                    float[] floatData = (float[])(object)data;
                    for (long i = 0; i < expectedSize; i++)
                    {
                        floatPtr[i] = floatData[i];
                    }
                }
                else if (typeof(T) == typeof(int))
                {
                    int* intPtr = (int*)dataPtr;
                    int[] intData = (int[])(object)data;
                    for (long i = 0; i < expectedSize; i++)
                    {
                        intPtr[i] = intData[i];
                    }
                }
                else
                {
                    // Generic copy
                    System.Runtime.InteropServices.Marshal.Copy(
                        System.Runtime.InteropServices.Marshal.UnsafeAddrOfPinnedArrayElement(data, 0),
                        new byte[byteSize], 0, (int)byteSize);
                }
            }

            return new Tensor(shape, dataType, dataPtr, true);
        }

        /// <summary>
        /// Creates a view tensor that references existing memory
        /// </summary>
        public Tensor CreateView(IntPtr data, int[] shape, DataType dataType)
        {
            if (data == IntPtr.Zero)
                throw new ArgumentException("Data pointer cannot be zero");
            if (shape == null)
                throw new ArgumentNullException(nameof(shape));

            return new Tensor(shape, dataType, data, false);
        }

        /// <summary>
        /// Gets the memory pool used by this factory
        /// </summary>
        public IMemoryPool MemoryPool => _memoryPool;

        #region Private Helpers

        private static long CalculateSize(int[] shape)
        {
            if (shape == null || shape.Length == 0)
                return 0;

            long size = 1;
            foreach (int dim in shape)
            {
                size *= dim;
            }
            return size;
        }

        private static int GetDataTypeSize(DataType dataType)
        {
            return dataType switch
            {
                DataType.Float32 => 4,
                DataType.Float16 => 2,
                DataType.Int8 => 1,
                DataType.Int16 => 2,
                DataType.Int32 => 4,
                _ => throw new ArgumentException($"Unknown data type: {dataType}")
            };
        }

        private static DataType GetDataType<T>() where T : struct
        {
            Type type = typeof(T);
            if (type == typeof(float) || type == typeof(double))
                return DataType.Float32;
            else if (type == typeof(sbyte) || type == typeof(byte))
                return DataType.Int8;
            else if (type == typeof(short) || type == typeof(ushort))
                return DataType.Int16;
            else if (type == typeof(int) || type == typeof(uint))
                return DataType.Int32;
            else
                throw new ArgumentException($"Unsupported type: {type}");
        }

        #endregion
    }
}
