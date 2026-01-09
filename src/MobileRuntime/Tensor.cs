using System;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace MobileRuntime
{
    /// <summary>
    /// Lightweight tensor implementation optimized for mobile/edge devices without gradient computation
    /// </summary>
    public sealed class Tensor : ITensor
    {
        private IntPtr _data;
        private int[] _shape;
        private DataType _dataType;
        private bool _disposed;
        private bool _ownData;

        /// <summary>
        /// Initializes a new instance of the Tensor class
        /// </summary>
        public Tensor(int[] shape, DataType dataType)
        {
            _shape = shape ?? throw new ArgumentNullException(nameof(shape));
            _dataType = dataType;
            _ownData = true;
            _data = Marshal.AllocHGlobal((int)ByteCount);
            InitializeData();
        }

        /// <summary>
        /// Initializes a new instance of the Tensor class with pre-allocated data
        /// </summary>
        public Tensor(int[] shape, DataType dataType, IntPtr data, bool ownData)
        {
            _shape = shape ?? throw new ArgumentNullException(nameof(shape));
            _dataType = dataType;
            _data = data;
            _ownData = ownData;
        }

        ~Tensor()
        {
            Dispose(false);
        }

        /// <summary>
        /// Gets the tensor shape
        /// </summary>
        public int[] Shape => _shape;

        /// <summary>
        /// Gets the data type
        /// </summary>
        public DataType DataType => _dataType;

        /// <summary>
        /// Gets the total number of elements
        /// </summary>
        public long Size => _shape.Aggregate(1L, (a, b) => a * b);

        /// <summary>
        /// Gets the total number of bytes
        /// </summary>
        public long ByteCount => Size * GetDataTypeSize(_dataType);

        /// <summary>
        /// Gets the raw data pointer
        /// </summary>
        public IntPtr DataPointer => _data;

        /// <summary>
        /// Gets or sets tensor data (legacy property for backward compatibility)
        /// </summary>
        [Obsolete("Use DataPointer and ToArray() instead")]
        public float[] Data
        {
            get => ToArray<float>();
            set
            {
                if (value == null)
                    throw new ArgumentNullException(nameof(value));
                if (value.Length != Size)
                    throw new ArgumentException($"Data length {value.Length} does not match tensor size {Size}");

                unsafe
                {
                    float* ptr = (float*)_data.ToPointer();
                    for (long i = 0; i < Size; i++)
                    {
                        ptr[i] = value[i];
                    }
                }
            }
        }

        /// <summary>
        /// Gets the total number of elements (legacy property)
        /// </summary>
        [Obsolete("Use Size instead")]
        public int Length => (int)Size;

        /// <summary>
        /// Gets data at specified indices
        /// </summary>
        public T GetData<T>(params int[] indices) where T : struct
        {
            int offset = CalculateOffset(indices);
            return Marshal.PtrToStructure<T>(IntPtr.Add(_data, offset));
        }

        /// <summary>
        /// Converts tensor data to array
        /// </summary>
        public T[] ToArray<T>() where T : struct
        {
            int elementSize = Marshal.SizeOf<T>();
            T[] result = new T[Size];

            for (long i = 0; i < Size; i++)
            {
                result[i] = Marshal.PtrToStructure<T>(IntPtr.Add(_data, (int)i * elementSize));
            }

            return result;
        }

        /// <summary>
        /// Adds a scalar value in-place
        /// </summary>
        public void AddScalar<T>(T scalar) where T : struct
        {
            unsafe
            {
                void* ptr = _data.ToPointer();
                long count = Size;

                if (typeof(T) == typeof(float))
                {
                    float* floatPtr = (float*)ptr;
                    float val = (float)(object)scalar;
                    for (long i = 0; i < count; i++)
                    {
                        floatPtr[i] += val;
                    }
                }
                else if (typeof(T) == typeof(double))
                {
                    double* doublePtr = (double*)ptr;
                    double val = (double)(object)scalar;
                    for (long i = 0; i < count; i++)
                    {
                        doublePtr[i] += val;
                    }
                }
            }
        }

        /// <summary>
        /// Multiplies by a scalar value in-place
        /// </summary>
        public void MultiplyScalar<T>(T scalar) where T : struct
        {
            unsafe
            {
                void* ptr = _data.ToPointer();
                long count = Size;

                if (typeof(T) == typeof(float))
                {
                    float* floatPtr = (float*)ptr;
                    float val = (float)(object)scalar;
                    for (long i = 0; i < count; i++)
                    {
                        floatPtr[i] *= val;
                    }
                }
                else if (typeof(T) == typeof(double))
                {
                    double* doublePtr = (double*)ptr;
                    double val = (double)(object)scalar;
                    for (long i = 0; i < count; i++)
                    {
                        doublePtr[i] *= val;
                    }
                }
            }
        }

        /// <summary>
        /// Clamps values between min and max in-place
        /// </summary>
        public void Clamp<T>(T min, T max) where T : struct, IComparable
        {
            unsafe
            {
                void* ptr = _data.ToPointer();
                long count = Size;
                IComparable minVal = min;
                IComparable maxVal = max;

                if (typeof(T) == typeof(float))
                {
                    float* floatPtr = (float*)ptr;
                    float minFloat = (float)(object)min;
                    float maxFloat = (float)(object)max;
                    for (long i = 0; i < count; i++)
                    {
                        if (floatPtr[i] < minFloat)
                            floatPtr[i] = minFloat;
                        else if (floatPtr[i] > maxFloat)
                            floatPtr[i] = maxFloat;
                    }
                }
            }
        }

        /// <summary>
        /// Creates a tensor filled with zeros
        /// </summary>
        public static Tensor Zeros(int[] shape, DataType dataType)
        {
            var tensor = new Tensor(shape, dataType);
            unsafe
            {
                void* ptr = tensor._data.ToPointer();
                long bytes = tensor.ByteCount;
                byte* bytePtr = (byte*)ptr;
                for (long i = 0; i < bytes; i++)
                {
                    bytePtr[i] = 0;
                }
            }
            return tensor;
        }

        /// <summary>
        /// Creates a tensor filled with ones
        /// </summary>
        public static Tensor Ones(int[] shape, DataType dataType)
        {
            var tensor = new Tensor(shape, dataType);
            unsafe
            {
                void* ptr = tensor._data.ToPointer();
                long count = tensor.Size;

                switch (dataType)
                {
                    case DataType.Float32:
                        {
                            float* floatPtr = (float*)ptr;
                            for (long i = 0; i < count; i++)
                            {
                                floatPtr[i] = 1.0f;
                            }
                            break;
                        }
                    case DataType.Float16:
                        {
                            short* shortPtr = (short*)ptr;
                            // FP16 1.0 = 0x3C00
                            for (long i = 0; i < count; i++)
                            {
                                shortPtr[i] = 0x3C00;
                            }
                            break;
                        }
                    case DataType.Int8:
                        {
                            sbyte* sbytePtr = (sbyte*)ptr;
                            for (long i = 0; i < count; i++)
                            {
                                sbytePtr[i] = 1;
                            }
                            break;
                        }
                    case DataType.Int16:
                        {
                            short* shortPtr = (short*)ptr;
                            for (long i = 0; i < count; i++)
                            {
                                shortPtr[i] = 1;
                            }
                            break;
                        }
                }
            }
            return tensor;
        }

        /// <summary>
        /// Creates a tensor from an array
        /// </summary>
        public static Tensor FromArray<T>(T[] data, int[] shape) where T : struct
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));
            if (shape == null)
                throw new ArgumentNullException(nameof(shape));

            long expectedSize = shape.Aggregate(1L, (a, b) => a * b);
            if (data.Length != expectedSize)
                throw new ArgumentException($"Data length {data.Length} does not match shape {expectedSize}");

            DataType dataType = typeof(T) switch
            {
                var t when t == typeof(float) => DataType.Float32,
                var t when t == typeof(double) => DataType.Float32,
                var t when t == typeof(sbyte) => DataType.Int8,
                var t when t == typeof(byte) => DataType.Int8,
                var t when t == typeof(short) => DataType.Int16,
                var t when t == typeof(ushort) => DataType.Int16,
                var t when t == typeof(int) => DataType.Int32,
                var t when t == typeof(uint) => DataType.Int32,
                _ => throw new ArgumentException($"Unsupported type: {typeof(T)}")
            };

            var tensor = new Tensor(shape, dataType);

            unsafe
            {
                void* ptr = tensor._data.ToPointer();
                long count = tensor.Size;

                if (typeof(T) == typeof(float))
                {
                    float* floatPtr = (float*)ptr;
                    float[] floatData = (float[])(object)data;
                    for (long i = 0; i < count; i++)
                    {
                        floatPtr[i] = floatData[i];
                    }
                }
                else if (typeof(T) == typeof(int))
                {
                    int* intPtr = (int*)ptr;
                    int[] intData = (int[])(object)data;
                    for (long i = 0; i < count; i++)
                    {
                        intPtr[i] = intData[i];
                    }
                }
            }

            return tensor;
        }

        /// <summary>
        /// Creates an empty (uninitialized) tensor
        /// </summary>
        public static Tensor Empty(int[] shape, DataType dataType)
        {
            return new Tensor(shape, dataType);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (_ownData && _data != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(_data);
                }
                _data = IntPtr.Zero;
                _shape = Array.Empty<int>();
                _disposed = true;
            }
        }

        private void InitializeData()
        {
            // Initialize to zero for safety
            unsafe
            {
                byte* ptr = (byte*)_data.ToPointer();
                long bytes = ByteCount;
                for (long i = 0; i < bytes; i++)
                {
                    ptr[i] = 0;
                }
            }
        }

        private int CalculateOffset(int[] indices)
        {
            if (indices.Length != _shape.Length)
                throw new ArgumentException($"Expected {_shape.Length} indices, got {indices.Length}");

            int offset = 0;
            int stride = 1;
            for (int i = _shape.Length - 1; i >= 0; i--)
            {
                if (indices[i] < 0 || indices[i] >= _shape[i])
                    throw new ArgumentOutOfRangeException(nameof(indices));

                offset += indices[i] * stride;
                stride *= _shape[i];
            }

            return offset;
        }

        private static int GetDataTypeSize(DataType type)
        {
            return type switch
            {
                DataType.Float32 => 4,
                DataType.Float16 => 2,
                DataType.Int8 => 1,
                DataType.Int16 => 2,
                DataType.Int32 => 4,
                _ => throw new ArgumentException($"Unknown data type: {type}")
            };
        }
    }
}
