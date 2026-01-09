using System;

namespace MobileRuntime
{
    /// <summary>
    /// Simple tensor implementation
    /// </summary>
    public class Tensor : ITensor, IDisposable
    {
        private float[] _data;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the Tensor class
        /// </summary>
        public Tensor()
        {
            _data = Array.Empty<float>();
        }

        /// <summary>
        /// Initializes a new instance of the Tensor class with data
        /// </summary>
        public Tensor(float[] data, int[] shape)
        {
            _data = data ?? throw new ArgumentNullException(nameof(data));
            Shape = shape ?? throw new ArgumentNullException(nameof(shape));
        }

        /// <summary>
        /// Gets the tensor shape
        /// </summary>
        public int[] Shape { get; set; } = Array.Empty<int>();

        /// <summary>
        /// Gets the data type
        /// </summary>
        public DataType DataType { get; set; } = DataType.Float32;

        /// <summary>
        /// Gets the total number of elements
        /// </summary>
        public int Length => _data.Length;

        /// <summary>
        /// Gets or sets tensor data
        /// </summary>
        public float[] Data
        {
            get => _data;
            set => _data = value ?? throw new ArgumentNullException(nameof(value));
        }

        /// <summary>
        /// Disposes resources
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _data = Array.Empty<float>();
                Shape = Array.Empty<int>();
                _disposed = true;
            }
        }
    }
}
