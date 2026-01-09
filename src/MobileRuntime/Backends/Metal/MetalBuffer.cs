using System;
using System.Runtime.InteropServices;

namespace MobileRuntime.Backends.Metal
{
    /// <summary>
    /// Implementation of Metal buffer (MTLBuffer wrapper)
    /// </summary>
    public sealed class MetalBuffer : IMetalBuffer
    {
        private IntPtr _buffer;
        private readonly long _length;
        private readonly IntPtr _device;
        private bool _disposed;

        // Metal P/Invoke declarations
        [DllImport("__Internal", EntryPoint = "MTLBuffer_release")]
        private static extern void MTLBuffer_release(IntPtr buffer);

        [DllImport("__Internal", EntryPoint = "MTLBuffer_contents")]
        private static extern IntPtr MTLBuffer_contents(IntPtr buffer);

        [DllImport("__Internal", EntryPoint = "MTLBuffer_didModifyRange")]
        private static extern void MTLBuffer_didModifyRange(IntPtr buffer, long location, long length);

        [DllImport("__Internal", EntryPoint = "memcpy")]
        private static extern void memcpy(IntPtr destination, IntPtr source, long size);

        /// <summary>
        /// Creates a new Metal buffer
        /// </summary>
        /// <param name="device">The Metal device pointer</param>
        /// <param name="length">The buffer length in bytes</param>
        public MetalBuffer(IntPtr device, long length)
        {
            _device = device;
            _length = length;
            _buffer = AllocateMetalBuffer(device, length);
        }

        /// <inheritdoc/>
        public IntPtr NativeBuffer => _buffer;

        /// <inheritdoc/>
        public long Length => _length;

        /// <inheritdoc/>
        public IntPtr Contents
        {
            get
            {
                if (_disposed)
                    throw new ObjectDisposedException(nameof(MetalBuffer));
                return MTLBuffer_contents(_buffer);
            }
        }

        /// <inheritdoc/>
        public void CopyFrom(IntPtr source, long size)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(MetalBuffer));
            if (size > _length)
                throw new ArgumentException($"Size {size} exceeds buffer length {_length}");

            memcpy(Contents, source, size);
            DidModifyRange(0, size);
        }

        /// <inheritdoc/>
        public void CopyTo(IntPtr destination, long size)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(MetalBuffer));
            if (size > _length)
                throw new ArgumentException($"Size {size} exceeds buffer length {_length}");

            memcpy(destination, Contents, size);
        }

        /// <inheritdoc/>
        public void DidModifyRange(long location, long length)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(MetalBuffer));

            MTLBuffer_didModifyRange(_buffer, location, length);
        }

        /// <summary>
        /// Allocates a Metal buffer (native implementation)
        /// </summary>
        private IntPtr AllocateMetalBuffer(IntPtr device, long length)
        {
            // Native Metal buffer allocation
            // This would call MTLDevice_newBufferWithLength:options:
            // For now, we'll use managed memory as a placeholder
            IntPtr buffer = Marshal.AllocHGlobal((IntPtr)length);
            return buffer;
        }

        /// <summary>
        /// Finalizer
        /// </summary>
        ~MetalBuffer()
        {
            Dispose(false);
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Disposes of the buffer
        /// </summary>
        private void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (_buffer != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(_buffer);
                    _buffer = IntPtr.Zero;
                }
                _disposed = true;
            }
        }
    }
}
