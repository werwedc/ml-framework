using System;

namespace MobileRuntime.Backends.Metal
{
    /// <summary>
    /// Interface for Metal buffer (MTLBuffer wrapper)
    /// </summary>
    public interface IMetalBuffer : IDisposable
    {
        /// <summary>
        /// Gets the native Metal buffer pointer
        /// </summary>
        IntPtr NativeBuffer { get; }

        /// <summary>
        /// Gets the buffer length in bytes
        /// </summary>
        long Length { get; }

        /// <summary>
        /// Gets a pointer to the buffer's contents
        /// </summary>
        IntPtr Contents { get; }

        /// <summary>
        /// Copies data from a source pointer to the buffer
        /// </summary>
        void CopyFrom(IntPtr source, long size);

        /// <summary>
        /// Copies data from the buffer to a destination pointer
        /// </summary>
        void CopyTo(IntPtr destination, long size);

        /// <summary>
        /// Marks a range of the buffer as modified
        /// </summary>
        void DidModifyRange(long location, long length);
    }
}
