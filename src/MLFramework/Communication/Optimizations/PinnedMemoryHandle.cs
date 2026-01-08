using System;
using System.Runtime.InteropServices;

namespace MLFramework.Communication.Optimizations
{
    /// <summary>
    /// Handle to pinned memory
    /// </summary>
    public class PinnedMemoryHandle : IDisposable
    {
        private readonly IntPtr _ptr;
        private readonly int _size;
        private readonly PinnedMemoryManager _manager;
        private readonly GCHandle _handle;
        private bool _disposed;

        public IntPtr Pointer => _ptr;
        public int Size => _size;

        internal PinnedMemoryHandle(IntPtr ptr, int size, PinnedMemoryManager manager, GCHandle handle)
        {
            _ptr = ptr;
            _size = size;
            _manager = manager;
            _handle = handle;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _manager.UnpinMemory(_ptr, _handle);
                _disposed = true;
            }
        }
    }
}
