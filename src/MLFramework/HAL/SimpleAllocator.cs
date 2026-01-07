namespace MLFramework.HAL;

/// <summary>
/// Simple allocator that allocates memory directly without caching
/// </summary>
public class SimpleAllocator : IMemoryAllocator
{
    private readonly IDevice _device;
    private long _allocatedSize;

    public IDevice Device => _device;
    public long CacheSize => 0; // No caching
    public long AllocatedSize => _allocatedSize;

    public SimpleAllocator(IDevice device)
    {
        _device = device;
    }

    public IMemoryBuffer Allocate(long size)
    {
        if (size <= 0)
            throw new ArgumentException("Size must be positive", nameof(size));

        var pointer = System.Runtime.InteropServices.Marshal.AllocHGlobal((System.IntPtr)size);
        _allocatedSize += size;

        return new SimpleMemoryBuffer(pointer, size, _device, this);
    }

    public void Free(IMemoryBuffer buffer)
    {
        if (buffer == null)
            throw new ArgumentNullException(nameof(buffer));

        if (buffer is SimpleMemoryBuffer simpleBuffer)
        {
            _allocatedSize -= simpleBuffer.Size;
            System.Runtime.InteropServices.Marshal.FreeHGlobal(simpleBuffer.Pointer);
        }
        else
        {
            throw new ArgumentException("Invalid buffer type");
        }
    }

    public void EmptyCache()
    {
        // No-op for simple allocator
    }

    public void Dispose()
    {
        // Any remaining buffers should be freed
        // In a real implementation, track all buffers
    }

    private class SimpleMemoryBuffer : IMemoryBuffer
    {
        private readonly SimpleAllocator _allocator;
        private bool _disposed;

        public System.IntPtr Pointer { get; }
        public long Size { get; }
        public IDevice Device { get; }
        public bool IsValid => !_disposed;

        public SimpleMemoryBuffer(System.IntPtr pointer, long size, IDevice device, SimpleAllocator allocator)
        {
            Pointer = pointer;
            Size = size;
            Device = device;
            _allocator = allocator;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _allocator.Free(this);
                _disposed = true;
            }
        }
    }
}
