namespace MLFramework.HAL.CUDA;

/// <summary>
/// CUDA stream implementation
/// </summary>
public class CudaStream : IStream
{
    private readonly CudaStreamHandle _handle;
    private readonly CudaDevice _device;
    private readonly Queue<Action> _pendingOperations;

    public IDevice Device => _device;

    public CudaStream(CudaDevice device)
    {
        _device = device;

        CudaException.CheckError(
            CudaApi.CudaStreamCreate(out IntPtr streamHandle));

        _handle = new CudaStreamHandle(streamHandle);
        _pendingOperations = new Queue<Action>();
    }

    public void Enqueue(Action operation)
    {
        if (operation == null)
            throw new ArgumentNullException(nameof(operation));

        // Execute the operation (should launch CUDA kernel)
        operation();
    }

    public IEvent RecordEvent()
    {
        CudaException.CheckError(
            CudaApi.CudaEventCreate(out IntPtr eventHandle));

        var eventHandleWrapper = new CudaEventHandle(eventHandle);

        CudaException.CheckError(
            CudaApi.CudaEventRecord(
                eventHandleWrapper.DangerousGetHandle(),
                _handle.DangerousGetHandle()));

        return new CudaEvent(this, eventHandleWrapper);
    }

    public void Wait(IEvent evt)
    {
        if (evt == null)
            throw new ArgumentNullException(nameof(evt));

        if (evt is CudaEvent cudaEvent)
        {
            CudaException.CheckError(
                CudaApi.CudaStreamWaitEvent(
                    _handle.DangerousGetHandle(),
                    cudaEvent.Handle.DangerousGetHandle(),
                    0));
        }
        else
        {
            throw new ArgumentException("Invalid event type");
        }
    }

    public void Synchronize()
    {
        CudaException.CheckError(
            CudaApi.CudaStreamSynchronize(_handle.DangerousGetHandle()));
    }

    public void Dispose()
    {
        _handle.Dispose();
    }

    // Internal accessor for other CUDA components
    internal CudaStreamHandle Handle => _handle;
}
