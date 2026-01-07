namespace MLFramework.HAL.CUDA;

/// <summary>
/// CUDA event implementation
/// </summary>
public class CudaEvent : IEvent
{
    private readonly CudaEventHandle _handle;

    public IStream Stream { get; }
    public bool IsCompleted { get; private set; }

    public CudaEvent(IStream stream, CudaEventHandle handle)
    {
        Stream = stream;
        _handle = handle;
        IsCompleted = false;
    }

    public void Synchronize()
    {
        if (IsCompleted)
            return;

        var result = CudaApi.CudaEventQuery(_handle.DangerousGetHandle());

        if (result == CudaError.Success)
        {
            IsCompleted = true;
        }
        else if (result == CudaError.NotReady)
        {
            // Block until event completes
            CudaException.CheckError(
                CudaApi.CudaEventQuery(_handle.DangerousGetHandle()));
            IsCompleted = true;
        }
        else
        {
            CudaException.CheckError(result);
        }
    }

    public void Dispose()
    {
        _handle.Dispose();
    }

    // Internal accessor for CudaStream
    internal CudaEventHandle Handle => _handle;
}
