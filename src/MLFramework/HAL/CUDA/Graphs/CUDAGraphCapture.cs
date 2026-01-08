namespace MLFramework.HAL.CUDA;

/// <summary>
/// CUDA graph capture implementation
/// </summary>
public class CUDAGraphCapture : ICUDAGraphCapture
{
    private CudaStreamHandle _cuStream;
    private CudaGraphHandle _cuGraph;
    private bool _isCapturing;
    private bool _disposed;

    /// <summary>
    /// Gets whether capture is currently active
    /// </summary>
    public bool IsCapturing => _isCapturing;

    public CUDAGraphCapture()
    {
        _cuGraph = null;
        _cuStream = null;
        _isCapturing = false;
        _disposed = false;
    }

    /// <summary>
    /// Begins capturing kernel launches from the specified stream
    /// </summary>
    /// <param name="stream">CUDA stream to capture from</param>
    public void BeginCapture(CudaStream stream)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CUDAGraphCapture));

        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        if (_isCapturing)
            throw new InvalidOperationException("Capture is already in progress");

        _cuStream = stream.Handle;

        // Call CUDA driver API: cuStreamBeginCapture
        var result = CudaApi.CudaStreamBeginCapture(
            _cuStream.DangerousGetHandle(),
            CudaCaptureMode.CaptureModeThreadLocal);

        CudaException.CheckError(result, "Failed to begin stream capture");

        _isCapturing = true;
    }

    /// <summary>
    /// Ends capture and instantiates the graph
    /// </summary>
    /// <returns>The captured CUDA graph</returns>
    public ICUDAGraph EndCapture()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CUDAGraphCapture));

        if (!_isCapturing)
            throw new InvalidOperationException("No capture in progress");

        // Call CUDA driver API: cuStreamEndCapture
        var result = CudaApi.CudaStreamEndCapture(
            _cuStream.DangerousGetHandle(),
            out IntPtr graph);

        CudaException.CheckError(result, "Failed to end stream capture");

        _cuGraph = new CudaGraphHandle(graph);
        _isCapturing = false;

        return new CUDAGraph(graph);
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (_disposed)
            return;

        if (disposing)
        {
            // Abort capture if disposing while capturing
            if (_isCapturing && _cuStream != null)
            {
                try
                {
                    CudaApi.CudaStreamEndCapture(
                        _cuStream.DangerousGetHandle(),
                        out _);
                }
                catch
                {
                    // Ignore errors during abort
                }

                _isCapturing = false;
            }

            _cuGraph?.Dispose();
        }

        _disposed = true;
    }
}
