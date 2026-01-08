namespace MLFramework.HAL.CUDA;

/// <summary>
/// CUDA graph implementation
/// </summary>
internal class CUDAGraph : ICUDAGraph
{
    private readonly CudaGraphHandle _cuGraph;
    private IntPtr _cuGraphExec;
    private readonly object _lock = new object();
    private bool _disposed;

    /// <summary>
    /// Gets the unique identifier for this graph
    /// </summary>
    public string GraphId { get; }

    /// <summary>
    /// Gets the current state of the graph
    /// </summary>
    public CUDAGraphState State { get; private set; }

    /// <summary>
    /// Gets the native CUDA graph handle
    /// </summary>
    internal CudaGraphHandle Handle => _cuGraph;

    /// <summary>
    /// Internal property for weight updates
    /// </summary>
    internal IntPtr GraphExecHandle => _cuGraphExec;

    public CUDAGraph(IntPtr graph)
    {
        _cuGraph = new CudaGraphHandle(graph);
        _cuGraphExec = IntPtr.Zero;
        GraphId = Guid.NewGuid().ToString();
        State = CUDAGraphState.Created;
    }

    /// <summary>
    /// Instantiates the graph for execution
    /// </summary>
    private void Instantiate()
    {
        if (_cuGraphExec != IntPtr.Zero)
            return;

        lock (_lock)
        {
            // Double-check pattern
            if (_cuGraphExec != IntPtr.Zero)
                return;

            var result = CudaApi.CudaGraphInstantiate(
                out IntPtr graphExec,
                _cuGraph.DangerousGetHandle(),
                IntPtr.Zero,
                0);

            CudaException.CheckError(result);
            _cuGraphExec = graphExec;
            State = CUDAGraphState.Ready;
        }
    }

    /// <summary>
    /// Executes the captured graph
    /// </summary>
    /// <param name="stream">CUDA stream for execution</param>
    public void Execute(CudaStream stream)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CUDAGraph));

        // Instantiate graph if not already done
        Instantiate();

        State = CUDAGraphState.Executing;

        try
        {
            var result = CudaApi.CudaGraphLaunch(_cuGraphExec, stream.Handle.DangerousGetHandle());
            CudaException.CheckError(result);
        }
        finally
        {
            State = CUDAGraphState.Ready;
        }
    }

    /// <summary>
    /// Validates that the graph is executable
    /// </summary>
    /// <returns>Validation result with any errors</returns>
    public CUDAGraphValidationResult Validate()
    {
        var errors = new List<string>();
        var warnings = new List<string>();

        // Get number of nodes in the graph
        var result = CudaApi.CudaGraphGetNodes(
            _cuGraph.DangerousGetHandle(),
            out IntPtr nodes,
            out ulong nodeCount);
        CudaException.CheckError(result);

        // Basic validation
        if (nodeCount == 0)
        {
            errors.Add("Graph contains no operations");
        }

        // Check if graph can be instantiated
        try
        {
            Instantiate();
        }
        catch (CudaException ex)
        {
            errors.Add($"Graph instantiation failed: {ex.Message}");
        }

        return new CUDAGraphValidationResult
        {
            IsValid = errors.Count == 0,
            Errors = errors,
            Warnings = warnings,
            OperationCount = (int)nodeCount
        };
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
            lock (_lock)
            {
                // Dispose managed resources
                if (_cuGraphExec != IntPtr.Zero)
                {
                    CudaException.CheckError(
                        CudaApi.CudaGraphExecDestroy(_cuGraphExec));
                    _cuGraphExec = IntPtr.Zero;
                }

                _cuGraph?.Dispose();
            }
        }

        State = CUDAGraphState.Disposed;
        _disposed = true;
    }
}
