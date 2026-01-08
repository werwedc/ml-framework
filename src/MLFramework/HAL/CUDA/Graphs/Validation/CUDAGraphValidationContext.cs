namespace MLFramework.HAL.CUDA.Graphs.Validation;

/// <summary>
/// Provides context for validation by tracking operations and memory allocations
/// during graph capture
/// </summary>
public class CUDAGraphValidationContext
{
    private readonly List<string> _capturedOperations;
    private readonly HashSet<IntPtr> _allocatedMemory;
    private readonly Dictionary<string, int> _operationCounts;

    public CUDAGraphValidationContext()
    {
        _capturedOperations = new List<string>();
        _allocatedMemory = new HashSet<IntPtr>();
        _operationCounts = new Dictionary<string, int>();
    }

    /// <summary>
    /// Records an operation that was captured in the graph
    /// </summary>
    /// <param name="operationName">Name of the operation</param>
    public void RecordOperation(string operationName)
    {
        if (string.IsNullOrEmpty(operationName))
            return;

        _capturedOperations.Add(operationName);

        // Track operation counts
        if (!_operationCounts.ContainsKey(operationName))
        {
            _operationCounts[operationName] = 0;
        }
        _operationCounts[operationName]++;
    }

    /// <summary>
    /// Records a memory allocation that was made
    /// </summary>
    /// <param name="ptr">Pointer to the allocated memory</param>
    public void RecordMemoryAllocation(IntPtr ptr)
    {
        if (ptr != IntPtr.Zero)
        {
            _allocatedMemory.Add(ptr);
        }
    }

    /// <summary>
    /// Gets a read-only list of all captured operations
    /// </summary>
    public IReadOnlyList<string> CapturedOperations => _capturedOperations;

    /// <summary>
    /// Gets a read-only set of all allocated memory pointers
    /// </summary>
    public IReadOnlySet<IntPtr> AllocatedMemory => _allocatedMemory;

    /// <summary>
    /// Gets the count of a specific operation type
    /// </summary>
    /// <param name="operationName">Name of the operation</param>
    /// <returns>Number of times this operation was captured</returns>
    public int GetOperationCount(string operationName)
    {
        return _operationCounts.TryGetValue(operationName, out var count) ? count : 0;
    }

    /// <summary>
    /// Gets all operation types and their counts
    /// </summary>
    public IReadOnlyDictionary<string, int> OperationCounts => _operationCounts;

    /// <summary>
    /// Clears all recorded data
    /// </summary>
    public void Clear()
    {
        _capturedOperations.Clear();
        _allocatedMemory.Clear();
        _operationCounts.Clear();
    }

    /// <summary>
    /// Checks if an operation type was captured
    /// </summary>
    /// <param name="operationName">Name of the operation</param>
    /// <returns>True if the operation was captured at least once</returns>
    public bool HasOperation(string operationName)
    {
        return _operationCounts.ContainsKey(operationName);
    }
}
