namespace MLFramework.Data;

/// <summary>
/// Encapsulates worker-specific state for partitioned data loading.
/// </summary>
public sealed class WorkerContext
{
    /// <summary>
    /// Gets the unique identifier for this worker.
    /// </summary>
    public int WorkerId { get; }

    /// <summary>
    /// Gets the total number of workers in the pool.
    /// </summary>
    public int NumWorkers { get; }

    /// <summary>
    /// Gets the starting index for this worker's partition.
    /// </summary>
    public int StartIndex { get; }

    /// <summary>
    /// Gets the ending index (exclusive) for this worker's partition.
    /// </summary>
    public int EndIndex { get; }

    /// <summary>
    /// Gets the cancellation token for this worker.
    /// </summary>
    public CancellationToken CancellationToken { get; }

    /// <summary>
    /// Initializes a new instance of the WorkerContext class.
    /// </summary>
    /// <param name="workerId">Unique identifier for this worker (0 to NumWorkers-1).</param>
    /// <param name="numWorkers">Total number of workers in the pool.</param>
    /// <param name="totalItems">Total number of items to process across all workers.</param>
    /// <param name="cancellationToken">Cancellation token for graceful shutdown.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when workerId or numWorkers is invalid.</exception>
    public WorkerContext(int workerId, int numWorkers, int totalItems, CancellationToken cancellationToken)
    {
        if (workerId < 0 || workerId >= numWorkers)
            throw new ArgumentOutOfRangeException(nameof(workerId), $"WorkerId must be between 0 and {numWorkers - 1}.");

        if (numWorkers <= 0)
            throw new ArgumentOutOfRangeException(nameof(numWorkers), "NumWorkers must be > 0.");

        WorkerId = workerId;
        NumWorkers = numWorkers;
        CancellationToken = cancellationToken;

        // Calculate partition using static partitioning
        int basePartitionSize = totalItems / numWorkers;
        int remainder = totalItems % numWorkers;

        // Workers 0 to remainder-1 get one extra item
        if (workerId < remainder)
        {
            StartIndex = workerId * (basePartitionSize + 1);
            EndIndex = StartIndex + basePartitionSize + 1;
        }
        else
        {
            StartIndex = workerId * basePartitionSize + remainder;
            EndIndex = StartIndex + basePartitionSize;
        }
    }

    /// <summary>
    /// Gets the number of items this worker should process.
    /// </summary>
    public int PartitionSize => EndIndex - StartIndex;

    /// <summary>
    /// Checks if a given index belongs to this worker's partition.
    /// </summary>
    /// <param name="index">The index to check.</param>
    /// <returns>True if the index is in this worker's partition, false otherwise.</returns>
    public bool ContainsIndex(int index) => index >= StartIndex && index < EndIndex;

    /// <summary>
    /// Returns a human-readable string representation of this context.
    /// </summary>
    public override string ToString()
    {
        return $"WorkerContext {{ WorkerId: {WorkerId}, StartIndex: {StartIndex}, EndIndex: {EndIndex}, PartitionSize: {PartitionSize} }}";
    }
}
