namespace MLFramework.Data;

/// <summary>
/// Thread-safe collection and aggregator for worker errors that occur during data loading.
/// Provides query methods for error statistics and analysis.
/// </summary>
public sealed class ErrorAggregator
{
    private readonly List<WorkerError> _errors;
    private readonly object _lock = new object();

    /// <summary>
    /// Gets the total number of errors that have been aggregated.
    /// </summary>
    public int TotalErrors
    {
        get
        {
            lock (_lock)
            {
                return _errors.Count;
            }
        }
    }

    /// <summary>
    /// Initializes a new instance of the ErrorAggregator class.
    /// </summary>
    public ErrorAggregator()
    {
        _errors = new List<WorkerError>();
    }

    /// <summary>
    /// Adds an error to the aggregator in a thread-safe manner.
    /// </summary>
    /// <param name="error">The error to add.</param>
    /// <exception cref="ArgumentNullException">Thrown when error is null.</exception>
    public void AddError(WorkerError error)
    {
        if (error == null)
            throw new ArgumentNullException(nameof(error));

        lock (_lock)
        {
            _errors.Add(error);
        }
    }

    /// <summary>
    /// Gets all errors that have been aggregated.
    /// </summary>
    /// <returns>A read-only list of all errors.</returns>
    public IReadOnlyList<WorkerError> GetErrors()
    {
        lock (_lock)
        {
            return _errors.ToList().AsReadOnly();
        }
    }

    /// <summary>
    /// Gets the most recent error that was added to the aggregator.
    /// </summary>
    /// <returns>The last error, or null if no errors have been added.</returns>
    public WorkerError? GetLastError()
    {
        lock (_lock)
        {
            return _errors.Count > 0 ? _errors[^1] : null;
        }
    }

    /// <summary>
    /// Gets the number of errors that occurred for a specific worker.
    /// </summary>
    /// <param name="workerId">The worker ID to query.</param>
    /// <returns>The count of errors for the specified worker.</returns>
    public int GetErrorCount(int workerId)
    {
        lock (_lock)
        {
            return _errors.Count(e => e.WorkerId == workerId);
        }
    }

    /// <summary>
    /// Gets all errors that occurred for a specific worker.
    /// </summary>
    /// <param name="workerId">The worker ID to query.</param>
    /// <returns>A read-only list of errors for the specified worker.</returns>
    public IReadOnlyList<WorkerError> GetErrorsByWorker(int workerId)
    {
        lock (_lock)
        {
            return _errors.Where(e => e.WorkerId == workerId).ToList().AsReadOnly();
        }
    }

    /// <summary>
    /// Gets all errors that occurred after a specific timestamp.
    /// </summary>
    /// <param name="timestamp">The timestamp to filter from.</param>
    /// <returns>A read-only list of errors that occurred after the timestamp.</returns>
    public IReadOnlyList<WorkerError> GetErrorsAfter(DateTime timestamp)
    {
        lock (_lock)
        {
            return _errors.Where(e => e.Timestamp > timestamp).ToList().AsReadOnly();
        }
    }

    /// <summary>
    /// Clears all errors from the aggregator.
    /// </summary>
    public void ClearErrors()
    {
        lock (_lock)
        {
            _errors.Clear();
        }
    }

    /// <summary>
    /// Checks if there are any errors in the aggregator.
    /// </summary>
    /// <returns>True if there are errors, false otherwise.</returns>
    public bool HasErrors()
    {
        lock (_lock)
        {
            return _errors.Count > 0;
        }
    }

    /// <summary>
    /// Gets the count of unique workers that have encountered errors.
    /// </summary>
    /// <returns>The number of unique worker IDs in the error collection.</returns>
    public int GetUniqueWorkerCount()
    {
        lock (_lock)
        {
            return _errors.Select(e => e.WorkerId).Distinct().Count();
        }
    }

    /// <summary>
    /// Gets a summary of errors grouped by worker ID.
    /// </summary>
    /// <returns>A dictionary mapping worker IDs to error counts.</returns>
    public Dictionary<int, int> GetErrorSummaryByWorker()
    {
        lock (_lock)
        {
            return _errors.GroupBy(e => e.WorkerId)
                          .ToDictionary(g => g.Key, g => g.Count());
        }
    }

    /// <summary>
    /// Gets a summary of errors grouped by exception type.
    /// </summary>
    /// <returns>A dictionary mapping exception type names to error counts.</returns>
    public Dictionary<string, int> GetErrorSummaryByExceptionType()
    {
        lock (_lock)
        {
            return _errors.GroupBy(e => e.Exception.GetType().Name)
                          .ToDictionary(g => g.Key, g => g.Count());
        }
    }

    /// <summary>
    /// Returns a formatted summary of all errors.
    /// </summary>
    /// <returns>A string containing a summary of the aggregated errors.</returns>
    public string GetErrorSummary()
    {
        lock (_lock)
        {
            if (_errors.Count == 0)
                return "No errors.";

            var summary = new System.Text.StringBuilder();
            summary.AppendLine($"Total Errors: {_errors.Count}");
            summary.AppendLine($"Unique Workers with Errors: {GetUniqueWorkerCount()}");

            var byWorker = GetErrorSummaryByWorker();
            if (byWorker.Count > 0)
            {
                summary.AppendLine("\nErrors by Worker:");
                foreach (var kvp in byWorker.OrderByDescending(x => x.Value))
                {
                    summary.AppendLine($"  Worker {kvp.Key}: {kvp.Value} error(s)");
                }
            }

            var byExceptionType = GetErrorSummaryByExceptionType();
            if (byExceptionType.Count > 0)
            {
                summary.AppendLine("\nErrors by Exception Type:");
                foreach (var kvp in byExceptionType.OrderByDescending(x => x.Value))
                {
                    summary.AppendLine($"  {kvp.Key}: {kvp.Value} error(s)");
                }
            }

            return summary.ToString();
        }
    }
}
