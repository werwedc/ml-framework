namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Interface for scheduler metrics collection.
/// </summary>
public interface ISchedulerMetrics
{
    /// <summary>
    /// Records completion of an iteration.
    /// </summary>
    void RecordIteration(IterationResult result);

    /// <summary>
    /// Records completion of a request.
    /// </summary>
    void RecordRequestCompletion(RequestResult result);

    /// <summary>
    /// Records batch utilization.
    /// </summary>
    void RecordBatchUtilization(double utilization);

    /// <summary>
    /// Records an error.
    /// </summary>
    void RecordError(string errorType, Exception exception);
}
