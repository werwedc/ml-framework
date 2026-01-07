using MLFramework.Inference.ContinuousBatching;

namespace MLFramework.Inference;

/// <summary>
/// Interface for model execution operations.
/// </summary>
public interface IModelExecutor
{
    /// <summary>
    /// Executes a batch of requests through the model.
    /// </summary>
    /// <param name="batch">The batch of requests to execute.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The batch output containing generated tokens and logits for each request.</returns>
    Task<BatchOutput> ExecuteBatchAsync(Batch batch, CancellationToken cancellationToken = default);
}
