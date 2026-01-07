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
    /// <returns>The model output containing logits for each request.</returns>
    Task<ModelOutput> ExecuteBatchAsync(Batch batch, CancellationToken cancellationToken = default);
}

/// <summary>
/// Model output containing logits for each request.
/// </summary>
public record ModelOutput(
    Dictionary<RequestId, float[]> Logits
);
