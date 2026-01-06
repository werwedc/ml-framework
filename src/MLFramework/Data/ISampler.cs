namespace MLFramework.Data;

/// <summary>
/// Interface for sampling strategies that determine how samples are selected from a dataset.
/// </summary>
public interface ISampler
{
    /// <summary>
    /// Iterates over the sampled indices.
    /// </summary>
    /// <returns>An enumerable of indices.</returns>
    IEnumerable<int> Iterate();

    /// <summary>
    /// Gets the total number of samples that will be returned.
    /// </summary>
    int Length { get; }
}
