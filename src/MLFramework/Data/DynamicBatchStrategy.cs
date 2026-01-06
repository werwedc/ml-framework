namespace MLFramework.Data;

/// <summary>
/// Defines the strategy for dynamic batching of variable-length sequences.
/// </summary>
public enum DynamicBatchStrategy
{
    /// <summary>
    /// Pad all sequences to the maximum length in the batch.
    /// Simple and straightforward, but may waste computation on short sequences.
    /// </summary>
    PadToMax,

    /// <summary>
    /// Group sequences of similar lengths together into buckets.
    /// Reduces padding overhead compared to PadToMax.
    /// Requires sorting or binning.
    /// </summary>
    Bucket,

    /// <summary>
    /// Dynamically adjust batch size to fit within a token limit.
    /// Ensures uniform computation across batches.
    /// More complex but efficient for token budgets.
    /// </summary>
    Dynamic
}
