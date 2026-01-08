namespace MachineLearning.Checkpointing;

/// <summary>
/// Interface for distributed models that support different training strategies
/// </summary>
public interface IDistributedModel : IStateful
{
    /// <summary>
    /// Gets the distributed training strategy used by this model
    /// </summary>
    DistributedStrategy Strategy { get; }

    /// <summary>
    /// Gets the total number of processes in the distributed group
    /// </summary>
    int WorldSize { get; }

    /// <summary>
    /// Gets the rank of the current process in the distributed group
    /// </summary>
    int Rank { get; }

    /// <summary>
    /// Get the local shard of the model (for FSDP, TP)
    /// </summary>
    StateDict GetLocalStateDict();

    /// <summary>
    /// Get the full state dict (for DDP, gathered)
    /// </summary>
    StateDict GetFullStateDict();

    /// <summary>
    /// Load local shard (for FSDP, TP)
    /// </summary>
    void LoadLocalStateDict(StateDict state);

    /// <summary>
    /// Load full state dict (for DDP)
    /// </summary>
    void LoadFullStateDict(StateDict state);
}
