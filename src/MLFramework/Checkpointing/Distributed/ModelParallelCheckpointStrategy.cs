using RitterFramework.Core.Tensor;

namespace MLFramework.Checkpointing.Distributed;

/// <summary>
/// Checkpointing strategy for model parallelism
/// </summary>
public class ModelParallelCheckpointStrategy
{
    private readonly DistributedCheckpointManager _checkpointManager;
    private readonly int _tensorParallelSize;
    private readonly Dictionary<string, int> _layerToRankMap;

    /// <summary>
    /// Initializes a new instance of ModelParallelCheckpointStrategy
    /// </summary>
    /// <param name="checkpointManager">Distributed checkpoint manager</param>
    /// <param name="tensorParallelSize">Tensor parallel size</param>
    /// <param name="layerToRankMap">Mapping of layers to ranks</param>
    public ModelParallelCheckpointStrategy(
        DistributedCheckpointManager checkpointManager,
        int tensorParallelSize,
        Dictionary<string, int>? layerToRankMap = null)
    {
        _checkpointManager = checkpointManager ?? throw new ArgumentNullException(nameof(checkpointManager));
        _tensorParallelSize = tensorParallelSize;
        _layerToRankMap = layerToRankMap ?? new Dictionary<string, int>();
    }

    /// <summary>
    /// Determines whether to checkpoint a layer
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="activation">The activation tensor</param>
    /// <param name="isTensorParallel">Whether the layer is tensor parallel</param>
    /// <returns>True if should checkpoint, false otherwise</returns>
    public bool ShouldCheckpoint(
        string layerId,
        Tensor activation,
        bool isTensorParallel = false)
    {
        // If tensor parallel, checkpoint only on first rank
        if (isTensorParallel)
        {
            return _checkpointManager.Rank % _tensorParallelSize == 0;
        }

        // Otherwise, checkpoint based on strategy
        return true;
    }

    /// <summary>
    /// Gets the rank that should store the checkpoint for a layer
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <returns>Rank to store checkpoint</returns>
    public int GetCheckpointRank(string layerId)
    {
        return _layerToRankMap.TryGetValue(layerId, out var rank) ? rank : 0;
    }

    /// <summary>
    /// Registers a layer-rank mapping
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="rank">Rank to store checkpoint</param>
    public void RegisterLayerRank(string layerId, int rank)
    {
        _layerToRankMap[layerId] = rank;
    }
}
