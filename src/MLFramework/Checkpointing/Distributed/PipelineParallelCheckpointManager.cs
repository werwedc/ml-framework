using RitterFramework.Core.Tensor;

namespace MLFramework.Checkpointing.Distributed;

/// <summary>
/// Manages checkpointing for pipeline parallelism
/// </summary>
public class PipelineParallelCheckpointManager : IDisposable
{
    private readonly DistributedCheckpointManager _checkpointManager;
    private readonly int _numStages;
    private readonly int _currentStage;
    private readonly List<StageCheckpoint> _stageCheckpoints;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of PipelineParallelCheckpointManager
    /// </summary>
    /// <param name="checkpointManager">Distributed checkpoint manager</param>
    /// <param name="numStages">Number of pipeline stages</param>
    /// <param name="currentStage">Current pipeline stage</param>
    public PipelineParallelCheckpointManager(
        DistributedCheckpointManager checkpointManager,
        int numStages,
        int currentStage)
    {
        _checkpointManager = checkpointManager ?? throw new ArgumentNullException(nameof(checkpointManager));
        _numStages = numStages;
        _currentStage = currentStage;
        _stageCheckpoints = new List<StageCheckpoint>();
        _disposed = false;
    }

    /// <summary>
    /// Registers a checkpoint for the current pipeline stage
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="activation">The activation tensor to checkpoint</param>
    /// <param name="isBoundary">Whether this is a stage boundary</param>
    public void RegisterStageCheckpoint(
        string layerId,
        Tensor activation,
        bool isBoundary = false)
    {
        var stageCheckpoint = new StageCheckpoint
        {
            LayerId = layerId,
            Activation = activation,
            Stage = _currentStage,
            IsBoundary = isBoundary,
            Timestamp = DateTime.UtcNow
        };

        _stageCheckpoints.Add(stageCheckpoint);

        // If it's a boundary, store it in distributed manager
        if (isBoundary)
        {
            _checkpointManager.RegisterCheckpoint(layerId, activation);
        }
    }

    /// <summary>
    /// Gets checkpoints for a specific stage
    /// </summary>
    /// <param name="stage">Stage to get checkpoints for</param>
    /// <returns>List of checkpoints</returns>
    public List<Tensor> GetStageCheckpoints(int stage)
    {
        return _stageCheckpoints
            .Where(sc => sc.Stage == stage)
            .Select(sc => sc.Activation)
            .ToList();
    }

    /// <summary>
    /// Gets boundary checkpoints for the next stage
    /// </summary>
    /// <returns>List of boundary checkpoints</returns>
    public List<Tensor> GetNextStageBoundaries()
    {
        return _stageCheckpoints
            .Where(sc => sc.IsBoundary && sc.Stage == _currentStage)
            .Select(sc => sc.Activation)
            .ToList();
    }

    /// <summary>
    /// Clears all stage checkpoints
    /// </summary>
    public void ClearStageCheckpoints()
    {
        _stageCheckpoints.Clear();
    }

    /// <summary>
    /// Disposes the manager and releases resources
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            ClearStageCheckpoints();
            _disposed = true;
        }
    }

    private class StageCheckpoint
    {
        public string LayerId { get; set; } = string.Empty;
        public Tensor Activation { get; set; } = null!;
        public int Stage { get; set; }
        public bool IsBoundary { get; set; }
        public DateTime Timestamp { get; set; }
    }
}
