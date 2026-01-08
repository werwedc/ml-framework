using System;

namespace MLFramework.Checkpointing.Extensions;

/// <summary>
/// Context for checkpointed training
/// </summary>
/// <typeparam name="TModel">Type of the model</typeparam>
public class CheckpointedTrainingContext<TModel> : IDisposable
    where TModel : class
{
    private readonly TModel _model;
    private readonly CheckpointConfig _config;
    private readonly CheckpointContext _checkpointContext;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of CheckpointedTrainingContext
    /// </summary>
    /// <param name="model">The model to train</param>
    /// <param name="config">Checkpoint configuration</param>
    public CheckpointedTrainingContext(TModel model, CheckpointConfig config)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _checkpointContext = new CheckpointContext(config);
        _checkpointContext.Enter();
        _disposed = false;
    }

    /// <summary>
    /// Gets the model
    /// </summary>
    public TModel Model => _model;

    /// <summary>
    /// Gets the checkpoint configuration
    /// </summary>
    public CheckpointConfig Config => _config;

    /// <summary>
    /// Gets checkpointing statistics
    /// </summary>
    public CheckpointStatistics GetStatistics()
    {
        ThrowIfDisposed();
        return _checkpointContext.GetStatistics();
    }

    /// <summary>
    /// Disposes the context
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            _checkpointContext.Exit();
            _checkpointContext.Dispose();
            _disposed = true;
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CheckpointedTrainingContext<TModel>));
    }
}
