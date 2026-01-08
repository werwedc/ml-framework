namespace MLFramework.Checkpointing.Extensions;

/// <summary>
/// Extension methods for training loop integration
/// </summary>
public static class TrainingLoopExtensions
{
    /// <summary>
    /// Creates a checkpoint-aware training context
    /// </summary>
    /// <typeparam name="TModel">Type of the model</typeparam>
    /// <param name="model">The model to train</param>
    /// <param name="config">Checkpoint configuration</param>
    /// <returns>Checkpointed training context</returns>
    public static CheckpointedTrainingContext<TModel> WithCheckpointing<TModel>(
        this TModel model,
        CheckpointConfig config)
        where TModel : class
    {
        return new CheckpointedTrainingContext<TModel>(model, config);
    }

    /// <summary>
    /// Creates a checkpoint-aware training context with default configuration
    /// </summary>
    /// <typeparam name="TModel">Type of the model</typeparam>
    /// <param name="model">The model to train</param>
    /// <returns>Checkpointed training context</returns>
    public static CheckpointedTrainingContext<TModel> WithCheckpointing<TModel>(
        this TModel model)
        where TModel : class
    {
        return new CheckpointedTrainingContext<TModel>(model, CheckpointConfig.Default);
    }
}
