namespace MachineLearning.Checkpointing;

/// <summary>
/// Optimizer-specific state dictionary that tracks optimizer parameters
/// </summary>
public class OptimizerStateDict : StateDict
{
    /// <summary>
    /// Type of optimizer
    /// </summary>
    public OptimizerType OptimizerType { get; set; }

    /// <summary>
    /// Current optimization step
    /// </summary>
    public long Step { get; set; }

    /// <summary>
    /// Current learning rate
    /// </summary>
    public float LearningRate { get; set; }

    /// <summary>
    /// Creates an empty state dict for a specific optimizer type
    /// </summary>
    /// <param name="type">The optimizer type</param>
    /// <returns>A new OptimizerStateDict instance</returns>
    public static OptimizerStateDict Create(OptimizerType type)
    {
        return new OptimizerStateDict
        {
            OptimizerType = type,
            Step = 0,
            LearningRate = 0.001f // Default learning rate
        };
    }

    /// <summary>
    /// Creates an optimizer state dict with initial parameters
    /// </summary>
    /// <param name="type">The optimizer type</param>
    /// <param name="learningRate">Initial learning rate</param>
    /// <returns>A new OptimizerStateDict instance</returns>
    public static OptimizerStateDict Create(OptimizerType type, float learningRate)
    {
        return new OptimizerStateDict
        {
            OptimizerType = type,
            Step = 0,
            LearningRate = learningRate
        };
    }

    /// <summary>
    /// Gets optimizer-specific state for a parameter
    /// </summary>
    /// <param name="parameterName">Name of the parameter</param>
    /// <returns>State dictionary containing optimizer state for the parameter</returns>
    public StateDict GetParameterState(string parameterName)
    {
        var paramState = new StateDict();
        var prefix = $"{parameterName}_";

        foreach (var (key, value) in this)
        {
            if (key.StartsWith(prefix))
            {
                var newKey = key.Substring(prefix.Length);
                paramState[newKey] = value;
            }
        }

        return paramState;
    }

    /// <summary>
    /// Sets optimizer-specific state for a parameter
    /// </summary>
    /// <param name="parameterName">Name of the parameter</param>
    /// <param name="paramState">State dictionary containing optimizer state for the parameter</param>
    public void SetParameterState(string parameterName, StateDict paramState)
    {
        var prefix = $"{parameterName}_";

        // Remove old parameter state
        var keysToRemove = Keys.Where(k => k.StartsWith(prefix)).ToList();
        foreach (var key in keysToRemove)
        {
            Remove(key);
        }

        // Add new parameter state
        foreach (var (key, value) in paramState)
        {
            this[$"{prefix}{key}"] = value;
        }
    }
}
