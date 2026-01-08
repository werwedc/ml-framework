namespace MachineLearning.Distributed.Enums;

/// <summary>
/// Types of rescaling operations
/// </summary>
public enum RescaleType
{
    /// <summary>
    /// Adding new workers to the cluster
    /// </summary>
    ScaleUp,

    /// <summary>
    /// Removing workers from the cluster
    /// </summary>
    ScaleDown,

    /// <summary>
    /// Replacement of failed workers
    /// </summary>
    Replace
}
