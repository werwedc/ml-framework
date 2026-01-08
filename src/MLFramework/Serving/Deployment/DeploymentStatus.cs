namespace MLFramework.Serving.Deployment;

/// <summary>
/// Status of a model deployment
/// </summary>
public enum DeploymentStatus
{
    /// <summary>
    /// Deployment was successful
    /// </summary>
    Success,

    /// <summary>
    /// Deployment was rolled back
    /// </summary>
    RolledBack,

    /// <summary>
    /// Deployment failed
    /// </summary>
    Failed
}
