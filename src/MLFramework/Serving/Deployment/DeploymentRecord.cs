namespace MLFramework.Serving.Deployment;

/// <summary>
/// Record of a model deployment
/// </summary>
public class DeploymentRecord
{
    /// <summary>
    /// Unique identifier for the deployment
    /// </summary>
    public string DeploymentId { get; }

    /// <summary>
    /// Name of the model
    /// </summary>
    public string ModelName { get; }

    /// <summary>
    /// Version being deployed from
    /// </summary>
    public string FromVersion { get; }

    /// <summary>
    /// Version being deployed to
    /// </summary>
    public string ToVersion { get; }

    /// <summary>
    /// Timestamp when deployment occurred
    /// </summary>
    public DateTime DeploymentTime { get; }

    /// <summary>
    /// Who initiated the deployment
    /// </summary>
    public string DeployedBy { get; }

    /// <summary>
    /// Current status of the deployment
    /// </summary>
    public DeploymentStatus Status { get; private set; }

    /// <summary>
    /// Reason for the deployment or rollback
    /// </summary>
    public string? Reason { get; set; }

    /// <summary>
    /// Metrics before deployment
    /// </summary>
    public Dictionary<string, float> PreDeploymentMetrics { get; set; }

    /// <summary>
    /// Metrics after deployment
    /// </summary>
    public Dictionary<string, float> PostDeploymentMetrics { get; set; }

    public DeploymentRecord(
        string deploymentId,
        string modelName,
        string fromVersion,
        string toVersion,
        DateTime deploymentTime,
        string deployedBy,
        DeploymentStatus status = DeploymentStatus.Success)
    {
        DeploymentId = deploymentId ?? throw new ArgumentNullException(nameof(deploymentId));
        ModelName = modelName ?? throw new ArgumentNullException(nameof(modelName));
        FromVersion = fromVersion ?? throw new ArgumentNullException(nameof(fromVersion));
        ToVersion = toVersion ?? throw new ArgumentNullException(nameof(toVersion));
        DeploymentTime = deploymentTime;
        DeployedBy = deployedBy ?? throw new ArgumentNullException(nameof(deployedBy));
        Status = status;
        PreDeploymentMetrics = new Dictionary<string, float>();
        PostDeploymentMetrics = new Dictionary<string, float>();
    }

    /// <summary>
    /// Mark this deployment as rolled back
    /// </summary>
    internal void MarkAsRolledBack()
    {
        Status = DeploymentStatus.RolledBack;
    }

    /// <summary>
    /// Mark this deployment as failed
    /// </summary>
    internal void MarkAsFailed()
    {
        Status = DeploymentStatus.Failed;
    }
}
