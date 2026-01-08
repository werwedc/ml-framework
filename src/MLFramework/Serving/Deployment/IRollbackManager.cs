namespace MLFramework.Serving.Deployment;

/// <summary>
/// Interface for managing model deployment rollbacks
/// </summary>
public interface IRollbackManager
{
    /// <summary>
    /// Record a deployment event
    /// </summary>
    /// <param name="modelName">Name of the model</param>
    /// <param name="fromVersion">Version being upgraded from</param>
    /// <param name="toVersion">Version being upgraded to</param>
    /// <param name="deployedBy">User or system initiating the deployment</param>
    /// <returns>Unique deployment ID</returns>
    string RecordDeployment(string modelName, string fromVersion, string toVersion, string deployedBy);

    /// <summary>
    /// Rollback to a specific deployment
    /// </summary>
    /// <param name="deploymentId">ID of the deployment to rollback from</param>
    /// <param name="reason">Reason for the rollback</param>
    /// <param name="initiatedBy">User or system initiating the rollback</param>
    Task<RollbackResult> RollbackAsync(string deploymentId, string reason, string initiatedBy);

    /// <summary>
    /// Rollback to a specific version
    /// </summary>
    /// <param name="modelName">Name of the model</param>
    /// <param name="version">Version to rollback to</param>
    /// <param name="reason">Reason for the rollback</param>
    /// <param name="initiatedBy">User or system initiating the rollback</param>
    Task<RollbackResult> RollbackToVersionAsync(string modelName, string version, string reason, string initiatedBy);

    /// <summary>
    /// Get deployment history for a model
    /// </summary>
    /// <param name="modelName">Name of the model</param>
    /// <param name="limit">Maximum number of records to return</param>
    /// <returns>Deployment records in chronological order (most recent first)</returns>
    IEnumerable<DeploymentRecord> GetDeploymentHistory(string modelName, int limit = 10);

    /// <summary>
    /// Get a specific deployment record
    /// </summary>
    /// <param name="deploymentId">ID of the deployment</param>
    /// <returns>Deployment record, or null if not found</returns>
    DeploymentRecord? GetDeployment(string deploymentId);

    /// <summary>
    /// Check if a deployment can be rolled back
    /// </summary>
    /// <param name="deploymentId">ID of the deployment</param>
    /// <returns>True if rollback is possible</returns>
    bool CanRollback(string deploymentId);

    /// <summary>
    /// Set auto-rollback threshold for a model
    /// </summary>
    /// <param name="modelName">Name of the model</param>
    /// <param name="errorRateThreshold">Error rate threshold (0.0 to 1.0)</param>
    /// <param name="observationWindow">Window of time to observe error rates</param>
    void SetAutoRollbackThreshold(string modelName, float errorRateThreshold, TimeSpan observationWindow);

    /// <summary>
    /// Monitor error rate for a model version (triggers auto-rollback if threshold exceeded)
    /// </summary>
    /// <param name="modelName">Name of the model</param>
    /// <param name="version">Version of the model</param>
    /// <param name="currentErrorRate">Current error rate (0.0 to 1.0)</param>
    void MonitorErrorRate(string modelName, string version, float currentErrorRate);
}
