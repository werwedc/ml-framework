using Microsoft.Extensions.Logging;

namespace MLFramework.Serving.Deployment;

/// <summary>
/// Manages rollback history and provides instant rollback capabilities
/// </summary>
public class RollbackManager : IRollbackManager
{
    private readonly ILogger<RollbackManager>? _logger;
    private readonly IModelHotswapper _modelHotswapper;
    private readonly IVersionRouterCore _versionRouter;
    private readonly object _lock = new object();
    private readonly Dictionary<string, List<DeploymentRecord>> _deploymentHistory;
    private readonly Dictionary<string, DeploymentRecord> _deploymentsById;
    private readonly Dictionary<string, AutoRollbackConfig> _autoRollbackConfigs;
    private readonly Dictionary<string, ErrorRateMonitor> _errorRateMonitors;
    private readonly int _maxHistorySize;

    /// <summary>
    /// Create a new RollbackManager
    /// </summary>
    /// <param name="modelHotswapper">Hotswapper for model version switching</param>
    /// <param name="versionRouter">Router for updating model version routing</param>
    /// <param name="maxHistorySize">Maximum number of deployments to keep in history per model</param>
    /// <param name="logger">Optional logger</param>
    public RollbackManager(
        IModelHotswapper modelHotswapper,
        IVersionRouterCore versionRouter,
        int maxHistorySize = 10,
        ILogger<RollbackManager>? logger = null)
    {
        _modelHotswapper = modelHotswapper ?? throw new ArgumentNullException(nameof(modelHotswapper));
        _versionRouter = versionRouter ?? throw new ArgumentNullException(nameof(versionRouter));
        _maxHistorySize = maxHistorySize <= 0 ? 10 : maxHistorySize;
        _logger = logger;
        _deploymentHistory = new Dictionary<string, List<DeploymentRecord>>();
        _deploymentsById = new Dictionary<string, DeploymentRecord>();
        _autoRollbackConfigs = new Dictionary<string, AutoRollbackConfig>();
        _errorRateMonitors = new Dictionary<string, ErrorRateMonitor>();
    }

    /// <inheritdoc />
    public string RecordDeployment(string modelName, string fromVersion, string toVersion, string deployedBy)
    {
        if (string.IsNullOrWhiteSpace(modelName))
            throw new ArgumentException("Model name cannot be null or whitespace", nameof(modelName));
        if (string.IsNullOrWhiteSpace(fromVersion))
            throw new ArgumentException("From version cannot be null or whitespace", nameof(fromVersion));
        if (string.IsNullOrWhiteSpace(toVersion))
            throw new ArgumentException("To version cannot be null or whitespace", nameof(toVersion));
        if (string.IsNullOrWhiteSpace(deployedBy))
            throw new ArgumentException("Deployed by cannot be null or whitespace", nameof(deployedBy));

        var deploymentId = Guid.NewGuid().ToString();
        var deploymentRecord = new DeploymentRecord(
            deploymentId,
            modelName,
            fromVersion,
            toVersion,
            DateTime.UtcNow,
            deployedBy,
            DeploymentStatus.Success);

        lock (_lock)
        {
            // Add to deployment history
            if (!_deploymentHistory.ContainsKey(modelName))
            {
                _deploymentHistory[modelName] = new List<DeploymentRecord>();
            }

            _deploymentHistory[modelName].Insert(0, deploymentRecord);

            // Limit history size (LRU - keep most recent)
            while (_deploymentHistory[modelName].Count > _maxHistorySize)
            {
                var oldest = _deploymentHistory[modelName].Last();
                _deploymentHistory[modelName].RemoveAt(_deploymentHistory[modelName].Count - 1);
                _deploymentsById.Remove(oldest.DeploymentId);
            }

            // Add to deployments by ID
            _deploymentsById[deploymentId] = deploymentRecord;

            _logger?.LogInformation(
                "Recorded deployment {DeploymentId} for model {ModelName}: {FromVersion} -> {ToVersion} by {DeployedBy}",
                deploymentId, modelName, fromVersion, toVersion, deployedBy);
        }

        return deploymentId;
    }

    /// <inheritdoc />
    public async Task<RollbackResult> RollbackAsync(string deploymentId, string reason, string initiatedBy)
    {
        if (string.IsNullOrWhiteSpace(deploymentId))
            throw new ArgumentException("Deployment ID cannot be null or whitespace", nameof(deploymentId));
        if (string.IsNullOrWhiteSpace(reason))
            throw new ArgumentException("Reason cannot be null or whitespace", nameof(reason));
        if (string.IsNullOrWhiteSpace(initiatedBy))
            throw new ArgumentException("Initiated by cannot be null or whitespace", nameof(initiatedBy));

        DeploymentRecord? currentDeployment;
        DeploymentRecord? previousDeployment;

        lock (_lock)
        {
            if (!_deploymentsById.TryGetValue(deploymentId, out currentDeployment))
            {
                return new RollbackResult(
                    false,
                    "",
                    deploymentId,
                    DateTime.UtcNow,
                    $"Deployment {deploymentId} not found");
            }

            if (currentDeployment.Status != DeploymentStatus.Success)
            {
                return new RollbackResult(
                    false,
                    "",
                    deploymentId,
                    DateTime.UtcNow,
                    $"Cannot rollback deployment {deploymentId} with status {currentDeployment.Status}");
            }

            // Find previous deployment
            var history = _deploymentHistory[currentDeployment.ModelName];
            var currentIndex = history.FindIndex(d => d.DeploymentId == deploymentId);

            if (currentIndex < 0 || currentIndex >= history.Count - 1)
            {
                return new RollbackResult(
                    false,
                    "",
                    deploymentId,
                    DateTime.UtcNow,
                    $"Cannot rollback deployment {deploymentId}: no previous deployment available");
            }

            previousDeployment = history[currentIndex + 1];
        }

        try
        {
            _logger?.LogInformation(
                "Starting rollback for deployment {DeploymentId} to previous deployment {PreviousDeploymentId}: {Reason}",
                deploymentId, previousDeployment.DeploymentId, reason);

            // 1. Load previous version (using hotswap logic)
            var swapOperation = await _modelHotswapper.SwapVersionAsync(
                currentDeployment.ModelName,
                currentDeployment.ToVersion,
                previousDeployment.ToVersion);

            // 2. Update router to route to previous version
            await _versionRouter.UpdateRoutingAsync(currentDeployment.ModelName, previousDeployment.ToVersion);

            // 3. Wait for current version to drain
            await _versionRouter.WaitForDrainAsync(currentDeployment.ModelName, TimeSpan.FromSeconds(30));

            // 4. Mark current deployment as rolled back
            lock (_lock)
            {
                currentDeployment.MarkAsRolledBack();
                currentDeployment.Reason = $"Rolled back: {reason}";

                // Create new deployment record for the rollback
                var rollbackDeploymentId = RecordDeployment(
                    currentDeployment.ModelName,
                    currentDeployment.ToVersion,
                    previousDeployment.ToVersion,
                    initiatedBy);

                var rollbackDeployment = _deploymentsById[rollbackDeploymentId];
                rollbackDeployment.Reason = $"Rollback from deployment {deploymentId}: {reason}";
            }

            _logger?.LogInformation(
                "Successfully rolled back from deployment {DeploymentId} to {PreviousDeploymentId}",
                deploymentId, previousDeployment.DeploymentId);

            return new RollbackResult(
                true,
                previousDeployment.DeploymentId,
                deploymentId,
                DateTime.UtcNow,
                $"Successfully rolled back to version {previousDeployment.ToVersion}");
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Failed to rollback deployment {DeploymentId}", deploymentId);
            return new RollbackResult(
                false,
                previousDeployment?.DeploymentId ?? "",
                deploymentId,
                DateTime.UtcNow,
                $"Rollback failed: {ex.Message}");
        }
    }

    /// <inheritdoc />
    public async Task<RollbackResult> RollbackToVersionAsync(string modelName, string version, string reason, string initiatedBy)
    {
        if (string.IsNullOrWhiteSpace(modelName))
            throw new ArgumentException("Model name cannot be null or whitespace", nameof(modelName));
        if (string.IsNullOrWhiteSpace(version))
            throw new ArgumentException("Version cannot be null or whitespace", nameof(version));
        if (string.IsNullOrWhiteSpace(reason))
            throw new ArgumentException("Reason cannot be null or whitespace", nameof(reason));
        if (string.IsNullOrWhiteSpace(initiatedBy))
            throw new ArgumentException("Initiated by cannot be null or whitespace", nameof(initiatedBy));

        DeploymentRecord? targetDeployment;
        DeploymentRecord? currentDeployment;

        lock (_lock)
        {
            if (!_deploymentHistory.ContainsKey(modelName))
            {
                return new RollbackResult(
                    false,
                    "",
                    "",
                    DateTime.UtcNow,
                    $"No deployment history found for model {modelName}");
            }

            // Find the deployment that deployed this version
            var history = _deploymentHistory[modelName];
            targetDeployment = history.FirstOrDefault(d => d.ToVersion == version);

            if (targetDeployment == null)
            {
                return new RollbackResult(
                    false,
                    "",
                    "",
                    DateTime.UtcNow,
                    $"Version {version} not found in deployment history for model {modelName}");
            }

            // Get the current deployment (most recent successful one)
            currentDeployment = history.FirstOrDefault(d => d.Status == DeploymentStatus.Success);

            if (currentDeployment == null)
            {
                return new RollbackResult(
                    false,
                    "",
                    "",
                    DateTime.UtcNow,
                    $"No current deployment found for model {modelName}");
            }

            if (currentDeployment.DeploymentId == targetDeployment.DeploymentId)
            {
                return new RollbackResult(
                    false,
                    "",
                    "",
                    DateTime.UtcNow,
                    $"Model {modelName} is already on version {version}");
            }
        }

        // Rollback from current deployment
        return await RollbackAsync(currentDeployment.DeploymentId, reason, initiatedBy);
    }

    /// <inheritdoc />
    public IEnumerable<DeploymentRecord> GetDeploymentHistory(string modelName, int limit = 10)
    {
        if (string.IsNullOrWhiteSpace(modelName))
            throw new ArgumentException("Model name cannot be null or whitespace", nameof(modelName));

        if (limit <= 0)
            limit = 10;

        lock (_lock)
        {
            if (!_deploymentHistory.ContainsKey(modelName))
            {
                return Enumerable.Empty<DeploymentRecord>();
            }

            return _deploymentHistory[modelName].Take(limit).ToList();
        }
    }

    /// <inheritdoc />
    public DeploymentRecord? GetDeployment(string deploymentId)
    {
        if (string.IsNullOrWhiteSpace(deploymentId))
            throw new ArgumentException("Deployment ID cannot be null or whitespace", nameof(deploymentId));

        lock (_lock)
        {
            _deploymentsById.TryGetValue(deploymentId, out var deployment);
            return deployment;
        }
    }

    /// <inheritdoc />
    public bool CanRollback(string deploymentId)
    {
        if (string.IsNullOrWhiteSpace(deploymentId))
            return false;

        lock (_lock)
        {
            if (!_deploymentsById.TryGetValue(deploymentId, out var deployment))
            {
                return false;
            }

            if (deployment.Status != DeploymentStatus.Success)
            {
                return false;
            }

            if (!_deploymentHistory.ContainsKey(deployment.ModelName))
            {
                return false;
            }

            var history = _deploymentHistory[deployment.ModelName];
            var currentIndex = history.FindIndex(d => d.DeploymentId == deploymentId);

            return currentIndex >= 0 && currentIndex < history.Count - 1;
        }
    }

    /// <inheritdoc />
    public void SetAutoRollbackThreshold(string modelName, float errorRateThreshold, TimeSpan observationWindow)
    {
        if (string.IsNullOrWhiteSpace(modelName))
            throw new ArgumentException("Model name cannot be null or whitespace", nameof(modelName));
        if (errorRateThreshold < 0.0f || errorRateThreshold > 1.0f)
            throw new ArgumentOutOfRangeException(nameof(errorRateThreshold), "Error rate threshold must be between 0.0 and 1.0");
        if (observationWindow <= TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(observationWindow), "Observation window must be positive");

        lock (_lock)
        {
            _autoRollbackConfigs[modelName] = new AutoRollbackConfig
            {
                ErrorRateThreshold = errorRateThreshold,
                ObservationWindow = observationWindow
            };

            _logger?.LogInformation(
                "Set auto-rollback threshold for model {ModelName}: {Threshold} over {Window}",
                modelName, errorRateThreshold, observationWindow);
        }
    }

    /// <inheritdoc />
    public void MonitorErrorRate(string modelName, string version, float currentErrorRate)
    {
        if (string.IsNullOrWhiteSpace(modelName))
            throw new ArgumentException("Model name cannot be null or whitespace", nameof(modelName));
        if (string.IsNullOrWhiteSpace(version))
            throw new ArgumentException("Version cannot be null or whitespace", nameof(version));
        if (currentErrorRate < 0.0f || currentErrorRate > 1.0f)
            throw new ArgumentOutOfRangeException(nameof(currentErrorRate), "Error rate must be between 0.0 and 1.0");

        lock (_lock)
        {
            if (!_autoRollbackConfigs.ContainsKey(modelName))
            {
                return; // No auto-rollback configured for this model
            }

            var config = _autoRollbackConfigs[modelName];
            var monitorKey = $"{modelName}:{version}";

            if (!_errorRateMonitors.ContainsKey(monitorKey))
            {
                _errorRateMonitors[monitorKey] = new ErrorRateMonitor();
            }

            var monitor = _errorRateMonitors[monitorKey];

            // Add error rate sample
            monitor.AddSample(currentErrorRate);

            // Check if average error rate exceeds threshold within observation window
            var averageErrorRate = monitor.GetAverageErrorRate(config.ObservationWindow);

            if (averageErrorRate >= config.ErrorRateThreshold)
            {
                _logger?.LogWarning(
                    "Auto-rollback triggered for model {ModelName} version {Version}: error rate {ErrorRate} exceeds threshold {Threshold}",
                    modelName, version, averageErrorRate, config.ErrorRateThreshold);

                // Find current deployment for this model/version and trigger rollback
                var history = _deploymentHistory[modelName];
                var currentDeployment = history.FirstOrDefault(d =>
                    d.ToVersion == version && d.Status == DeploymentStatus.Success);

                if (currentDeployment != null)
                {
                    // Trigger rollback asynchronously
                    Task.Run(async () =>
                    {
                        await RollbackAsync(
                            currentDeployment.DeploymentId,
                            $"Auto-rollback: error rate {averageErrorRate:P2} exceeded threshold {config.ErrorRateThreshold:P2}",
                            "AutoRollbackSystem");
                    });
                }
            }
        }
    }

    /// <summary>
    /// Configuration for auto-rollback
    /// </summary>
    private class AutoRollbackConfig
    {
        public float ErrorRateThreshold { get; set; }
        public TimeSpan ObservationWindow { get; set; }
    }

    /// <summary>
    /// Monitor for error rate samples
    /// </summary>
    private class ErrorRateMonitor
    {
        private readonly List<(DateTime timestamp, float errorRate)> _samples = new List<(DateTime, float)>();

        public void AddSample(float errorRate)
        {
            _samples.Add((DateTime.UtcNow, errorRate));

            // Remove old samples (older than 1 hour)
            var cutoff = DateTime.UtcNow.AddHours(-1);
            _samples.RemoveAll(s => s.timestamp < cutoff);
        }

        public float GetAverageErrorRate(TimeSpan observationWindow)
        {
            var cutoff = DateTime.UtcNow.Subtract(observationWindow);
            var recentSamples = _samples.Where(s => s.timestamp >= cutoff).ToList();

            if (recentSamples.Count == 0)
            {
                return 0.0f;
            }

            return recentSamples.Average(s => s.errorRate);
        }
    }
}
