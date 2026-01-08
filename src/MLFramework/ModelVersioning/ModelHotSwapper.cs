using MLFramework.Serving.Routing;

namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Implementation of <see cref="IModelHotSwapper"/> for zero-downtime model version swapping.
    /// </summary>
    public class ModelHotSwapper : IModelHotSwapper
    {
        private readonly IModelVersionManager _versionManager;
        private readonly IVersionRouter _router;
        private readonly Dictionary<string, SwapStatus> _swapStatuses;
        private readonly Dictionary<string, int> _inFlightRequests;
        private readonly object _swapLock;
        private const int DefaultDrainTimeoutMs = 30000; // 30 seconds
        private const long MaxMemoryUsageBytes = 4L * 1024 * 1024 * 1024; // 4 GB

        /// <summary>
        /// Initializes a new instance of the <see cref="ModelHotSwapper"/> class.
        /// </summary>
        /// <param name="versionManager">The version manager.</param>
        /// <param name="router">The version router.</param>
        public ModelHotSwapper(
            IModelVersionManager versionManager,
            IVersionRouter router)
        {
            _versionManager = versionManager ?? throw new ArgumentNullException(nameof(versionManager));
            _router = router ?? throw new ArgumentNullException(nameof(router));
            _swapStatuses = new Dictionary<string, SwapStatus>();
            _inFlightRequests = new Dictionary<string, int>();
            _swapLock = new object();
        }

        /// <inheritdoc/>
        public async Task<SwapResult> SwapVersion(string modelId, string fromVersion, string toVersion)
        {
            var startTime = DateTime.UtcNow;
            var result = new SwapResult
            {
                StartTime = startTime
            };

            try
            {
                // Validate inputs
                if (string.IsNullOrWhiteSpace(modelId))
                {
                    throw new ArgumentException("Model ID cannot be null or whitespace", nameof(modelId));
                }
                if (string.IsNullOrWhiteSpace(fromVersion))
                {
                    throw new ArgumentException("From version cannot be null or whitespace", nameof(fromVersion));
                }
                if (string.IsNullOrWhiteSpace(toVersion))
                {
                    throw new ArgumentException("To version cannot be null or whitespace", nameof(toVersion));
                }
                if (fromVersion == toVersion)
                {
                    throw new ArgumentException("Cannot swap to the same version", nameof(toVersion));
                }

                // Check if swap is already in progress
                lock (_swapLock)
                {
                    if (_swapStatuses.ContainsKey(modelId))
                    {
                        var existingStatus = _swapStatuses[modelId];
                        if (existingStatus.State == SwapState.Draining || existingStatus.State == SwapState.Swapping)
                        {
                            throw new InvalidOperationException($"Swap already in progress for model {modelId}");
                        }
                    }

                    // Initialize swap status
                    _swapStatuses[modelId] = new SwapStatus
                    {
                        ModelId = modelId,
                        CurrentVersion = fromVersion,
                        TargetVersion = toVersion,
                        State = SwapState.Draining,
                        StartTime = startTime,
                        PendingRequests = GetInFlightRequestCount(modelId)
                    };
                }

                // Step 1: Validate both versions exist and are loaded
                if (!_versionManager.IsVersionLoaded(modelId, fromVersion))
                {
                    throw new InvalidOperationException($"Source version {fromVersion} is not loaded");
                }

                // Step 2: Check health of target version
                var healthCheck = CheckVersionHealth(modelId, toVersion);
                if (!healthCheck.IsHealthy)
                {
                    throw new InvalidOperationException($"Target version {toVersion} is unhealthy: {healthCheck.Message}");
                }

                // Step 3: Load target version if not loaded (placeholder - would load from registry)
                if (!_versionManager.IsVersionLoaded(modelId, toVersion))
                {
                    // In a real implementation, this would load from a registry
                    // For now, we'll assume it's already loaded or throw an exception
                    throw new InvalidOperationException($"Target version {toVersion} is not loaded. Please load it first.");
                }

                // Step 4: Drain requests from source version
                UpdateSwapStatus(modelId, SwapState.Draining);
                var drainedSuccessfully = await DrainVersion(modelId, fromVersion, TimeSpan.FromMilliseconds(DefaultDrainTimeoutMs));

                if (!drainedSuccessfully)
                {
                    UpdateSwapStatus(modelId, SwapState.Failed);
                    throw new TimeoutException($"Failed to drain requests from version {fromVersion} within timeout");
                }

                result.RequestsDrained = GetInFlightRequestCount(modelId);

                // Step 5: Update routing policy
                UpdateSwapStatus(modelId, SwapState.Swapping);
                _router.SetDefaultVersion(modelId, toVersion);

                // Step 6: Verify complete switch
                var currentDefault = _router.GetDefaultVersion(modelId);
                if (currentDefault != toVersion)
                {
                    UpdateSwapStatus(modelId, SwapState.Failed);
                    throw new InvalidOperationException("Failed to update routing policy");
                }

                // Step 7: Complete the swap
                UpdateSwapStatus(modelId, SwapState.Completed);
                result.RequestsRemaining = GetInFlightRequestCount(modelId);
                result.Success = true;
                result.Message = $"Successfully swapped from {fromVersion} to {toVersion}";
                result.EndTime = DateTime.UtcNow;

                // Step 8: Optionally unload source version (not done for safety)

                return result;
            }
            catch (Exception ex)
            {
                UpdateSwapStatus(modelId, SwapState.Failed);
                result.Success = false;
                result.Message = $"Swap failed: {ex.Message}";
                result.EndTime = DateTime.UtcNow;
                return result;
            }
        }

        /// <inheritdoc/>
        public async Task<RollbackResult> RollbackVersion(string modelId, string targetVersion)
        {
            var rollbackTime = DateTime.UtcNow;
            var result = new RollbackResult
            {
                RollbackTime = rollbackTime
            };

            try
            {
                // Validate inputs
                if (string.IsNullOrWhiteSpace(modelId))
                {
                    throw new ArgumentException("Model ID cannot be null or whitespace", nameof(modelId));
                }
                if (string.IsNullOrWhiteSpace(targetVersion))
                {
                    throw new ArgumentException("Target version cannot be null or whitespace", nameof(targetVersion));
                }

                // Get current version
                var currentVersion = _router.GetDefaultVersion(modelId);
                result.PreviousVersion = currentVersion;

                // Step 1: Validate target version exists and is loaded
                if (!_versionManager.IsVersionLoaded(modelId, targetVersion))
                {
                    throw new InvalidOperationException($"Target version {targetVersion} is not loaded");
                }

                // Step 2: Check health of target version
                var healthCheck = CheckVersionHealth(modelId, targetVersion);
                if (!healthCheck.IsHealthy)
                {
                    throw new InvalidOperationException($"Target version {targetVersion} is unhealthy: {healthCheck.Message}");
                }

                // Step 3: Update routing policy immediately (rollback is instant)
                _router.SetDefaultVersion(modelId, targetVersion);
                result.NewVersion = targetVersion;

                // Step 4: Drain from current version (now previous version)
                await DrainVersion(modelId, currentVersion ?? string.Empty, TimeSpan.FromMilliseconds(DefaultDrainTimeoutMs));

                // Reset swap status to Idle after rollback
                lock (_swapLock)
                {
                    if (_swapStatuses.ContainsKey(modelId))
                    {
                        _swapStatuses[modelId].State = SwapState.Idle;
                    }
                }

                result.Success = true;
                result.Message = $"Successfully rolled back to {targetVersion}";
                return result;
            }
            catch (Exception ex)
            {
                result.Success = false;
                result.Message = $"Rollback failed: {ex.Message}";
                return result;
            }
        }

        /// <inheritdoc/>
        public HealthCheckResult CheckVersionHealth(string modelId, string version)
        {
            var diagnostics = new Dictionary<string, object>();

            try
            {
                // Validate inputs
                if (string.IsNullOrWhiteSpace(modelId))
                {
                    throw new ArgumentException("Model ID cannot be null or whitespace", nameof(modelId));
                }
                if (string.IsNullOrWhiteSpace(version))
                {
                    throw new ArgumentException("Version cannot be null or whitespace", nameof(version));
                }

                // Step 1: Check if version is loaded
                var isLoaded = _versionManager.IsVersionLoaded(modelId, version);
                diagnostics["IsLoaded"] = isLoaded;

                if (!isLoaded)
                {
                    return HealthCheckResult.Unhealthy($"Version {version} is not loaded");
                }

                // Step 2: Run sample inference (placeholder)
                // In a real implementation, this would run actual inference
                diagnostics["InferenceTest"] = "Passed (placeholder)";

                // Step 3: Validate inference result format (placeholder)
                diagnostics["ResultFormat"] = "Valid (placeholder)";

                // Step 4: Check memory usage
                var loadInfo = _versionManager.GetLoadInfo(modelId, version);
                var memoryUsageOk = loadInfo.MemoryUsageBytes < MaxMemoryUsageBytes;
                diagnostics["MemoryUsageBytes"] = loadInfo.MemoryUsageBytes;
                diagnostics["MemoryUsageOK"] = memoryUsageOk;

                if (!memoryUsageOk)
                {
                    return new HealthCheckResult
                    {
                        IsHealthy = false,
                        Message = $"Memory usage exceeds limit: {loadInfo.MemoryUsageBytes} bytes",
                        CheckTimestamp = DateTime.UtcNow,
                        Diagnostics = diagnostics
                    };
                }

                return new HealthCheckResult
                {
                    IsHealthy = true,
                    Message = "Version is healthy",
                    CheckTimestamp = DateTime.UtcNow,
                    Diagnostics = diagnostics
                };
            }
            catch (Exception ex)
            {
                return new HealthCheckResult
                {
                    IsHealthy = false,
                    Message = $"Health check failed: {ex.Message}",
                    CheckTimestamp = DateTime.UtcNow,
                    Diagnostics = diagnostics
                };
            }
        }

        /// <inheritdoc/>
        public async Task<bool> DrainVersion(string modelId, string version, TimeSpan timeout)
        {
            var startTime = DateTime.UtcNow;
            var key = $"{modelId}:{version}";

            try
            {
                // Validate inputs
                if (string.IsNullOrWhiteSpace(modelId))
                {
                    throw new ArgumentException("Model ID cannot be null or whitespace", nameof(modelId));
                }
                if (string.IsNullOrWhiteSpace(version))
                {
                    throw new ArgumentException("Version cannot be null or whitespace", nameof(version));
                }

                // Initialize request count if not exists
                lock (_inFlightRequests)
                {
                    if (!_inFlightRequests.ContainsKey(key))
                    {
                        _inFlightRequests[key] = 0;
                    }
                }

                // Wait for in-flight requests to complete
                while ((DateTime.UtcNow - startTime) < timeout)
                {
                    int requestCount;
                    lock (_inFlightRequests)
                    {
                        requestCount = _inFlightRequests.ContainsKey(key) ? _inFlightRequests[key] : 0;
                    }

                    if (requestCount == 0)
                    {
                        return true; // Successfully drained
                    }

                    // Wait a bit before checking again
                    await Task.Delay(100);
                }

                // Timeout occurred
                return false;
            }
            catch (Exception)
            {
                return false;
            }
        }

        /// <inheritdoc/>
        public SwapStatus GetSwapStatus(string modelId)
        {
            lock (_swapLock)
            {
                if (_swapStatuses.TryGetValue(modelId, out var status))
                {
                    // Update pending requests count
                    status.PendingRequests = GetInFlightRequestCount(modelId);
                    return status;
                }

                // Return default idle status if no swap in progress
                return new SwapStatus
                {
                    ModelId = modelId,
                    CurrentVersion = string.Empty,
                    TargetVersion = string.Empty,
                    State = SwapState.Idle,
                    StartTime = DateTime.UtcNow,
                    PendingRequests = 0
                };
            }
        }

        /// <summary>
        /// Increments the in-flight request count for a model version.
        /// </summary>
        /// <param name="modelId">The model identifier.</param>
        /// <param name="version">The version.</param>
        public void IncrementInFlightRequest(string modelId, string version)
        {
            var key = $"{modelId}:{version}";
            lock (_inFlightRequests)
            {
                if (!_inFlightRequests.ContainsKey(key))
                {
                    _inFlightRequests[key] = 0;
                }
                _inFlightRequests[key]++;
            }
        }

        /// <summary>
        /// Decrements the in-flight request count for a model version.
        /// </summary>
        /// <param name="modelId">The model identifier.</param>
        /// <param name="version">The version.</param>
        public void DecrementInFlightRequest(string modelId, string version)
        {
            var key = $"{modelId}:{version}";
            lock (_inFlightRequests)
            {
                if (_inFlightRequests.ContainsKey(key) && _inFlightRequests[key] > 0)
                {
                    _inFlightRequests[key]--;
                }
            }
        }

        private void UpdateSwapStatus(string modelId, SwapState state)
        {
            lock (_swapLock)
            {
                if (_swapStatuses.ContainsKey(modelId))
                {
                    _swapStatuses[modelId].State = state;
                }
            }
        }

        private int GetInFlightRequestCount(string modelId)
        {
            // Get total in-flight requests for all versions of this model
            int count = 0;
            lock (_inFlightRequests)
            {
                foreach (var kvp in _inFlightRequests)
                {
                    if (kvp.Key.StartsWith($"{modelId}:"))
                    {
                        count += kvp.Value;
                    }
                }
            }
            return count;
        }
    }
}
