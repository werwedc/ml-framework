using System.Text.RegularExpressions;

namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Implementation of IModelRegistry for managing model versions.
    /// </summary>
    public class ModelRegistry : IModelRegistry
    {
        private readonly Dictionary<string, ModelInfo> _modelsById;
        private readonly Dictionary<string, ModelInfo> _modelsByVersion;
        private readonly object _registryLock;

        // Semantic version pattern: v1.2.3
        private static readonly Regex VersionTagPattern = new Regex(@"^v\d+\.\d+\.\d+$", RegexOptions.Compiled);

        // Valid state transitions
        private static readonly Dictionary<LifecycleState, HashSet<LifecycleState>> ValidTransitions = new()
        {
            { LifecycleState.Draft, new HashSet<LifecycleState> { LifecycleState.Staging, LifecycleState.Archived } },
            { LifecycleState.Staging, new HashSet<LifecycleState> { LifecycleState.Production, LifecycleState.Draft, LifecycleState.Archived } },
            { LifecycleState.Production, new HashSet<LifecycleState> { LifecycleState.Staging, LifecycleState.Archived } },
            { LifecycleState.Archived, new HashSet<LifecycleState>() } // No transitions from Archived
        };

        /// <summary>
        /// Initializes a new instance of the ModelRegistry class.
        /// </summary>
        public ModelRegistry()
        {
            _modelsById = new Dictionary<string, ModelInfo>(StringComparer.OrdinalIgnoreCase);
            _modelsByVersion = new Dictionary<string, ModelInfo>(StringComparer.OrdinalIgnoreCase);
            _registryLock = new object();
        }

        /// <inheritdoc />
        public string RegisterModel(string modelPath, ModelMetadata metadata)
        {
            if (string.IsNullOrWhiteSpace(modelPath))
            {
                throw new ArgumentException("Model path cannot be null or whitespace.", nameof(modelPath));
            }

            if (metadata == null)
            {
                throw new ArgumentNullException(nameof(metadata));
            }

            lock (_registryLock)
            {
                // Generate unique model ID
                string modelId = Guid.NewGuid().ToString();

                // Create ModelInfo with initial state
                var modelInfo = new ModelInfo
                {
                    ModelId = modelId,
                    ModelPath = modelPath,
                    Metadata = metadata,
                    State = LifecycleState.Draft,
                    CreatedAt = DateTime.UtcNow,
                    UpdatedAt = DateTime.UtcNow
                };

                // Store in modelsById dictionary
                _modelsById[modelId] = modelInfo;

                return modelId;
            }
        }

        /// <inheritdoc />
        public void TagModel(string modelId, string versionTag)
        {
            if (string.IsNullOrWhiteSpace(modelId))
            {
                throw new ArgumentException("Model ID cannot be null or whitespace.", nameof(modelId));
            }

            if (string.IsNullOrWhiteSpace(versionTag))
            {
                throw new ArgumentException("Version tag cannot be null or whitespace.", nameof(versionTag));
            }

            // Validate version tag format
            if (!VersionTagPattern.IsMatch(versionTag))
            {
                throw new ArgumentException(
                    $"Version tag must match semantic versioning pattern (e.g., v1.2.3). Got: {versionTag}",
                    nameof(versionTag));
            }

            lock (_registryLock)
            {
                // Check if model exists
                if (!_modelsById.TryGetValue(modelId, out var modelInfo))
                {
                    throw new KeyNotFoundException($"Model with ID '{modelId}' not found.");
                }

                // Check if version tag already exists
                if (_modelsByVersion.ContainsKey(versionTag))
                {
                    throw new InvalidOperationException($"Version tag '{versionTag}' is already in use.");
                }

                // Update model info
                modelInfo.VersionTag = versionTag;
                modelInfo.UpdatedAt = DateTime.UtcNow;

                // Store in modelsByVersion dictionary
                _modelsByVersion[versionTag] = modelInfo;
            }
        }

        /// <inheritdoc />
        public ModelInfo? GetModel(string versionTag)
        {
            if (string.IsNullOrWhiteSpace(versionTag))
            {
                throw new ArgumentException("Version tag cannot be null or whitespace.", nameof(versionTag));
            }

            lock (_registryLock)
            {
                return _modelsByVersion.TryGetValue(versionTag, out var modelInfo) ? modelInfo : null;
            }
        }

        /// <inheritdoc />
        public ModelInfo? GetModelById(string modelId)
        {
            if (string.IsNullOrWhiteSpace(modelId))
            {
                throw new ArgumentException("Model ID cannot be null or whitespace.", nameof(modelId));
            }

            lock (_registryLock)
            {
                return _modelsById.TryGetValue(modelId, out var modelInfo) ? modelInfo : null;
            }
        }

        /// <inheritdoc />
        public IEnumerable<ModelInfo> ListModels()
        {
            lock (_registryLock)
            {
                return _modelsById.Values
                    .OrderBy(m => m.CreatedAt)
                    .ToList();
            }
        }

        /// <inheritdoc />
        public IEnumerable<ModelInfo> ListModels(LifecycleState state)
        {
            lock (_registryLock)
            {
                return _modelsById.Values
                    .Where(m => m.State == state)
                    .OrderBy(m => m.CreatedAt)
                    .ToList();
            }
        }

        /// <inheritdoc />
        public void UpdateModelState(string modelId, LifecycleState newState)
        {
            if (string.IsNullOrWhiteSpace(modelId))
            {
                throw new ArgumentException("Model ID cannot be null or whitespace.", nameof(modelId));
            }

            lock (_registryLock)
            {
                // Check if model exists
                if (!_modelsById.TryGetValue(modelId, out var modelInfo))
                {
                    throw new KeyNotFoundException($"Model with ID '{modelId}' not found.");
                }

                // Check if state transition is valid
                if (!IsValidStateTransition(modelInfo.State, newState))
                {
                    throw new InvalidOperationException(
                        $"Invalid state transition from {modelInfo.State} to {newState}. " +
                        $"Valid transitions from {modelInfo.State}: {GetValidTransitions(modelInfo.State)}");
                }

                // Update state
                modelInfo.State = newState;
                modelInfo.UpdatedAt = DateTime.UtcNow;
            }
        }

        /// <inheritdoc />
        public void SetParentModel(string modelId, string parentModelId)
        {
            if (string.IsNullOrWhiteSpace(modelId))
            {
                throw new ArgumentException("Model ID cannot be null or whitespace.", nameof(modelId));
            }

            if (string.IsNullOrWhiteSpace(parentModelId))
            {
                throw new ArgumentException("Parent model ID cannot be null or whitespace.", nameof(parentModelId));
            }

            lock (_registryLock)
            {
                // Check if model exists
                if (!_modelsById.TryGetValue(modelId, out var modelInfo))
                {
                    throw new KeyNotFoundException($"Model with ID '{modelId}' not found.");
                }

                // Validate parent model exists
                if (!_modelsById.ContainsKey(parentModelId))
                {
                    throw new KeyNotFoundException($"Parent model with ID '{parentModelId}' not found.");
                }

                // Validate parent is not the same as the model
                if (modelId.Equals(parentModelId, StringComparison.OrdinalIgnoreCase))
                {
                    throw new ArgumentException("Model cannot be its own parent.");
                }

                // Set parent model ID
                modelInfo.ParentModelId = parentModelId;
                modelInfo.UpdatedAt = DateTime.UtcNow;
            }
        }

        /// <summary>
        /// Checks if a state transition is valid.
        /// </summary>
        /// <param name="currentState">The current state.</param>
        /// <param name="newState">The desired new state.</param>
        /// <returns>True if the transition is valid, false otherwise.</returns>
        private static bool IsValidStateTransition(LifecycleState currentState, LifecycleState newState)
        {
            if (currentState == newState)
            {
                return false; // No change needed
            }

            if (!ValidTransitions.TryGetValue(currentState, out var validNextStates))
            {
                return false;
            }

            return validNextStates.Contains(newState);
        }

        /// <summary>
        /// Gets a comma-separated string of valid transition states from the given state.
        /// </summary>
        /// <param name="currentState">The current state.</param>
        /// <returns>A string representation of valid next states.</returns>
        private static string GetValidTransitions(LifecycleState currentState)
        {
            if (!ValidTransitions.TryGetValue(currentState, out var validNextStates))
            {
                return "none";
            }

            return validNextStates.Any() ? string.Join(", ", validNextStates) : "none";
        }
    }
}
