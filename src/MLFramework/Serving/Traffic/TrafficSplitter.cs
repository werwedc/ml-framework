using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

namespace MLFramework.Serving.Traffic
{
    /// <summary>
    /// Thread-safe implementation of traffic distribution between multiple model versions.
    /// Uses deterministic hashing to ensure consistent routing for the same request ID.
    /// </summary>
    public class TrafficSplitter : ITrafficSplitter
    {
        private readonly ConcurrentDictionary<string, TrafficSplitConfig> _splits;
        private readonly object _lock = new object();

        public TrafficSplitter()
        {
            _splits = new ConcurrentDictionary<string, TrafficSplitConfig>();
        }

        /// <summary>
        /// Configures the traffic split for a specific model.
        /// </summary>
        /// <exception cref="ArgumentException">Thrown when percentages don't sum to 1.0, contain negative values, or are out of range</exception>
        /// <exception cref="ArgumentNullException">Thrown when modelName or versionPercentages is null</exception>
        public void SetTrafficSplit(string modelName, Dictionary<string, float> versionPercentages)
        {
            if (string.IsNullOrWhiteSpace(modelName))
            {
                throw new ArgumentNullException(nameof(modelName), "Model name cannot be null or empty");
            }

            if (versionPercentages == null)
            {
                throw new ArgumentNullException(nameof(versionPercentages), "Version percentages cannot be null");
            }

            if (versionPercentages.Count == 0)
            {
                throw new ArgumentException("Version percentages cannot be empty", nameof(versionPercentages));
            }

            // Validate all percentages
            float total = 0.0f;
            foreach (var kvp in versionPercentages)
            {
                if (string.IsNullOrWhiteSpace(kvp.Key))
                {
                    throw new ArgumentException("Version name cannot be null or empty", nameof(versionPercentages));
                }

                if (kvp.Value < 0.0f || kvp.Value > 1.0f)
                {
                    throw new ArgumentException($"Percentage for version '{kvp.Key}' must be between 0.0 and 1.0", nameof(versionPercentages));
                }

                if (kvp.Value < 0.0f)
                {
                    throw new ArgumentException($"Negative percentage not allowed for version '{kvp.Key}'", nameof(versionPercentages));
                }

                total += kvp.Value;
            }

            // Check if percentages sum to approximately 1.0 (allowing for floating point precision)
            if (Math.Abs(total - 1.0f) > 0.0001f)
            {
                throw new ArgumentException($"Percentages must sum to 1.0, but sum is {total:F4}", nameof(versionPercentages));
            }

            // Create and configure the split
            var config = new TrafficSplitConfig
            {
                VersionPercentages = new Dictionary<string, float>(versionPercentages),
                LastUpdated = DateTime.UtcNow,
                UpdatedBy = Environment.UserName
            };

            config.RebuildRanges();

            // Store the configuration (thread-safe)
            _splits[modelName] = config;
        }

        /// <summary>
        /// Selects a model version for the given request based on configured traffic split.
        /// Uses deterministic hashing to ensure the same request ID consistently routes to the same version.
        /// </summary>
        /// <exception cref="KeyNotFoundException">Thrown when no traffic split is configured for the model</exception>
        public string SelectVersion(string modelName, string requestId)
        {
            if (string.IsNullOrWhiteSpace(modelName))
            {
                throw new ArgumentNullException(nameof(modelName), "Model name cannot be null or empty");
            }

            if (string.IsNullOrWhiteSpace(requestId))
            {
                throw new ArgumentNullException(nameof(requestId), "Request ID cannot be null or empty");
            }

            // Get the traffic split configuration
            if (!_splits.TryGetValue(modelName, out var config))
            {
                throw new KeyNotFoundException($"No traffic split configured for model '{modelName}'");
            }

            // Generate deterministic hash value (0.0 to 1.0)
            float hashValue = ComputeDeterministicHash(modelName + ":" + requestId);

            // Find the version based on cached ranges
            foreach (var range in config.CachedRanges)
            {
                if (hashValue >= range.Start && hashValue < range.End)
                {
                    return range.Version;
                }
            }

            // Handle edge case for hashValue == 1.0 (should be rare due to float precision)
            return config.CachedRanges.Last().Version;
        }

        /// <summary>
        /// Retrieves the current traffic split configuration for a model.
        /// </summary>
        public TrafficSplitConfig GetTrafficSplit(string modelName)
        {
            if (string.IsNullOrWhiteSpace(modelName))
            {
                throw new ArgumentNullException(nameof(modelName), "Model name cannot be null or empty");
            }

            _splits.TryGetValue(modelName, out var config);
            return config;
        }

        /// <summary>
        /// Removes the traffic split configuration for a model.
        /// </summary>
        public void ClearTrafficSplit(string modelName)
        {
            if (string.IsNullOrWhiteSpace(modelName))
            {
                throw new ArgumentNullException(nameof(modelName), "Model name cannot be null or empty");
            }

            _splits.TryRemove(modelName, out _);
        }

        /// <summary>
        /// Gets the allocation percentage for a specific model version.
        /// </summary>
        public float GetVersionAllocation(string modelName, string version)
        {
            if (string.IsNullOrWhiteSpace(modelName))
            {
                throw new ArgumentNullException(nameof(modelName), "Model name cannot be null or empty");
            }

            if (string.IsNullOrWhiteSpace(version))
            {
                throw new ArgumentNullException(nameof(version), "Version cannot be null or empty");
            }

            if (_splits.TryGetValue(modelName, out var config))
            {
                if (config.VersionPercentages.TryGetValue(version, out var percentage))
                {
                    return percentage;
                }
            }

            return 0.0f;
        }

        /// <summary>
        /// Computes a deterministic hash value (0.0 to 1.0) from a string.
        /// Uses SHA-256 for consistent distribution.
        /// </summary>
        private float ComputeDeterministicHash(string input)
        {
            using (var sha256 = SHA256.Create())
            {
                byte[] hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(input));

                // Use first 4 bytes to generate a float between 0 and 1
                uint hashValue = BitConverter.ToUInt32(hashBytes, 0);

                // Normalize to [0.0, 1.0) range
                return hashValue / (float)uint.MaxValue;
            }
        }
    }
}
