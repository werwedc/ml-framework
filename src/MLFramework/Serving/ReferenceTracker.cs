using System;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace MLFramework.Serving
{
    /// <summary>
    /// Thread-safe implementation of IReferenceTracker for tracking active inference requests per model version.
    /// </summary>
    public class ReferenceTracker : IReferenceTracker
    {
        private readonly ConcurrentDictionary<string, ModelReferenceInfo> _references;
        private readonly ConcurrentDictionary<string, SemaphoreSlim> _waitSignals;
        private readonly bool _enableReferenceLeakDetection;

        /// <summary>
        /// Initializes a new instance of the ReferenceTracker class.
        /// </summary>
        public ReferenceTracker()
            : this(enableReferenceLeakDetection: false)
        {
        }

        /// <summary>
        /// Initializes a new instance of the ReferenceTracker class.
        /// </summary>
        /// <param name="enableReferenceLeakDetection">Whether to enable reference leak detection.</param>
        public ReferenceTracker(bool enableReferenceLeakDetection)
        {
            _references = new ConcurrentDictionary<string, ModelReferenceInfo>();
            _waitSignals = new ConcurrentDictionary<string, SemaphoreSlim>();
            _enableReferenceLeakDetection = enableReferenceLeakDetection;
        }

        /// <inheritdoc/>
        public void AcquireReference(string modelName, string version, string requestId)
        {
            if (string.IsNullOrWhiteSpace(modelName))
                throw new ArgumentException("Model name cannot be null or empty.", nameof(modelName));

            if (string.IsNullOrWhiteSpace(version))
                throw new ArgumentException("Version cannot be null or empty.", nameof(version));

            if (string.IsNullOrWhiteSpace(requestId))
                throw new ArgumentException("Request ID cannot be null or empty.", nameof(requestId));

            var key = GetModelKey(modelName, version);

            var modelInfo = _references.GetOrAdd(key, _ => new ModelReferenceInfo(modelName, version));

            // Track the request ID for leak detection
            if (_enableReferenceLeakDetection)
            {
                modelInfo.AddRequest(requestId);
            }

            // Atomically increment the reference count
            Interlocked.Increment(ref modelInfo.Count);
        }

        /// <inheritdoc/>
        public void ReleaseReference(string modelName, string version, string requestId)
        {
            if (string.IsNullOrWhiteSpace(modelName))
                throw new ArgumentException("Model name cannot be null or empty.", nameof(modelName));

            if (string.IsNullOrWhiteSpace(version))
                throw new ArgumentException("Version cannot be null or empty.", nameof(version));

            if (string.IsNullOrWhiteSpace(requestId))
                throw new ArgumentException("Request ID cannot be null or empty.", nameof(requestId));

            var key = GetModelKey(modelName, version);

            if (!_references.TryGetValue(key, out var modelInfo))
            {
                throw new InvalidOperationException(
                    $"Cannot release reference for model '{modelName}' version '{version}' - no references were ever acquired.");
            }

            // Remove the request ID from tracking
            if (_enableReferenceLeakDetection)
            {
                var removed = modelInfo.RemoveRequest(requestId);
                if (!removed)
                {
                    // Log warning: reference released for unknown request ID
                }
            }

            // Atomically decrement the reference count
            var newCount = Interlocked.Decrement(ref modelInfo.Count);

            if (newCount < 0)
            {
                throw new InvalidOperationException(
                    $"Reference count went negative for model '{modelName}' version '{version}'. " +
                    $"This indicates a reference was released more times than it was acquired.");
            }

            // Signal waiting threads if count reached zero
            if (newCount == 0)
            {
                if (_waitSignals.TryGetValue(key, out var signal))
                {
                    signal.Release();
                }
            }
        }

        /// <inheritdoc/>
        public int GetReferenceCount(string modelName, string version)
        {
            if (string.IsNullOrWhiteSpace(modelName))
                throw new ArgumentException("Model name cannot be null or empty.", nameof(modelName));

            if (string.IsNullOrWhiteSpace(version))
                throw new ArgumentException("Version cannot be null or empty.", nameof(version));

            var key = GetModelKey(modelName, version);

            return _references.TryGetValue(key, out var modelInfo)
                ? Volatile.Read(ref modelInfo.Count)
                : 0;
        }

        /// <inheritdoc/>
        public bool HasReferences(string modelName, string version)
        {
            return GetReferenceCount(modelName, version) > 0;
        }

        /// <inheritdoc/>
        public async Task WaitForZeroReferencesAsync(string modelName, string version, TimeSpan timeout, CancellationToken ct = default)
        {
            if (string.IsNullOrWhiteSpace(modelName))
                throw new ArgumentException("Model name cannot be null or empty.", nameof(modelName));

            if (string.IsNullOrWhiteSpace(version))
                throw new ArgumentException("Version cannot be null or empty.", nameof(version));

            var key = GetModelKey(modelName, version);

            // Quick check: if no references exist, return immediately
            if (GetReferenceCount(modelName, version) == 0)
            {
                return;
            }

            // Get or create a signal for this model
            var signal = _waitSignals.GetOrAdd(key, _ => new SemaphoreSlim(0, int.MaxValue));

            try
            {
                // Wait for the signal or timeout
                var waited = await signal.WaitAsync(timeout, ct).ConfigureAwait(false);

                if (!waited)
                {
                    throw new TimeoutException(
                        $"Timed out waiting for zero references for model '{modelName}' version '{version}'. " +
                        $"Timeout: {timeout.TotalMilliseconds}ms. Current count: {GetReferenceCount(modelName, version)}");
                }
            }
            finally
            {
                // Clean up the signal if count is zero
                if (GetReferenceCount(modelName, version) == 0)
                {
                    _waitSignals.TryRemove(key, out _);
                    signal.Dispose();
                }
            }
        }

        /// <inheritdoc/>
        public Dictionary<string, int> GetAllReferenceCounts()
        {
            return _references.ToDictionary(
                kvp => kvp.Key,
                kvp => Volatile.Read(ref kvp.Value.Count));
        }

        /// <summary>
        /// Clears all reference tracking. This is a destructive operation and should only be used in testing or emergency scenarios.
        /// </summary>
        public void ClearAll()
        {
            _references.Clear();

            // Dispose all wait signals
            foreach (var signal in _waitSignals.Values)
            {
                signal.Dispose();
            }

            _waitSignals.Clear();
        }

        /// <summary>
        /// Gets a composite key for a model name and version.
        /// </summary>
        private static string GetModelKey(string modelName, string version)
        {
            return $"{modelName}:{version}";
        }

        /// <summary>
        /// Internal class to track reference information for a specific model version.
        /// </summary>
        private class ModelReferenceInfo
        {
            public string ModelName { get; }
            public string Version { get; }
            public int Count;  // Changed to field for Interlocked operations

            private readonly ConcurrentDictionary<string, byte> _activeRequests;

            public ModelReferenceInfo(string modelName, string version)
            {
                ModelName = modelName;
                Version = version;
                Count = 0;
                _activeRequests = new ConcurrentDictionary<string, byte>();
            }

            public void AddRequest(string requestId)
            {
                _activeRequests.TryAdd(requestId, 0);
            }

            public bool RemoveRequest(string requestId)
            {
                return _activeRequests.TryRemove(requestId, out _);
            }

            public HashSet<string> GetActiveRequests()
            {
                return new HashSet<string>(_activeRequests.Keys);
            }
        }
    }
}
