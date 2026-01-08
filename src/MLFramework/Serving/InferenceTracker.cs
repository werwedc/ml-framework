using System;
using System.Collections.Generic;

namespace MLFramework.Serving
{
    /// <summary>
    /// Implementation of IInferenceTracker that tracks individual inference metrics
    /// and records them to an experiment tracker upon disposal.
    /// </summary>
    public class InferenceTracker : IInferenceTracker
    {
        private readonly IExperimentTracker _experimentTracker;
        private readonly string _experimentId;
        private readonly string _version;
        private readonly DateTime _startTime;
        private readonly Dictionary<string, double> _customMetrics;
        private bool _disposed;
        private bool _hasRecorded;

        /// <summary>
        /// Initializes a new instance of the InferenceTracker.
        /// </summary>
        /// <param name="experimentTracker">The experiment tracker to record metrics to.</param>
        /// <param name="experimentId">The experiment ID.</param>
        /// <param name="version">The model version being tracked.</param>
        public InferenceTracker(IExperimentTracker experimentTracker, string experimentId, string version)
        {
            _experimentTracker = experimentTracker ?? throw new ArgumentNullException(nameof(experimentTracker));
            _experimentId = experimentId ?? throw new ArgumentNullException(nameof(experimentId));
            _version = version ?? throw new ArgumentNullException(nameof(version));
            _startTime = DateTime.UtcNow;
            _customMetrics = new Dictionary<string, double>();
            _disposed = false;
            _hasRecorded = false;
        }

        /// <inheritdoc/>
        public void RecordSuccess(double latencyMs)
        {
            if (latencyMs < 0)
                throw new ArgumentException("Latency must be non-negative.", nameof(latencyMs));

            if (_disposed)
                throw new ObjectDisposedException(nameof(InferenceTracker));

            if (_hasRecorded)
                throw new InvalidOperationException("Inference has already been recorded.");

            _experimentTracker.RecordInference(_experimentId, _version, latencyMs, true, _customMetrics);
            _hasRecorded = true;
        }

        /// <inheritdoc/>
        public void RecordError(double latencyMs, string errorType)
        {
            if (latencyMs < 0)
                throw new ArgumentException("Latency must be non-negative.", nameof(latencyMs));

            if (string.IsNullOrWhiteSpace(errorType))
                throw new ArgumentException("Error type cannot be null or empty.", nameof(errorType));

            if (_disposed)
                throw new ObjectDisposedException(nameof(InferenceTracker));

            if (_hasRecorded)
                throw new InvalidOperationException("Inference has already been recorded.");

            // Add error type as a custom metric
            _customMetrics[$"error_type_{errorType}"] = 1.0;

            _experimentTracker.RecordInference(_experimentId, _version, latencyMs, false, _customMetrics);
            _hasRecorded = true;
        }

        /// <inheritdoc/>
        public void AddCustomMetric(string name, double value)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentException("Custom metric name cannot be null or empty.", nameof(name));

            if (_disposed)
                throw new ObjectDisposedException(nameof(InferenceTracker));

            if (_hasRecorded)
                throw new InvalidOperationException("Cannot add custom metrics after recording inference.");

            _customMetrics[name] = value;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Protected implementation of Dispose pattern.
        /// </summary>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // If inference wasn't recorded, record it as an error
                    if (!_hasRecorded)
                    {
                        try
                        {
                            var latency = (DateTime.UtcNow - _startTime).TotalMilliseconds;
                            RecordError(latency, "disposal_timeout");
                        }
                        catch
                        {
                            // Ignore errors during disposal
                        }
                    }
                }

                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer for InferenceTracker.
        /// </summary>
        ~InferenceTracker()
        {
            Dispose(false);
        }
    }
}
