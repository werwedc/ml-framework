using System;

namespace MLFramework.Serving
{
    /// <summary>
    /// Helper class for automatic reference management using the IDisposable pattern.
    /// Acquires a reference on construction and automatically releases it on disposal.
    /// </summary>
    public sealed class RequestTracker : IDisposable
    {
        private readonly IReferenceTracker _tracker;
        private readonly string _modelName;
        private readonly string _version;
        private readonly string _requestId;
        private bool _disposed;

        /// <summary>
        /// Gets whether the reference has been released.
        /// </summary>
        public bool IsReleased { get; private set; }

        /// <summary>
        /// Initializes a new instance of the RequestTracker class and acquires the reference.
        /// </summary>
        /// <param name="tracker">The reference tracker to use.</param>
        /// <param name="modelName">The name of the model.</param>
        /// <param name="version">The version of the model.</param>
        /// <param name="requestId">The unique identifier for the request.</param>
        /// <exception cref="ArgumentNullException">Thrown if tracker is null.</exception>
        /// <exception cref="ArgumentException">Thrown if any parameter is null or empty.</exception>
        public RequestTracker(IReferenceTracker tracker, string modelName, string version, string requestId)
        {
            _tracker = tracker ?? throw new ArgumentNullException(nameof(tracker));
            _modelName = modelName ?? throw new ArgumentNullException(nameof(modelName));
            _version = version ?? throw new ArgumentNullException(nameof(version));
            _requestId = requestId ?? throw new ArgumentNullException(nameof(requestId));

            _disposed = false;
            IsReleased = false;

            // Acquire the reference
            _tracker.AcquireReference(_modelName, _version, _requestId);
        }

        /// <summary>
        /// Releases the reference. This method is also called automatically when disposed.
        /// </summary>
        public void Release()
        {
            if (_disposed || IsReleased)
                return;

            try
            {
                _tracker.ReleaseReference(_modelName, _version, _requestId);
                IsReleased = true;
            }
            catch (InvalidOperationException ex)
            {
                // Log warning but don't throw - reference may have already been released
                // or never acquired properly
                throw new InvalidOperationException(
                    $"Failed to release reference for model '{_modelName}' version '{_version}' request '{_requestId}': {ex.Message}",
                    ex);
            }
        }

        /// <summary>
        /// Releases the reference and disposes of the tracker.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
                return;

            Release();
            _disposed = true;
        }
    }
}
