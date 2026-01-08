using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace MLFramework.Serving
{
    /// <summary>
    /// Interface for tracking active inference requests per model version using reference counting.
    /// Ensures models are not unloaded while requests are still processing.
    /// </summary>
    public interface IReferenceTracker
    {
        /// <summary>
        /// Acquires a reference for a model version, incrementing the reference count.
        /// </summary>
        /// <param name="modelName">The name of the model.</param>
        /// <param name="version">The version of the model.</param>
        /// <param name="requestId">The unique identifier for the request.</param>
        /// <exception cref="ArgumentException">Thrown if any parameter is null or empty.</exception>
        void AcquireReference(string modelName, string version, string requestId);

        /// <summary>
        /// Releases a reference for a model version, decrementing the reference count.
        /// </summary>
        /// <param name="modelName">The name of the model.</param>
        /// <param name="version">The version of the model.</param>
        /// <param name="requestId">The unique identifier for the request.</param>
        /// <exception cref="ArgumentException">Thrown if any parameter is null or empty.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the reference was never acquired.</exception>
        void ReleaseReference(string modelName, string version, string requestId);

        /// <summary>
        /// Gets the current reference count for a specific model version.
        /// </summary>
        /// <param name="modelName">The name of the model.</param>
        /// <param name="version">The version of the model.</param>
        /// <returns>The current reference count, or 0 if the model is not being tracked.</returns>
        int GetReferenceCount(string modelName, string version);

        /// <summary>
        /// Checks if a model version currently has any active references.
        /// </summary>
        /// <param name="modelName">The name of the model.</param>
        /// <param name="version">The version of the model.</param>
        /// <returns>True if the model version has active references, false otherwise.</returns>
        bool HasReferences(string modelName, string version);

        /// <summary>
        /// Waits asynchronously for all references to be released for a specific model version.
        /// </summary>
        /// <param name="modelName">The name of the model.</param>
        /// <param name="version">The version of the model.</param>
        /// <param name="timeout">The maximum time to wait for references to reach zero.</param>
        /// <param name="ct">Optional cancellation token to cancel the wait operation.</param>
        /// <returns>A task that completes when all references are released or the timeout expires.</returns>
        Task WaitForZeroReferencesAsync(string modelName, string version, TimeSpan timeout, CancellationToken ct = default);

        /// <summary>
        /// Gets all current reference counts for all tracked models.
        /// </summary>
        /// <returns>A dictionary mapping composite model keys to reference counts.</returns>
        Dictionary<string, int> GetAllReferenceCounts();
    }
}
