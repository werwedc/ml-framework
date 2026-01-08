using System;

namespace MLFramework.Serving
{
    /// <summary>
    /// Interface for tracking individual inference metrics within an experiment.
    /// </summary>
    public interface IInferenceTracker : IDisposable
    {
        /// <summary>
        /// Records a successful inference with its latency.
        /// </summary>
        /// <param name="latencyMs">Latency in milliseconds.</param>
        void RecordSuccess(double latencyMs);

        /// <summary>
        /// Records a failed inference with its latency and error type.
        /// </summary>
        /// <param name="latencyMs">Latency in milliseconds.</param>
        /// <param name="errorType">Type of error that occurred.</param>
        void RecordError(double latencyMs, string errorType);

        /// <summary>
        /// Adds a custom metric value to this inference.
        /// </summary>
        /// <param name="name">Name of the custom metric.</param>
        /// <param name="value">Value of the custom metric.</param>
        void AddCustomMetric(string name, double value);
    }
}
