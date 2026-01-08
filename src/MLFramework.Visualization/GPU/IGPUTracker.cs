using System;
using System.Collections.Generic;

namespace MLFramework.Visualization.GPU
{
    /// <summary>
    /// Interface for tracking GPU utilization and metrics
    /// </summary>
    public interface IGPUTracker
    {
        // Device information
        /// <summary>
        /// Gets information about all available GPU devices
        /// </summary>
        List<GPUDeviceInfo> GetAvailableDevices();

        /// <summary>
        /// Gets information about a specific GPU device
        /// </summary>
        GPUDeviceInfo GetDeviceInfo(int deviceId);

        // Sampling
        /// <summary>
        /// Starts tracking GPU utilization
        /// </summary>
        /// <param name="deviceId">Device ID to track, or -1 to track all devices</param>
        void StartTracking(int deviceId = -1);

        /// <summary>
        /// Stops tracking GPU utilization
        /// </summary>
        void StopTracking();

        /// <summary>
        /// Takes an immediate sample of GPU utilization
        /// </summary>
        void SampleUtilization();

        /// <summary>
        /// Indicates whether tracking is currently active
        /// </summary>
        bool IsTracking { get; }

        // Statistics
        /// <summary>
        /// Gets statistics for a specific device
        /// </summary>
        GPUStatistics GetStatistics(int deviceId);

        /// <summary>
        /// Gets statistics for all tracked devices
        /// </summary>
        Dictionary<int, GPUStatistics> GetAllStatistics();

        // Timeline
        /// <summary>
        /// Gets samples for a specific device within a time range
        /// </summary>
        IEnumerable<GPUUtilizationSample> GetSamples(int deviceId, DateTime start, DateTime end);

        // Configuration
        /// <summary>
        /// Sets the sampling interval for automatic tracking
        /// </summary>
        void SetSamplingInterval(TimeSpan interval);

        /// <summary>
        /// Enables the GPU tracker
        /// </summary>
        void Enable();

        /// <summary>
        /// Disables the GPU tracker
        /// </summary>
        void Disable();

        /// <summary>
        /// Indicates whether the GPU tracker is enabled
        /// </summary>
        bool IsEnabled { get; }
    }
}
