using System;
using System.Collections.Generic;

namespace MLFramework.Visualization.GPU.Vendor
{
    /// <summary>
    /// Generic GPU tracker for unsupported GPU types
    /// Provides stub implementations with mock data for testing purposes
    /// </summary>
    public class GenericGpuTracker : IDisposable
    {
        private readonly Random _random = new Random();
        private readonly int _deviceId;
        private readonly string _deviceName;
        private bool _disposed = false;

        public GenericGpuTracker(int deviceId, string deviceName)
        {
            _deviceId = deviceId;
            _deviceName = deviceName ?? "Generic GPU";
        }

        /// <summary>
        /// Gets GPU utilization percentage (mock implementation)
        /// </summary>
        public virtual float GetUtilizationPercent()
        {
            return (float)(_random.NextDouble() * 100);
        }

        /// <summary>
        /// Gets used memory bytes (mock implementation)
        /// </summary>
        public virtual long GetUsedMemoryBytes()
        {
            long totalMemory = GetTotalMemoryBytes();
            return (long)(_random.NextDouble() * totalMemory);
        }

        /// <summary>
        /// Gets free memory bytes (mock implementation)
        /// </summary>
        public virtual long GetFreeMemoryBytes()
        {
            long totalMemory = GetTotalMemoryBytes();
            return totalMemory - GetUsedMemoryBytes();
        }

        /// <summary>
        /// Gets total memory bytes (mock implementation)
        /// </summary>
        public virtual long GetTotalMemoryBytes()
        {
            // Default to 8 GB
            return 8L * 1024 * 1024 * 1024;
        }

        /// <summary>
        /// Gets temperature in Celsius (mock implementation)
        /// </summary>
        public virtual float GetTemperatureCelsius()
        {
            return 30f + (float)(_random.NextDouble() * 40);
        }

        /// <summary>
        /// Gets power usage in watts (mock implementation)
        /// </summary>
        public virtual float GetPowerUsageWatts()
        {
            return 50f + (float)(_random.NextDouble() * 100);
        }

        /// <summary>
        /// Gets fan speed in RPM (mock implementation)
        /// </summary>
        public virtual long GetFanSpeedRPM()
        {
            return 1000L + (long)(_random.NextDouble() * 3000);
        }

        /// <summary>
        /// Gets device information
        /// </summary>
        public virtual GPUDeviceInfo GetDeviceInfo()
        {
            return new GPUDeviceInfo(
                _deviceId,
                _deviceName,
                GetTotalMemoryBytes(),
                0, // Compute capability not available for generic GPU
                0,
                true);
        }

        /// <summary>
        /// Checks if this tracker is available
        /// </summary>
        public virtual bool IsAvailable()
        {
            return true;
        }

        #region IDisposable

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Dispose managed resources
                }

                _disposed = true;
            }
        }

        #endregion
    }
}
