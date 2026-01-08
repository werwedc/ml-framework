using System;

namespace MLFramework.Visualization.GPU
{
    /// <summary>
    /// Represents a single sample of GPU utilization metrics
    /// </summary>
    public class GPUUtilizationSample
    {
        public DateTime Timestamp { get; }
        public int DeviceId { get; }
        public float UtilizationPercent { get; } // 0.0 to 100.0
        public long UsedMemoryBytes { get; }
        public long FreeMemoryBytes { get; }
        public long TotalMemoryBytes { get; }
        public float TemperatureCelsius { get; }
        public float PowerUsageWatts { get; }
        public long FanSpeedRPM { get; }

        public GPUUtilizationSample(
            int deviceId,
            float utilizationPercent,
            long usedMemoryBytes,
            long freeMemoryBytes,
            long totalMemoryBytes,
            float temperatureCelsius = -1f,
            float powerUsageWatts = -1f,
            long fanSpeedRPM = -1,
            DateTime? timestamp = null)
        {
            if (deviceId < 0)
                throw new ArgumentException("Device ID must be non-negative", nameof(deviceId));
            
            if (utilizationPercent < 0 || utilizationPercent > 100)
                throw new ArgumentException("Utilization percent must be between 0 and 100", nameof(utilizationPercent));
            
            if (usedMemoryBytes < 0)
                throw new ArgumentException("Used memory must be non-negative", nameof(usedMemoryBytes));
            
            if (freeMemoryBytes < 0)
                throw new ArgumentException("Free memory must be non-negative", nameof(freeMemoryBytes));
            
            if (totalMemoryBytes <= 0)
                throw new ArgumentException("Total memory must be positive", nameof(totalMemoryBytes));
            
            if (usedMemoryBytes + freeMemoryBytes > totalMemoryBytes)
                throw new ArgumentException("Used + free memory cannot exceed total memory", nameof(usedMemoryBytes));

            Timestamp = timestamp ?? DateTime.UtcNow;
            DeviceId = deviceId;
            UtilizationPercent = utilizationPercent;
            UsedMemoryBytes = usedMemoryBytes;
            FreeMemoryBytes = freeMemoryBytes;
            TotalMemoryBytes = totalMemoryBytes;
            TemperatureCelsius = temperatureCelsius;
            PowerUsageWatts = powerUsageWatts;
            FanSpeedRPM = fanSpeedRPM;
        }

        /// <summary>
        /// Gets memory usage percentage
        /// </summary>
        public float MemoryUsagePercent => (float)UsedMemoryBytes / TotalMemoryBytes * 100f;

        /// <summary>
        /// Checks if temperature data is available
        /// </summary>
        public bool HasTemperature => TemperatureCelsius >= 0;

        /// <summary>
        /// Checks if power data is available
        /// </summary>
        public bool HasPower => PowerUsageWatts >= 0;

        /// <summary>
        /// Checks if fan speed data is available
        /// </summary>
        public bool HasFanSpeed => FanSpeedRPM >= 0;

        /// <summary>
        /// Returns a string representation of the sample
        /// </summary>
        public override string ToString()
        {
            return $"[Device {DeviceId} @ {Timestamp:HH:mm:ss}] " +
                   $"GPU: {UtilizationPercent:F1}% | " +
                   $"Memory: {UsedMemoryBytes / (1024 * 1024)} MB / {TotalMemoryBytes / (1024 * 1024)} MB" +
                   (HasTemperature ? $" | Temp: {TemperatureCelsius:F1}°C" : "") +
                   (HasPower ? $" | Power: {PowerUsageWatts:F1}W" : "") +
                   (HasFanSpeed ? $" | Fan: {FanSpeedRPM} RPM" : "");
        }
    }
}
