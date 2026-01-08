using System;

namespace MLFramework.Visualization.GPU
{
    /// <summary>
    /// Represents information about a GPU device
    /// </summary>
    public class GPUDeviceInfo
    {
        public int DeviceId { get; }
        public string Name { get; }
        public long TotalMemoryBytes { get; }
        public int ComputeCapabilityMajor { get; }
        public int ComputeCapabilityMinor { get; }
        public bool IsAvailable { get; }

        public GPUDeviceInfo(
            int deviceId,
            string name,
            long totalMemoryBytes,
            int computeCapabilityMajor,
            int computeCapabilityMinor,
            bool isAvailable)
        {
            if (deviceId < 0)
                throw new ArgumentException("Device ID must be non-negative", nameof(deviceId));
            
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentException("GPU name cannot be empty", nameof(name));
            
            if (totalMemoryBytes <= 0)
                throw new ArgumentException("Total memory must be positive", nameof(totalMemoryBytes));

            DeviceId = deviceId;
            Name = name;
            TotalMemoryBytes = totalMemoryBytes;
            ComputeCapabilityMajor = computeCapabilityMajor;
            ComputeCapabilityMinor = computeCapabilityMinor;
            IsAvailable = isAvailable;
        }

        /// <summary>
        /// Returns a string representation of the GPU device
        /// </summary>
        public override string ToString()
        {
            return $"GPU {DeviceId}: {Name} ({ComputeCapabilityMajor}.{ComputeCapabilityMinor}) - " +
                   $"{TotalMemoryBytes / (1024 * 1024)} MB - {(IsAvailable ? "Available" : "Unavailable")}";
        }
    }
}
