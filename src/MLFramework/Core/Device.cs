namespace MLFramework.Core
{
    /// <summary>
    /// Represents a compute device (GPU/CPU) for tensor operations
    /// </summary>
    public class Device
    {
        /// <summary>
        /// Gets the unique identifier for this device
        /// </summary>
        public DeviceId Id { get; }

        /// <summary>
        /// Gets the device type
        /// </summary>
        public DeviceType Type => Id.Type;

        /// <summary>
        /// Gets the device index
        /// </summary>
        public int Index => Id.Index;

        /// <summary>
        /// Gets the device name
        /// </summary>
        public string Name { get; }

        /// <summary>
        /// Gets the total memory in bytes
        /// </summary>
        public long TotalMemory { get; }

        /// <summary>
        /// Gets the available memory in bytes
        /// </summary>
        public long AvailableMemory { get; }

        /// <summary>
        /// Gets whether tensor cores are supported on this device
        /// </summary>
        public bool SupportsTensorCores { get; }

        /// <summary>
        /// Gets the compute capability (for CUDA devices)
        /// </summary>
        public (int Major, int Minor)? ComputeCapability { get; }

        /// <summary>
        /// Creates a new Device instance
        /// </summary>
        public Device(
            DeviceId id,
            string name,
            long totalMemory,
            long availableMemory,
            bool supportsTensorCores = false,
            (int Major, int Minor)? computeCapability = null)
        {
            Id = id;
            Name = name;
            TotalMemory = totalMemory;
            AvailableMemory = availableMemory;
            SupportsTensorCores = supportsTensorCores;
            ComputeCapability = computeCapability;
        }

        /// <summary>
        /// Creates a CPU device
        /// </summary>
        public static Device CreateCpu(string name = "CPU", bool supportsTensorCores = false)
        {
            return new Device(
                DeviceId.CPU,
                name,
                totalMemory: 16L * 1024 * 1024 * 1024, // 16GB default
                availableMemory: 16L * 1024 * 1024 * 1024,
                supportsTensorCores: supportsTensorCores);
        }

        /// <summary>
        /// Creates a CUDA GPU device
        /// </summary>
        public static Device CreateCuda(
            int index,
            string name,
            long totalMemory,
            long availableMemory,
            bool supportsTensorCores = true,
            (int Major, int Minor)? computeCapability = null)
        {
            return new Device(
                new DeviceId(DeviceType.CUDA, index),
                name,
                totalMemory,
                availableMemory,
                supportsTensorCores,
                computeCapability);
        }

        public override bool Equals(object? obj)
        {
            return obj is Device device && Id.Equals(device.Id);
        }

        public override int GetHashCode()
        {
            return Id.GetHashCode();
        }

        public override string ToString()
        {
            return $"{Name} ({Id})";
        }
    }
}
