namespace MLFramework.Core
{
    /// <summary>
    /// Unique identifier for a compute device (GPU/CPU)
    /// </summary>
    public readonly struct DeviceId : IEquatable<DeviceId>
    {
        /// <summary>
        /// Device type (CPU, CUDA, etc.)
        /// </summary>
        public DeviceType Type { get; }

        /// <summary>
        /// Device index (0, 1, 2, ...)
        /// </summary>
        public int Index { get; }

        public DeviceId(DeviceType type, int index = 0)
        {
            Type = type;
            Index = index;
        }

        public static readonly DeviceId CPU = new DeviceId(DeviceType.CPU, 0);

        public bool Equals(DeviceId other)
        {
            return Type == other.Type && Index == other.Index;
        }

        public override bool Equals(object? obj)
        {
            return obj is DeviceId other && Equals(other);
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(Type, Index);
        }

        public static bool operator ==(DeviceId left, DeviceId right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(DeviceId left, DeviceId right)
        {
            return !left.Equals(right);
        }

        public override string ToString()
        {
            return $"{Type}:{Index}";
        }
    }

    /// <summary>
    /// Device type enumeration
    /// </summary>
    public enum DeviceType
    {
        /// <summary>
        /// CPU device
        /// </summary>
        CPU = 0,

        /// <summary>
        /// CUDA GPU device
        /// </summary>
        CUDA = 1,

        /// <summary>
        /// ROCm GPU device
        /// </summary>
        ROCm = 2,

        /// <summary>
        /// Metal GPU device
        /// </summary>
        Metal = 3,

        /// <summary>
        /// OpenCL device
        /// </summary>
        OpenCL = 4
    }
}
