namespace MLFramework.Functional.Distributed
{
    /// <summary>
    /// Represents a computation device (CPU, GPU, TPU, etc.).
    /// </summary>
    public class Device
    {
        public int Id { get; }
        public DeviceType Type { get; }
        public string Name { get; }
        public bool IsAvailable { get; }

        public Device(int id, DeviceType type, string name, bool isAvailable = true)
        {
            Id = id;
            Type = type;
            Name = name;
            IsAvailable = isAvailable;
        }

        public override string ToString() => $"Device({Type}:{Id})";

        // Factory methods for common device types
        public static Device CPU(int id) => new Device(id, DeviceType.CPU, $"cpu:{id}");
        public static Device GPU(int id) => new Device(id, DeviceType.GPU, $"gpu:{id}");
    }

    public enum DeviceType
    {
        CPU,
        GPU,
        TPU,
        Other
    }
}
