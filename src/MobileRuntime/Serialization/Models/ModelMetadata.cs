namespace MobileRuntime.Serialization.Models
{
    /// <summary>
    /// Metadata about the model
    /// </summary>
    public class ModelMetadata
    {
        public string Name { get; set; }
        public uint FrameworkVersion { get; set; }
        public ulong CreationTimestamp { get; set; }
        public uint InputCount { get; set; }
        public uint OutputCount { get; set; }
    }
}
