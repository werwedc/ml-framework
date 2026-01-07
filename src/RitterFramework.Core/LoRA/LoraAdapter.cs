namespace RitterFramework.Core.LoRA
{
    /// <summary>
    /// Weights for a LoRA module (low-rank matrices A and B).
    /// </summary>
    public class LoraModuleWeights
    {
        /// <summary>LoRA matrix A (down-projection)</summary>
        public Core.Tensor.Tensor LoraA { get; set; }

        /// <summary>LoRA matrix B (up-projection)</summary>
        public Core.Tensor.Tensor LoraB { get; set; }

        /// <summary>
        /// Constructor.
        /// </summary>
        public LoraModuleWeights(Core.Tensor.Tensor loraA, Core.Tensor.Tensor loraB)
        {
            LoraA = loraA ?? throw new ArgumentNullException(nameof(loraA));
            LoraB = loraB ?? throw new ArgumentNullException(nameof(loraB));
        }

        /// <summary>
        /// Create a deep copy of these weights.
        /// </summary>
        public LoraModuleWeights Clone()
        {
            return new LoraModuleWeights(LoraA.Clone(), LoraB.Clone());
        }
    }

    /// <summary>
    /// Metadata for a LoRA adapter.
    /// </summary>
    public class LoraAdapterMetadata
    {
        /// <summary>Creation timestamp</summary>
        public DateTime CreatedAt { get; set; }

        /// <summary>Creator/user information</summary>
        public string Creator { get; set; }

        /// <summary>Description of the adapter</summary>
        public string Description { get; set; }

        /// <summary>Version of the adapter format</summary>
        public string Version { get; set; }

        public LoraAdapterMetadata()
        {
            CreatedAt = DateTime.UtcNow;
            Creator = "unknown";
            Description = "";
            Version = "1.0";
        }
    }

    /// <summary>
    /// Represents a LoRA adapter that can be loaded and saved.
    /// </summary>
    public class LoraAdapter
    {
        /// <summary>Name of the adapter</summary>
        public string Name { get; set; }

        /// <summary>LoRA configuration</summary>
        public LoraConfig Config { get; set; }

        /// <summary>Module weights (key: module name, value: LoRA matrices)</summary>
        public Dictionary<string, LoraModuleWeights> Weights { get; set; }

        /// <summary>Adapter metadata</summary>
        public LoraAdapterMetadata Metadata { get; set; }

        public LoraAdapter()
        {
            Weights = new Dictionary<string, LoraModuleWeights>();
            Metadata = new LoraAdapterMetadata();
        }

        public LoraAdapter(string name, LoraConfig config) : this()
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Config = config ?? throw new ArgumentNullException(nameof(config));
        }

        /// <summary>
        /// Try to get weights for a specific module.
        /// </summary>
        public bool TryGetModuleWeights(string moduleName, out LoraModuleWeights weights)
        {
            return Weights.TryGetValue(moduleName, out weights);
        }

        /// <summary>
        /// Add weights for a module.
        /// </summary>
        public void AddModuleWeights(string moduleName, LoraModuleWeights weights)
        {
            Weights[moduleName] = weights;
        }

        /// <summary>
        /// Create a deep copy of this adapter.
        /// </summary>
        public LoraAdapter Clone()
        {
            var cloned = new LoraAdapter(Name, Config.Clone())
            {
                Metadata = new LoraAdapterMetadata
                {
                    CreatedAt = Metadata.CreatedAt,
                    Creator = Metadata.Creator,
                    Description = Metadata.Description,
                    Version = Metadata.Version
                }
            };

            foreach (var kvp in Weights)
            {
                cloned.Weights[kvp.Key] = kvp.Value.Clone();
            }

            return cloned;
        }
    }
}
