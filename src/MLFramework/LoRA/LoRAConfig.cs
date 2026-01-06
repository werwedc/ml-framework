using RitterFramework.Core.Tensor;

namespace MLFramework.LoRA
{
    /// <summary>
    /// Configuration for LoRA (Low-Rank Adaptation) adapters
    /// </summary>
    public class LoRAConfig
    {
        /// <summary>
        /// Rank of the low-rank decomposition (typically 4-64)
        /// </summary>
        public int Rank { get; set; } = 8;

        /// <summary>
        /// Scaling factor alpha (controls adapter influence)
        /// </summary>
        public float Alpha { get; set; } = 16.0f;

        /// <summary>
        /// Target modules to apply LoRA to (e.g., ["attn.q_proj", "attn.v_proj"])
        /// If null/empty, applies to all compatible layers
        /// </summary>
        public string[]? TargetModules { get; set; } = null;

        /// <summary>
        /// Whether to apply LoRA to bias terms (default: false)
        /// </summary>
        public bool UseBias { get; set; } = false;

        /// <summary>
        /// Initialization strategy for adapter weights
        /// </summary>
        public LoRAInitializationStrategy Initialization { get; set; } = LoRAInitializationStrategy.Standard;

        private float _dropout = 0.0f;

        /// <summary>
        /// Dropout rate for LoRA layers (0.0 = no dropout)
        /// </summary>
        public float Dropout
        {
            get => _dropout;
            set
            {
                if (value < 0 || value >= 1)
                    throw new ArgumentException("Dropout must be in [0, 1)", nameof(Dropout));
                _dropout = value;
            }
        }

        /// <summary>
        /// Whether to use fused kernels if available (performance optimization)
        /// </summary>
        public bool UseFusedKernels { get; set; } = false;

        /// <summary>
        /// Target parameter types (Linear, Conv2d, Embedding, etc.)
        /// If null/empty, applies to all types
        /// </summary>
        public string[]? TargetLayerTypes { get; set; } = null;

        /// <summary>
        /// Creates a new LoRA configuration
        /// </summary>
        /// <param name="rank">Rank of the low-rank decomposition</param>
        /// <param name="alpha">Scaling factor alpha</param>
        public LoRAConfig(int rank = 8, float alpha = 16.0f)
        {
            Rank = rank;
            Alpha = alpha;
            Validate();
        }

        /// <summary>
        /// Validates the configuration parameters
        /// </summary>
        private void Validate()
        {
            if (Rank <= 0)
                throw new ArgumentException("Rank must be positive", nameof(Rank));
            if (Alpha <= 0)
                throw new ArgumentException("Alpha must be positive", nameof(Alpha));
            if (_dropout < 0 || _dropout >= 1)
                throw new ArgumentException("Dropout must be in [0, 1)", nameof(Dropout));
        }
    }
}
