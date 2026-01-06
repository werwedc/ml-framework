using RitterFramework.Core.Tensor;
using MLFramework.Modules;

namespace MLFramework.LoRA
{
    /// <summary>
    /// Abstract base class providing common functionality for LoRA adapters
    /// </summary>
    public abstract class LoRAAdapterBase : ILoRAAdapter
    {
    protected readonly IModule _baseLayer;
        protected readonly int _rank;
        protected readonly float _alpha;
        protected bool _isBaseLayerFrozen;
        protected bool _isEnabled = true;
        protected Tensor? _baseLayerWeightsBackup;

        /// <summary>
        /// Gets the LoRA alpha scaling factor
        /// </summary>
        public float Alpha => _alpha;

        /// <summary>
        /// Initializes a new instance of the LoRAAdapterBase
        /// </summary>
        /// <param name="baseLayer">The base layer to wrap</param>
        /// <param name="rank">LoRA rank</param>
        /// <param name="alpha">LoRA alpha scaling factor</param>
        protected LoRAAdapterBase(IModule baseLayer, int rank, float alpha)
        {
            _baseLayer = baseLayer ?? throw new ArgumentNullException(nameof(baseLayer));
            _rank = rank;
            _alpha = alpha;
            _isBaseLayerFrozen = false;
        }

        /// <summary>
        /// Gets the base (wrapped) layer
        /// </summary>
        public IModule BaseLayer => _baseLayer;

        /// <summary>
        /// Gets the LoRA rank
        /// </summary>
        public int Rank => _rank;

        /// <summary>
        /// Gets the LoRA scaling factor (alpha / rank)
        /// </summary>
        public float ScalingFactor => _alpha / _rank;

        /// <summary>
        /// Enables or disables the LoRA adapter
        /// </summary>
        public bool IsEnabled
        {
            get => _isEnabled;
            set => _isEnabled = value;
        }

        /// <summary>
        /// Freezes the base layer (only adapter parameters remain trainable)
        /// </summary>
        public abstract void FreezeBaseLayer();

        /// <summary>
        /// Unfreezes the base layer
        /// </summary>
        public abstract void UnfreezeBaseLayer();

        /// <summary>
        /// Gets all trainable parameters (adapter weights only if base is frozen)
        /// </summary>
        public abstract IEnumerable<Tensor> TrainableParameters { get; }

        /// <summary>
        /// Gets all frozen parameters
        /// </summary>
        public abstract IEnumerable<Tensor> FrozenParameters { get; }

        /// <summary>
        /// Merges adapter weights into the base layer (for deployment)
        /// </summary>
        public abstract void MergeAdapter();

        /// <summary>
        /// Resets the base layer to original weights (undoes merge)
        /// </summary>
        public abstract void ResetBaseLayer();

        /// <summary>
        /// Gets the adapter weights as tensors
        /// </summary>
        public abstract (Tensor? MatrixA, Tensor? MatrixB) GetAdapterWeights();

        /// <summary>
        /// Sets the adapter weights from tensors
        /// </summary>
        public abstract void SetAdapterWeights(Tensor? matrixA, Tensor? matrixB);
    }
}
