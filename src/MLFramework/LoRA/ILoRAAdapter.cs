using RitterFramework.Core.Tensor;
using MLFramework.Modules;

namespace MLFramework.LoRA
{
    /// <summary>
    /// Core interface that all LoRA wrapper layers must implement
    /// </summary>
    public interface ILoRAAdapter
    {
        /// <summary>
        /// Gets the base (wrapped) layer
        /// </summary>
        IModule BaseLayer { get; }

        /// <summary>
        /// Gets the LoRA rank
        /// </summary>
        int Rank { get; }

        /// <summary>
        /// Gets the LoRA scaling factor (alpha / rank)
        /// </summary>
        float ScalingFactor { get; }

        /// <summary>
        /// Freezes the base layer (only adapter parameters remain trainable)
        /// </summary>
        void FreezeBaseLayer();

        /// <summary>
        /// Unfreezes the base layer
        /// </summary>
        void UnfreezeBaseLayer();

        /// <summary>
        /// Gets all trainable parameters (adapter weights only if base is frozen)
        /// </summary>
        IEnumerable<Tensor> TrainableParameters { get; }

        /// <summary>
        /// Gets all frozen parameters
        /// </summary>
        IEnumerable<Tensor> FrozenParameters { get; }

        /// <summary>
        /// Enables or disables the LoRA adapter
        /// </summary>
        bool IsEnabled { get; set; }

        /// <summary>
        /// Merges adapter weights into the base layer (for deployment)
        /// </summary>
        void MergeAdapter();

        /// <summary>
        /// Resets the base layer to original weights (undoes merge)
        /// </summary>
        void ResetBaseLayer();

        /// <summary>
        /// Gets the adapter weights as tensors
        /// </summary>
        (Tensor? MatrixA, Tensor? MatrixB) GetAdapterWeights();

        /// <summary>
        /// Sets the adapter weights from tensors
        /// </summary>
        void SetAdapterWeights(Tensor? matrixA, Tensor? matrixB);
    }
}
