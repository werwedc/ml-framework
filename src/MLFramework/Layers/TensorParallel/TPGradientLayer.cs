using MLFramework.NN;
using RitterFramework.Core.Tensor;
using System.Collections.Generic;

namespace MLFramework.Layers.TensorParallel
{
    /// <summary>
    /// Abstract base class for tensor parallel layers with gradient support.
    /// Provides common functionality for TP layer forward and backward passes.
    /// </summary>
    public abstract class TPGradientLayer : Module
    {
        protected readonly TensorParallelGroup? _processGroup;
        protected readonly int _worldSize;
        protected readonly int _rank;

        /// <summary>
        /// Creates a new TPGradientLayer.
        /// </summary>
        /// <param name="processGroup">Optional process group for TP operations. If null, uses default.</param>
        protected TPGradientLayer(TensorParallelGroup? processGroup = null)
            : base("TPGradientLayer")
        {
            if (processGroup != null)
            {
                _processGroup = processGroup;
                _worldSize = processGroup.WorldSize;
                _rank = processGroup.Rank;
            }
            else if (TensorParallel.IsInitialized())
            {
                _worldSize = TensorParallel.GetWorldSize();
                _rank = TensorParallel.GetRank();
            }
            else
            {
                // Default to single-process mode
                _worldSize = 1;
                _rank = 0;
            }
        }

        /// <summary>
        /// Internal forward pass implementation.
        /// </summary>
        protected abstract Tensor ForwardInternal(Tensor input);

        /// <summary>
        /// Internal backward pass implementation.
        /// </summary>
        protected abstract Tensor BackwardInternal(Tensor gradOutput);

        /// <summary>
        /// Forward pass of the layer.
        /// </summary>
        public override Tensor Forward(Tensor input)
        {
            return ForwardInternal(input);
        }

        /// <summary>
        /// Gets all parameters of this layer.
        /// </summary>
        public override IEnumerable<Parameter> GetParameters()
        {
            return GetTrainableParameters();
        }

        /// <summary>
        /// Gets all named parameters of this layer.
        /// </summary>
        public override IEnumerable<(string Name, Parameter Parameter)> GetNamedParameters()
        {
            foreach (var param in GetTrainableParameters())
            {
                yield return (param.Name, param);
            }
        }

        /// <summary>
        /// Gets trainable parameters. Subclasses should override this.
        /// </summary>
        protected abstract IEnumerable<Parameter> GetTrainableParameters();
    }
}
