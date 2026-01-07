using System.Collections.Generic;
using MLFramework.NN;
using MLFramework.Quantization.DataStructures;

namespace MLFramework.Quantization.PTQ
{
    /// <summary>
    /// Interface for traversing model graphs to identify quantizable layers.
    /// </summary>
    public interface IModelTraversal
    {
        /// <summary>
        /// Gets all quantizable layers from the model.
        /// </summary>
        /// <param name="model">The model to traverse.</param>
        /// <returns>List of quantizable layers.</returns>
        List<Module> GetQuantizableLayers(Module model);

        /// <summary>
        /// Gets all layers (including non-quantizable) from the model.
        /// </summary>
        /// <param name="model">The model to traverse.</param>
        /// <returns>List of all layers.</returns>
        List<Module> GetAllLayers(Module model);

        /// <summary>
        /// Gets the name of a layer.
        /// </summary>
        /// <param name="layer">The layer.</param>
        /// <returns>The layer name.</returns>
        string GetLayerName(Module layer);

        /// <summary>
        /// Determines if a layer is quantizable.
        /// </summary>
        /// <param name="layer">The layer to check.</param>
        /// <returns>True if quantizable, false otherwise.</returns>
        bool IsQuantizable(Module layer);

        /// <summary>
        /// Determines if a layer supports per-channel quantization.
        /// </summary>
        /// <param name="layer">The layer to check.</param>
        /// <returns>True if per-channel quantization is supported, false otherwise.</returns>
        bool SupportsPerChannelQuantization(Module layer);
    }
}
