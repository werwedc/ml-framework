using System;
using System.Collections.Generic;
using System.Linq;
using MLFramework.NN;

namespace MLFramework.Quantization.PTQ
{
    /// <summary>
    /// Model traversal implementation for identifying quantizable layers.
    /// </summary>
    public class ModelTraversal : IModelTraversal
    {
        // Non-quantizable layer types will be determined dynamically
        // These are common patterns in layer names
        private static readonly HashSet<string> NonQuantizableLayerPatterns = new HashSet<string>
        {
            "relu",
            "sigmoid",
            "tanh",
            "softmax",
            "dropout",
            "pool",
            "batchnorm",
            "layernorm",
            "flatten",
            "identity"
        };

        /// <summary>
        /// Gets all quantizable layers from the model.
        /// </summary>
        /// <param name="model">The model to traverse.</param>
        /// <returns>List of quantizable layers.</returns>
        public List<Module> GetQuantizableLayers(Module model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            var allLayers = GetAllLayers(model);
            return allLayers.Where(IsQuantizable).ToList();
        }

        /// <summary>
        /// Gets all layers (including non-quantizable) from the model.
        /// </summary>
        /// <param name="model">The model to traverse.</param>
        /// <returns>List of all layers.</returns>
        public List<Module> GetAllLayers(Module model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            var layers = new List<Module>();
            Traverse(model, layers);
            return layers;
        }

        /// <summary>
        /// Gets the name of a layer.
        /// </summary>
        /// <param name="layer">The layer.</param>
        /// <returns>The layer name.</returns>
        public string GetLayerName(Module layer)
        {
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));

            return layer.Name;
        }

        /// <summary>
        /// Determines if a layer is quantizable.
        /// </summary>
        /// <param name="layer">The layer to check.</param>
        /// <returns>True if quantizable, false otherwise.</returns>
        public bool IsQuantizable(Module layer)
        {
            if (layer == null)
                return false;

            var layerName = layer.Name.ToLowerInvariant();
            var hasParameters = layer.GetParameters().Any();

            // Skip non-quantizable layers
            foreach (var pattern in NonQuantizableLayerPatterns)
            {
                if (layerName.Contains(pattern))
                    return false;
            }

            return hasParameters;
        }

        /// <summary>
        /// Determines if a layer supports per-channel quantization.
        /// </summary>
        /// <param name="layer">The layer to check.</param>
        /// <returns>True if per-channel quantization is supported, false otherwise.</returns>
        public bool SupportsPerChannelQuantization(Module layer)
        {
            if (layer == null)
                return false;

            // Conv2d layers support per-channel quantization
            // Linear layers typically use per-tensor quantization
            var layerName = layer.Name.ToLowerInvariant();
            return layerName.Contains("conv");
        }

        /// <summary>
        /// Recursively traverses the model to collect all layers.
        /// </summary>
        private void Traverse(Module module, List<Module> layers)
        {
            if (module == null)
                return;

            // Add current module to the list
            layers.Add(module);

            // Traverse child modules if this is a container module
            // Note: This is a simplified traversal - real implementation would need
            // to properly handle HierarchicalModule and other container types
            var parameters = module.GetParameters();
            if (parameters.Count() == 0)
            {
                // This might be a container module - try to get child modules
                // In a full implementation, we would have proper child module access
            }
        }
    }
}
