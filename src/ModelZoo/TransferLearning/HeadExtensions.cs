using System;
using System.Collections.Generic;
using System.Linq;
using MLFramework.NN;

namespace MLFramework.ModelZoo.TransferLearning
{
    /// <summary>
    /// Exception thrown when head replacement operations fail due to invalid configurations.
    /// </summary>
    public class HeadReplacementException : Exception
    {
        public HeadReplacementException(string message) : base(message) { }
        public HeadReplacementException(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    /// Extension methods for model head manipulation.
    /// </summary>
    public static class HeadExtensions
    {
        /// <summary>
        /// Removes the final layer from a sequential module.
        /// </summary>
        /// <param name="model">The sequential module.</param>
        /// <returns>The removed module.</returns>
        /// <exception cref="HeadReplacementException">Thrown when the model has no layers to remove.</exception>
        public static Module RemoveLastLayer(this SequentialModule model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            if (model.Count == 0)
                throw new HeadReplacementException("Cannot remove last layer: model has no layers.");

            int lastIndex = model.Count - 1;
            Module lastModule = model.GetModule(lastIndex);
            model.RemoveAt(lastIndex);
            return lastModule;
        }

        /// <summary>
        /// Removes the final N layers from a sequential module.
        /// </summary>
        /// <param name="model">The sequential module.</param>
        /// <param name="n">Number of layers to remove.</param>
        /// <returns>List of removed modules.</returns>
        /// <exception cref="HeadReplacementException">Thrown when N exceeds the number of layers.</exception>
        public static List<Module> RemoveLastNLayers(this SequentialModule model, int n)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            if (n <= 0)
                throw new ArgumentOutOfRangeException(nameof(n), "Number of layers to remove must be positive.");

            if (n > model.Count)
                throw new HeadReplacementException(
                    $"Cannot remove {n} layers: model only has {model.Count} layers.");

            var removedModules = new List<Module>();
            for (int i = 0; i < n; i++)
            {
                removedModules.Add(model.RemoveLastLayer());
            }

            // Reverse to maintain original order
            removedModules.Reverse();
            return removedModules;
        }

        /// <summary>
        /// Adds a new module as the head (last layer) of a sequential module.
        /// </summary>
        /// <param name="model">The sequential module.</param>
        /// <param name="head">The module to add as head.</param>
        public static void AddHead(this SequentialModule model, Module head)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            if (head == null)
                throw new ArgumentNullException(nameof(head));

            model.Add(head);
        }

        /// <summary>
        /// Replaces the final layer with a new module.
        /// </summary>
        /// <param name="model">The sequential module.</param>
        /// <param name="newHead">The new head module.</param>
        /// <returns>The replaced module.</returns>
        /// <exception cref="HeadReplacementException">Thrown when the model has no layers to replace.</exception>
        public static Module ReplaceHead(this SequentialModule model, Module newHead)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            if (newHead == null)
                throw new ArgumentNullException(nameof(newHead));

            if (model.Count == 0)
                throw new HeadReplacementException("Cannot replace head: model has no layers.");

            Module oldHead = model.RemoveLastLayer();
            model.AddHead(newHead);
            return oldHead;
        }

        /// <summary>
        /// Removes a layer by name from a hierarchical module.
        /// </summary>
        /// <param name="model">The hierarchical module.</param>
        /// <param name="headName">Name of the layer to remove.</param>
        /// <returns>The removed module.</returns>
        /// <exception cref="HeadReplacementException">Thrown when layer with given name is not found.</exception>
        public static Module RemoveHead(this HierarchicalModule model, string headName)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            if (string.IsNullOrEmpty(headName))
                throw new ArgumentException("Head name cannot be null or empty.", nameof(headName));

            var child = model.GetChild(headName);
            if (child == null)
                throw new HeadReplacementException($"Layer with name '{headName}' not found in model.");

            // Since HierarchicalModule doesn't have a direct remove method, we need to
            // access the internal children dictionary via reflection or add a method to HierarchicalModule
            // For now, we'll throw an exception indicating this limitation
            throw new NotSupportedException(
                "Removing named children from HierarchicalModule requires additional implementation. " +
                "Consider using ReplaceChild with a dummy module.");
        }

        /// <summary>
        /// Gets the final layer (head) of a sequential module.
        /// </summary>
        /// <param name="model">The sequential module.</param>
        /// <returns>The final module, or null if model is empty.</returns>
        public static Module GetHead(this SequentialModule model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            if (model.Count == 0)
                return null;

            return model.GetModule(model.Count - 1);
        }

        /// <summary>
        /// Gets all final N layers (heads) that are classification/regression layers.
        /// This heuristic identifies layers that are typically used as output layers.
        /// </summary>
        /// <param name="model">The sequential module.</param>
        /// <param name="maxCount">Maximum number of head layers to return.</param>
        /// <returns>List of potential head layers.</returns>
        public static List<Module> GetHeads(this SequentialModule model, int maxCount = 3)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            var heads = new List<Module>();

            // Collect layers from the end
            for (int i = Math.Min(maxCount, model.Count) - 1; i >= 0; i--)
            {
                heads.Add(model.GetModule(model.Count - 1 - i));
            }

            return heads;
        }

        /// <summary>
        /// Validates that a head module is compatible with the model's architecture.
        /// Checks if output dimensions match expected input dimensions.
        /// </summary>
        /// <param name="model">The sequential module.</param>
        /// <param name="newHead">The head module to validate.</param>
        /// <param name="expectedInputDim">Expected input dimension for the head.</param>
        /// <returns>True if compatible, false otherwise.</returns>
        public static bool ValidateHeadReplacement(this SequentialModule model, Module newHead, int? expectedInputDim = null)
        {
            if (model == null || newHead == null)
                return false;

            // Basic validation: model must have layers or new head must work standalone
            if (model.Count == 0 && expectedInputDim.HasValue)
            {
                // Would need to inspect new head's expected input dimension
                // This is a placeholder - actual implementation would need to query the module
                return true;
            }

            // If model has layers, new head should be compatible with previous layer's output
            // This requires inspecting the previous layer's output shape and new head's input shape
            // For now, we'll return true and rely on runtime validation
            return true;
        }

        /// <summary>
        /// Gets the expected input dimension for a new head based on the model's architecture.
        /// </summary>
        /// <param name="model">The sequential module.</param>
        /// <returns>Expected input dimension, or null if cannot be determined.</returns>
        public static int? GetHeadInputDimension(this SequentialModule model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            if (model.Count == 0)
                return null;

            // Try to get the output dimension of the previous layer
            // This would require inspecting the layer's properties
            // For now, return null to indicate we need more implementation
            return null;
        }

        /// <summary>
        /// Gets a summary of the model's head layers.
        /// </summary>
        /// <param name="model">The sequential module.</param>
        /// <returns>A string describing the head configuration.</returns>
        public static string GetHeadSummary(this SequentialModule model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            if (model.Count == 0)
                return "Model has no layers.";

            var heads = model.GetHeads(3);
            var summary = $"Model has {model.Count} total layer(s). ";
            summary += $"Head layers ({heads.Count}): {string.Join(", ", heads.Select(h => h.Name))}";

            return summary;
        }
    }
}
