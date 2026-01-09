using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using MLFramework.NN;

namespace MLFramework.ModelZoo.TransferLearning
{
    /// <summary>
    /// Helper class for selecting layers from a module based on various criteria.
    /// </summary>
    public static class LayerSelectionHelper
    {
        /// <summary>
        /// Selects layers by exact name(s).
        /// </summary>
        /// <param name="module">The root module to search.</param>
        /// <param name="names">Names of layers to select.</param>
        /// <returns>Enumerable of matching layers.</returns>
        public static IEnumerable<Module> SelectByName(Module module, params string[] names)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));
            if (names == null)
                throw new ArgumentNullException(nameof(names));

            var allLayers = module.GetAllModules();
            return allLayers.Where(layer => names.Contains(layer.Name));
        }

        /// <summary>
        /// Selects layers matching a regex pattern.
        /// </summary>
        /// <param name="module">The root module to search.</param>
        /// <param name="pattern">Regular expression pattern to match layer names.</param>
        /// <returns>Enumerable of matching layers.</returns>
        public static IEnumerable<Module> SelectByPattern(Module module, string pattern)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));
            if (string.IsNullOrEmpty(pattern))
                throw new ArgumentException("Pattern cannot be null or empty.", nameof(pattern));

            try
            {
                var regex = new Regex(pattern, RegexOptions.Compiled);
                var allLayers = module.GetAllModules();
                return allLayers.Where(layer => regex.IsMatch(layer.Name));
            }
            catch (ArgumentException ex)
            {
                throw new ArgumentException($"Invalid regex pattern: {pattern}", nameof(pattern), ex);
            }
        }

        /// <summary>
        /// Selects a layer by its index in the module hierarchy.
        /// </summary>
        /// <param name="module">The root module to search.</param>
        /// <param name="index">Index of the layer to select.</param>
        /// <returns>The layer at the specified index.</returns>
        public static Module SelectByIndex(Module module, int index)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));

            var allLayers = module.GetAllModules().ToList();
            if (index < 0 || index >= allLayers.Count)
                throw new ArgumentOutOfRangeException(nameof(index),
                    $"Index {index} is out of range. Module has {allLayers.Count} layers.");

            return allLayers[index];
        }

        /// <summary>
        /// Selects layers within a range of indices.
        /// </summary>
        /// <param name="module">The root module to search.</param>
        /// <param name="startIndex">Starting index (inclusive).</param>
        /// <param name="endIndex">Ending index (inclusive).</param>
        /// <returns>Enumerable of layers in the specified range.</returns>
        public static IEnumerable<Module> SelectByRange(Module module, int startIndex, int endIndex)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));

            if (startIndex < 0)
                throw new ArgumentOutOfRangeException(nameof(startIndex), "Start index cannot be negative.");

            if (endIndex < startIndex)
                throw new ArgumentOutOfRangeException(nameof(endIndex),
                    "End index must be greater than or equal to start index.");

            var allLayers = module.GetAllModules().ToList();
            if (startIndex >= allLayers.Count)
                throw new ArgumentOutOfRangeException(nameof(startIndex),
                    $"Start index {startIndex} is out of range. Module has {allLayers.Count} layers.");

            if (endIndex >= allLayers.Count)
                throw new ArgumentOutOfRangeException(nameof(endIndex),
                    $"End index {endIndex} is out of range. Module has {allLayers.Count} layers.");

            for (int i = startIndex; i <= endIndex; i++)
            {
                yield return allLayers[i];
            }
        }

        /// <summary>
        /// Selects all layers of a specific type.
        /// </summary>
        /// <typeparam name="T">Type of layer to select.</typeparam>
        /// <param name="module">The root module to search.</param>
        /// <returns>Enumerable of layers of the specified type.</returns>
        public static IEnumerable<T> SelectByType<T>(Module module) where T : Module
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));

            return module.GetAllModules().OfType<T>();
        }

        /// <summary>
        /// Selects layers of a specific type.
        /// </summary>
        /// <param name="module">The root module to search.</param>
        /// <param name="layerType">Type of layer to select.</param>
        /// <returns>Enumerable of layers of the specified type.</returns>
        public static IEnumerable<Module> SelectByType(Module module, Type layerType)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));
            if (layerType == null)
                throw new ArgumentNullException(nameof(layerType));

            if (!typeof(Module).IsAssignableFrom(layerType))
                throw new ArgumentException($"Type {layerType.Name} must derive from Module.", nameof(layerType));

            return module.GetAllModules().Where(layer => layerType.IsAssignableFrom(layer.GetType()));
        }

        /// <summary>
        /// Selects the last N layers in the module hierarchy.
        /// </summary>
        /// <param name="module">The root module to search.</param>
        /// <param name="count">Number of layers to select from the end.</param>
        /// <returns>Enumerable of the last N layers.</returns>
        public static IEnumerable<Module> SelectLastN(Module module, int count)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));
            if (count < 0)
                throw new ArgumentOutOfRangeException(nameof(count), "Count cannot be negative.");

            var allLayers = module.GetAllModules().ToList();
            int startIndex = Math.Max(0, allLayers.Count - count);

            for (int i = startIndex; i < allLayers.Count; i++)
            {
                yield return allLayers[i];
            }
        }

        /// <summary>
        /// Selects the first N layers in the module hierarchy.
        /// </summary>
        /// <param name="module">The root module to search.</param>
        /// <param name="count">Number of layers to select from the beginning.</param>
        /// <returns>Enumerable of the first N layers.</returns>
        public static IEnumerable<Module> SelectFirstN(Module module, int count)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));
            if (count < 0)
                throw new ArgumentOutOfRangeException(nameof(count), "Count cannot be negative.");

            var allLayers = module.GetAllModules().ToList();
            int endIndex = Math.Min(count - 1, allLayers.Count - 1);

            for (int i = 0; i <= endIndex && i < allLayers.Count; i++)
            {
                yield return allLayers[i];
            }
        }

        /// <summary>
        /// Selects layers that have parameters with gradient tracking enabled.
        /// </summary>
        /// <param name="module">The root module to search.</param>
        /// <returns>Enumerable of trainable layers.</returns>
        public static IEnumerable<Module> SelectTrainable(Module module)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));

            return module.GetAllModules()
                          .Where(layer => layer.GetParameters().Any(p => p.RequiresGrad));
        }

        /// <summary>
        /// Selects layers that have all parameters frozen (gradient tracking disabled).
        /// </summary>
        /// <param name="module">The root module to search.</param>
        /// <returns>Enumerable of frozen layers.</returns>
        public static IEnumerable<Module> SelectFrozen(Module module)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));

            return module.GetAllModules()
                          .Where(layer => !layer.GetParameters().Any(p => p.RequiresGrad));
        }

        /// <summary>
        /// Gets all modules in a hierarchical structure recursively.
        /// </summary>
        /// <param name="module">The root module.</param>
        /// <returns>All modules in the hierarchy including the root.</returns>
        private static IEnumerable<Module> GetAllModules(this Module module)
        {
            yield return module;

            if (module is SequentialModule sequential)
            {
                for (int i = 0; i < sequential.Count; i++)
                {
                    var child = sequential.GetModule(i);
                    foreach (var descendant in child.GetAllModules())
                    {
                        yield return descendant;
                    }
                }
            }
        }
    }
}
