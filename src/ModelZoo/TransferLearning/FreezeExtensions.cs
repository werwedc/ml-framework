using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using MLFramework.NN;

namespace MLFramework.ModelZoo.TransferLearning
{
    /// <summary>
    /// Extension methods for freezing and unfreezing model parameters.
    /// </summary>
    public static class FreezeExtensions
    {
        /// <summary>
        /// Freezes all parameters in the model (disables gradient computation).
        /// </summary>
        /// <param name="module">The module to freeze.</param>
        public static void Freeze(this Module module)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));

            module.SetRequiresGrad(false);
        }

        /// <summary>
        /// Unfreezes all parameters in the model (enables gradient computation).
        /// </summary>
        /// <param name="module">The module to unfreeze.</param>
        public static void Unfreeze(this Module module)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));

            module.SetRequiresGrad(true);
        }

        /// <summary>
        /// Freezes all parameters except the last N layers.
        /// </summary>
        /// <param name="module">The module to partially freeze.</param>
        /// <param name="exceptLastN">Number of layers from the end to keep unfrozen.</param>
        public static void Freeze(this Module module, int exceptLastN)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));
            if (exceptLastN < 0)
                throw new ArgumentOutOfRangeException(nameof(exceptLastN), "Number of layers cannot be negative.");

            var allLayers = module.GetAllModules().ToList();
            int freezeCount = Math.Max(0, allLayers.Count - exceptLastN);

            for (int i = 0; i < freezeCount; i++)
            {
                allLayers[i].Freeze();
            }
        }

        /// <summary>
        /// Freezes specific layers by exact name.
        /// </summary>
        /// <param name="module">The module containing layers to freeze.</param>
        /// <param name="layerNames">Names of layers to freeze.</param>
        public static void FreezeByName(this Module module, params string[] layerNames)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));
            if (layerNames == null)
                throw new ArgumentNullException(nameof(layerNames));

            var layersToFreeze = LayerSelectionHelper.SelectByName(module, layerNames);
            foreach (var layer in layersToFreeze)
            {
                layer.Freeze();
            }
        }

        /// <summary>
        /// Unfreezes specific layers by exact name.
        /// </summary>
        /// <param name="module">The module containing layers to unfreeze.</param>
        /// <param name="layerNames">Names of layers to unfreeze.</param>
        public static void UnfreezeByName(this Module module, params string[] layerNames)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));
            if (layerNames == null)
                throw new ArgumentNullException(nameof(layerNames));

            var layersToUnfreeze = LayerSelectionHelper.SelectByName(module, layerNames);
            foreach (var layer in layersToUnfreeze)
            {
                layer.Unfreeze();
            }
        }

        /// <summary>
        /// Freezes layers matching a regex pattern.
        /// </summary>
        /// <param name="module">The module containing layers to freeze.</param>
        /// <param name="regexPattern">Regular expression pattern to match layer names.</param>
        public static void FreezeByNamePattern(this Module module, string regexPattern)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));
            if (string.IsNullOrEmpty(regexPattern))
                throw new ArgumentException("Pattern cannot be null or empty.", nameof(regexPattern));

            var layersToFreeze = LayerSelectionHelper.SelectByPattern(module, regexPattern);
            foreach (var layer in layersToFreeze)
            {
                layer.Freeze();
            }
        }

        /// <summary>
        /// Unfreezes layers matching a regex pattern.
        /// </summary>
        /// <param name="module">The module containing layers to unfreeze.</param>
        /// <param name="regexPattern">Regular expression pattern to match layer names.</param>
        public static void UnfreezeByNamePattern(this Module module, string regexPattern)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));
            if (string.IsNullOrEmpty(regexPattern))
                throw new ArgumentException("Pattern cannot be null or empty.", nameof(regexPattern));

            var layersToUnfreeze = LayerSelectionHelper.SelectByPattern(module, regexPattern);
            foreach (var layer in layersToUnfreeze)
            {
                layer.Unfreeze();
            }
        }

        /// <summary>
        /// Gets all frozen layers in the module.
        /// </summary>
        /// <param name="module">The module to query.</param>
        /// <returns>Enumerable of frozen layers.</returns>
        public static IEnumerable<Module> GetFrozenLayers(this Module module)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));

            return module.GetAllModules()
                          .Where(m => !m.GetParameters().Any() || !m.GetParameters().First().RequiresGrad);
        }

        /// <summary>
        /// Gets all unfrozen layers in the module.
        /// </summary>
        /// <param name="module">The module to query.</param>
        /// <returns>Enumerable of unfrozen layers.</returns>
        public static IEnumerable<Module> GetUnfrozenLayers(this Module module)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));

            return module.GetAllModules()
                          .Where(m => m.GetParameters().Any() && m.GetParameters().First().RequiresGrad);
        }

        /// <summary>
        /// Prints the frozen/unfrozen state of all layers to the console.
        /// </summary>
        /// <param name="module">The module to print state for.</param>
        public static void PrintFrozenState(this Module module)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));

            Console.WriteLine($"Frozen State for: {module.Name}");
            Console.WriteLine(new string('-', 50));

            var allLayers = module.GetAllModules().ToList();
            foreach (var layer in allLayers)
            {
                bool hasParams = layer.GetParameters().Any();
                bool isFrozen = !hasParams || !layer.GetParameters().First().RequiresGrad;
                string status = hasParams ? (isFrozen ? "FROZEN" : "UNFROZEN") : "NO PARAMS";
                Console.WriteLine($"  {layer.Name}: {status}");
            }
        }

        /// <summary>
        /// Gets a summary of frozen/unfrozen parameters.
        /// </summary>
        /// <param name="module">The module to summarize.</param>
        /// <returns>A summary object with counts and statistics.</returns>
        public static FrozenStateSummary GetFrozenStateSummary(this Module module)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));

            var allParameters = module.GetParameters().ToList();
            int frozenCount = allParameters.Count(p => !p.RequiresGrad);
            int unfrozenCount = allParameters.Count(p => p.RequiresGrad);
            int totalParams = allParameters.Count;

            var frozenLayers = module.GetFrozenLayers().ToList();
            var unfrozenLayers = module.GetUnfrozenLayers().ToList();

            return new FrozenStateSummary
            {
                TotalParameters = totalParams,
                FrozenParameters = frozenCount,
                UnfrozenParameters = unfrozenCount,
                TotalLayers = module.GetAllModules().Count(),
                FrozenLayers = frozenLayers.Count,
                UnfrozenLayers = unfrozenLayers.Count,
                FrozenLayerNames = frozenLayers.Select(l => l.Name).ToList(),
                UnfrozenLayerNames = unfrozenLayers.Select(l => l.Name).ToList()
            };
        }

        /// <summary>
        /// Helper method to get all modules in a hierarchical structure.
        /// </summary>
        /// <param name="module">The root module.</param>
        /// <returns>All modules in the hierarchy.</returns>
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

    /// <summary>
    /// Summary of frozen/unfrozen state for a module.
    /// </summary>
    public class FrozenStateSummary
    {
        /// <summary>
        /// Total number of parameters in the module.
        /// </summary>
        public int TotalParameters { get; set; }

        /// <summary>
        /// Number of frozen parameters.
        /// </summary>
        public int FrozenParameters { get; set; }

        /// <summary>
        /// Number of unfrozen parameters.
        /// </summary>
        public int UnfrozenParameters { get; set; }

        /// <summary>
        /// Total number of layers in the module.
        /// </summary>
        public int TotalLayers { get; set; }

        /// <summary>
        /// Number of frozen layers.
        /// </summary>
        public int FrozenLayers { get; set; }

        /// <summary>
        /// Number of unfrozen layers.
        /// </summary>
        public int UnfrozenLayers { get; set; }

        /// <summary>
        /// Names of frozen layers.
        /// </summary>
        public List<string> FrozenLayerNames { get; set; } = new List<string>();

        /// <summary>
        /// Names of unfrozen layers.
        /// </summary>
        public List<string> UnfrozenLayerNames { get; set; } = new List<string>();

        /// <summary>
        /// Percentage of parameters that are frozen.
        /// </summary>
        public double FrozenParameterPercentage => TotalParameters > 0 ? (double)FrozenParameters / TotalParameters * 100 : 0;

        /// <summary>
        /// Percentage of parameters that are unfrozen.
        /// </summary>
        public double UnfrozenParameterPercentage => TotalParameters > 0 ? (double)UnfrozenParameters / TotalParameters * 100 : 0;

        /// <summary>
        /// Gets a formatted string representation of the summary.
        /// </summary>
        /// <returns>Formatted summary string.</returns>
        public override string ToString()
        {
            return $"Frozen: {FrozenParameters}/{TotalParameters} ({FrozenParameterPercentage:F1}%), " +
                   $"Unfrozen: {UnfrozenParameters}/{TotalParameters} ({UnfrozenParameterPercentage:F1}%), " +
                   $"Layers: {FrozenLayers} frozen, {UnfrozenLayers} unfrozen";
        }
    }
}
