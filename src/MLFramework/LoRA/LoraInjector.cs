using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using RitterFramework.Core.LoRA;
using RitterFramework.Core.Tensor;
using MLFramework.Modules;
using MLFramework.LoRA;
using MLFramework.NN;
using MLFrameworkIModule = MLFramework.Modules.IModule;

namespace MLFramework.LoRA
{
    /// <summary>
    /// Static utility class for injecting and removing LoRA layers from models.
    /// Automatically replaces target Linear layers with LoraLinear wrappers.
    /// </summary>
    public static class LoraInjector
    {
        /// <summary>
        /// Inject LoRA layers into a model based on the configuration.
        /// Replaces target Linear layers with LoraLinear wrappers.
        /// </summary>
        /// <param name="model">The model to inject LoRA into</param>
        /// <param name="config">LoRA configuration specifying target modules</param>
        public static void Inject(IHierarchicalModule model, LoraConfig config)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            config = config ?? new LoraConfig();
            config.Validate();

            var targetPatterns = ParseTargetModules(config.TargetModules);
            InjectRecursive(model, config, targetPatterns);
        }

        /// <summary>
        /// Inject LoRA layers into a model using extension method for easier API
        /// </summary>
        /// <param name="model">The model to inject LoRA into</param>
        /// <param name="config">LoRA configuration</param>
        public static void Inject(MLFrameworkIModule model, LoraConfig config)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            // If it's already hierarchical, use the direct method
            if (model is IHierarchicalModule hierarchical)
            {
                Inject(hierarchical, config);
                return;
            }

            throw new ArgumentException(
                "Model must implement IHierarchicalModule for LoRA injection. " +
                "Use HierarchicalWrapper or extend HierarchicalModule.");
        }

        /// <summary>
        /// Remove LoRA layers from a model, restoring the original Linear layers.
        /// </summary>
        /// <param name="model">The model to remove LoRA from</param>
        public static void Remove(IHierarchicalModule model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            RemoveRecursive(model);
        }

        /// <summary>
        /// Remove LoRA layers from a model using extension method
        /// </summary>
        /// <param name="model">The model to remove LoRA from</param>
        public static void Remove(MLFrameworkIModule model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            if (model is IHierarchicalModule hierarchical)
            {
                Remove(hierarchical);
                return;
            }

            throw new ArgumentException("Model must implement IHierarchicalModule for LoRA removal");
        }

        /// <summary>
        /// Check if a model has LoRA layers injected.
        /// </summary>
        /// <param name="model">The model to check</param>
        /// <returns>True if LoRA layers are present, false otherwise</returns>
        public static bool HasLoRA(IHierarchicalModule model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            return HasLoRARecursive(model);
        }

        /// <summary>
        /// Check if a model has LoRA layers using extension method
        /// </summary>
        /// <param name="model">The model to check</param>
        /// <returns>True if LoRA layers are present, false otherwise</returns>
        public static bool HasLoRA(MLFrameworkIModule model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            if (model is IHierarchicalModule hierarchical)
            {
                return HasLoRA(hierarchical);
            }

            return false;
        }

        /// <summary>
        /// Get all LoRA-injected layers in a model.
        /// </summary>
        /// <param name="model">The model to search</param>
        /// <returns>List of all LoraLinear layers in the model</returns>
        public static List<LoraLinear> GetLoRALayers(IHierarchicalModule model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            var layers = new List<LoraLinear>();
            FindLoRALayersRecursive(model, layers);
            return layers;
        }

        /// <summary>
        /// Get all LoRA-injected layers using extension method
        /// </summary>
        /// <param name="model">The model to search</param>
        /// <returns>List of all LoraLinear layers in the model</returns>
        public static List<LoraLinear> GetLoRALayers(MLFrameworkIModule model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            if (model is IHierarchicalModule hierarchical)
            {
                return GetLoRALayers(hierarchical);
            }

            return new List<LoraLinear>();
        }

        #region Private Methods

        /// <summary>
        /// Recursively inject LoRA into modules matching target patterns
        /// </summary>
        private static void InjectRecursive(
            IHierarchicalModule module,
            LoraConfig config,
            List<ModuleTargetPattern> patterns)
        {
            // Check if this is a wrapper around a Linear layer that needs LoRA
            if (module is LinearWrapper linearWrapper && ShouldInject(module.Name, patterns))
            {
                // Check if already has LoRA
                if (linearWrapper.Module is LoraLinear)
                {
                    // Already has LoRA, skip
                }
                else if (linearWrapper.Module is Linear linear)
                {
                    // Create LoraLinear to wrap the Linear
                    var loraLayer = new LoraLinear(linear, config.Rank, config.Alpha, config.Dropout);

                    // Create a new wrapper for the LoraLinear
                    var loraWrapper = new LinearWrapper(loraLayer, module.Name);

                    // Replace in parent
                    var parent = module.Parent;
                    if (parent != null)
                    {
                        parent.ReplaceChild(module.Name, loraWrapper);
                    }
                }
            }

            // Recursively process child modules
            foreach (var child in module.Children().ToList())
            {
                InjectRecursive(child, config, patterns);
            }
        }

        /// <summary>
        /// Recursively remove LoRA from modules, restoring original Linear layers
        /// </summary>
        private static void RemoveRecursive(IHierarchicalModule module)
        {
            if (module is LinearWrapper linearWrapper)
            {
                var underlyingModule = linearWrapper.Module;

                // Check if it's a LoraLinear
                if (underlyingModule is LoraLinear loraLinear)
                {
                    // Get the base Linear layer
                    var baseLinear = loraLinear.GetBaseLinear();

                    // Create a new wrapper for the base Linear
                    var newWrapper = new LinearWrapper(baseLinear, module.Name);

                    // Replace in parent
                    var parent = module.Parent;
                    if (parent != null)
                    {
                        parent.ReplaceChild(module.Name, newWrapper);
                    }
                }
            }

            // Recursively process child modules
            foreach (var child in module.Children().ToList())
            {
                RemoveRecursive(child);
            }
        }

        /// <summary>
        /// Recursively check if module has LoRA
        /// </summary>
        private static bool HasLoRARecursive(IHierarchicalModule module)
        {
            // Check if this module has LoRA
            if (module is LinearWrapper linearWrapper)
            {
                if (linearWrapper.Module is LoraLinear)
                {
                    return true;
                }
            }

            // Check children
            foreach (var child in module.Children())
            {
                if (HasLoRARecursive(child))
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Recursively find all LoRA layers
        /// </summary>
        private static void FindLoRALayersRecursive(IHierarchicalModule module, List<LoraLinear> layers)
        {
            if (module is LinearWrapper linearWrapper)
            {
                if (linearWrapper.Module is LoraLinear loraLinear)
                {
                    layers.Add(loraLinear);
                }
            }

            foreach (var child in module.Children())
            {
                FindLoRALayersRecursive(child, layers);
            }
        }

        /// <summary>
        /// Check if a module name matches any of the target patterns
        /// </summary>
        private static bool ShouldInject(string moduleName, List<ModuleTargetPattern> patterns)
        {
            if (string.IsNullOrEmpty(moduleName))
                return false;

            return patterns.Any(p => p.Matches(moduleName));
        }

        /// <summary>
        /// Parse target module strings into pattern objects
        /// </summary>
        private static List<ModuleTargetPattern> ParseTargetModules(string[] targetModules)
        {
            var patterns = new List<ModuleTargetPattern>();

            if (targetModules == null || targetModules.Length == 0)
            {
                // Default to common targets
                patterns.Add(ModuleTargetPattern.FromString("q_proj"));
                patterns.Add(ModuleTargetPattern.FromString("v_proj"));
                return patterns;
            }

            foreach (var target in targetModules)
            {
                patterns.Add(ModuleTargetPattern.FromString(target));
            }

            return patterns;
        }

        #endregion
    }
}
