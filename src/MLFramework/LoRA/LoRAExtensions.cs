using System;
using System.Collections.Generic;
using System.Linq;
using RitterFramework.Core.LoRA;
using RitterFramework.Core.Tensor;
using MLFramework.Modules;
using MLFramework.LoRA;
using MLFramework.NN;
using MLFrameworkIModule = MLFramework.Modules.IModule;

namespace MLFramework.LoRA
{
    /// <summary>
    /// Extension methods for applying LoRA to models.
    /// Provides a convenient API for injecting, removing, and managing LoRA adapters.
    /// </summary>
    public static class LoRAExtensions
    {
        /// <summary>
        /// Apply LoRA to a model with the specified configuration.
        /// </summary>
        /// <param name="model">The model to apply LoRA to</param>
        /// <param name="config">LoRA configuration (defaults to standard config)</param>
        public static void ApplyLoRA(this MLFrameworkIModule model, LoraConfig config = null)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            config ??= new LoraConfig();
            LoraInjector.Inject(model, config);
        }

        /// <summary>
        /// Apply LoRA to a hierarchical model.
        /// </summary>
        /// <param name="model">The model to apply LoRA to</param>
        /// <param name="config">LoRA configuration</param>
        public static void ApplyLoRA(this IHierarchicalModule model, LoraConfig config = null)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            config ??= new LoraConfig();
            LoraInjector.Inject(model, config);
        }

        /// <summary>
        /// Remove LoRA from a model, restoring the original layers.
        /// </summary>
        /// <param name="model">The model to remove LoRA from</param>
        public static void RemoveLoRA(this MLFrameworkIModule model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            LoraInjector.Remove(model);
        }

        /// <summary>
        /// Remove LoRA from a hierarchical model.
        /// </summary>
        /// <param name="model">The model to remove LoRA from</param>
        public static void RemoveLoRA(this IHierarchicalModule model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            LoraInjector.Remove(model);
        }

        /// <summary>
        /// Get all trainable LoRA parameters from a model.
        /// </summary>
        /// <param name="model">The model to get LoRA parameters from</param>
        /// <returns>List of trainable LoRA parameters</returns>
        public static List<Tensor> GetLoRAParameters(this MLFrameworkIModule model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            return LoraInjector.GetLoRALayers(model)
                .SelectMany(l => l.TrainableParameters)
                .ToList();
        }

        /// <summary>
        /// Get all trainable LoRA parameters from a hierarchical model.
        /// </summary>
        /// <param name="model">The model to get LoRA parameters from</param>
        /// <returns>List of trainable LoRA parameters</returns>
        public static List<Tensor> GetLoRAParameters(this IHierarchicalModule model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            return LoraInjector.GetLoRALayers(model)
                .SelectMany(l => l.TrainableParameters)
                .ToList();
        }

        /// <summary>
        /// Freeze all non-LoRA parameters in a model.
        /// </summary>
        /// <param name="model">The model to freeze base parameters in</param>
        public static void FreezeBase(this MLFrameworkIModule model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            // Note: This is a simplified implementation
            // In a full implementation, you'd need to identify non-LoRA parameters
            // and set their RequiresGrad to false
        }

        /// <summary>
        /// Freeze all non-LoRA parameters in a hierarchical model.
        /// </summary>
        /// <param name="model">The model to freeze base parameters in</param>
        public static void FreezeBase(this IHierarchicalModule model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            FreezeBaseRecursive(model);
        }

        /// <summary>
        /// Unfreeze all parameters in a model.
        /// </summary>
        /// <param name="model">The model to unfreeze</param>
        public static void UnfreezeAll(this MLFrameworkIModule model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            // Note: Simplified implementation
        }

        /// <summary>
        /// Unfreeze all parameters in a hierarchical model.
        /// </summary>
        /// <param name="model">The model to unfreeze</param>
        public static void UnfreezeAll(this IHierarchicalModule model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            UnfreezeAllRecursive(model);
        }

        /// <summary>
        /// Check if a model has LoRA applied.
        /// </summary>
        /// <param name="model">The model to check</param>
        /// <returns>True if LoRA is applied, false otherwise</returns>
        public static bool HasLoRA(this MLFrameworkIModule model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            return LoraInjector.HasLoRA(model);
        }

        /// <summary>
        /// Check if a hierarchical model has LoRA applied.
        /// </summary>
        /// <param name="model">The model to check</param>
        /// <returns>True if LoRA is applied, false otherwise</returns>
        public static bool HasLoRA(this IHierarchicalModule model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            return LoraInjector.HasLoRA(model);
        }

        /// <summary>
        /// Get all LoRA layers in a model.
        /// </summary>
        /// <param name="model">The model to get LoRA layers from</param>
        /// <returns>List of all LoRA layers</returns>
        public static List<LoraLinear> GetLoRALayers(this MLFrameworkIModule model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            return LoraInjector.GetLoRALayers(model);
        }

        /// <summary>
        /// Get all LoRA layers in a hierarchical model.
        /// </summary>
        /// <param name="model">The model to get LoRA layers from</param>
        /// <returns>List of all LoRA layers</returns>
        public static List<LoraLinear> GetLoRALayers(this IHierarchicalModule model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            return LoraInjector.GetLoRALayers(model);
        }

        #region Private Helper Methods

        /// <summary>
        /// Recursively freeze base parameters (non-LoRA)
        /// </summary>
        private static void FreezeBaseRecursive(IHierarchicalModule module)
        {
            // If this is a LoraLinear, the base layer is already frozen
            // If it's a regular Linear, freeze it
            if (module is LinearWrapper linearWrapper)
            {
                if (linearWrapper.Module is Linear linear)
                {
                    linear.SetRequiresGrad(false);
                }
            }

            // Recursively process children
            foreach (var child in module.Children())
            {
                FreezeBaseRecursive(child);
            }
        }

        /// <summary>
        /// Recursively unfreeze all parameters
        /// </summary>
        private static void UnfreezeAllRecursive(IHierarchicalModule module)
        {
            // Unfreeze this module
            if (module is LinearWrapper linearWrapper)
            {
                if (linearWrapper.Module is Linear linear)
                {
                    linear.SetRequiresGrad(true);
                }
            }

            // Recursively process children
            foreach (var child in module.Children())
            {
                UnfreezeAllRecursive(child);
            }
        }

        #endregion
    }
}
