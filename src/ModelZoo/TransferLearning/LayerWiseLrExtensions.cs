using System;
using System.Collections.Generic;
using System.Linq;
using MLFramework.NN;
using MLFramework.Optimizers;
using RitterFramework.Core.Tensor;

namespace MLFramework.ModelZoo.TransferLearning;

/// <summary>
/// Extension methods for applying layer-wise learning rates to optimizers.
/// </summary>
public static class LayerWiseLrExtensions
{
    /// <summary>
    /// Sets specific learning rates for each layer by name.
    /// </summary>
    /// <param name="optimizer">The optimizer to configure.</param>
    /// <param name="layerLrs">Dictionary mapping layer names to learning rates.</param>
    public static void SetLayerWiseLrs(this Optimizer optimizer, Dictionary<string, float> layerLrs)
    {
        if (optimizer == null)
            throw new ArgumentNullException(nameof(optimizer));

        if (layerLrs == null)
            throw new ArgumentNullException(nameof(layerLrs));

        var parameters = optimizer.Parameters;
        var groupMultipliers = new Dictionary<string, float>();

        // Determine which group each parameter belongs to and its multiplier
        foreach (var paramName in parameters.Keys)
        {
            float multiplier = 1.0f;

            // Find the layer name that contains this parameter
            foreach (var layerName in layerLrs.Keys)
            {
                if (paramName.StartsWith(layerName))
                {
                    multiplier = layerLrs[layerName] / optimizer.BaseLearningRate;
                    break;
                }
            }

            groupMultipliers[paramName] = multiplier;
        }

        // Store multipliers in the optimizer's state (custom field)
        optimizer.SetParameterGroupMultipliers(groupMultipliers);
    }

    /// <summary>
    /// Sets learning rates for frozen and unfrozen parameters.
    /// </summary>
    /// <param name="optimizer">The optimizer to configure.</param>
    /// <param name="model">The model containing the parameters.</param>
    /// <param name="frozenLr">Learning rate for frozen parameters.</param>
    /// <param name="unfrozenLr">Learning rate for unfrozen parameters.</param>
    public static void SetLayerWiseLrs(this Optimizer optimizer, Module model, float frozenLr, float unfrozenLr)
    {
        if (optimizer == null)
            throw new ArgumentNullException(nameof(optimizer));

        if (model == null)
            throw new ArgumentNullException(nameof(model));

        var parameters = optimizer.Parameters;
        var groupMultipliers = new Dictionary<string, float>();
        float baseLr = optimizer.BaseLearningRate;

        foreach (var (paramName, param) in model.GetNamedParameters())
        {
            if (parameters.ContainsKey(paramName))
            {
                float multiplier = param.RequiresGrad ? unfrozenLr / baseLr : frozenLr / baseLr;
                groupMultipliers[paramName] = multiplier;
            }
        }

        optimizer.SetParameterGroupMultipliers(groupMultipliers);
    }

    /// <summary>
    /// Sets a learning rate schedule for specific layers.
    /// </summary>
    /// <param name="optimizer">The optimizer to configure.</param>
    /// <param name="lrSchedule">Array of learning rates corresponding to layers.</param>
    /// <param name="layerNames">Array of layer names corresponding to the learning rates.</param>
    public static void SetLayerWiseLrs(this Optimizer optimizer, float[] lrSchedule, string[] layerNames)
    {
        if (optimizer == null)
            throw new ArgumentNullException(nameof(optimizer));

        if (lrSchedule == null)
            throw new ArgumentNullException(nameof(lrSchedule));

        if (layerNames == null)
            throw new ArgumentNullException(nameof(layerNames));

        if (lrSchedule.Length != layerNames.Length)
            throw new ArgumentException("Learning rate schedule and layer names must have the same length.");

        var layerLrs = new Dictionary<string, float>();
        for (int i = 0; i < layerNames.Length; i++)
        {
            layerLrs[layerNames[i]] = lrSchedule[i];
        }

        optimizer.SetLayerWiseLrs(layerLrs);
    }

    /// <summary>
    /// Gets the current parameter groups with their learning rates.
    /// </summary>
    /// <param name="optimizer">The optimizer to query.</param>
    /// <param name="model">The model containing the parameters.</param>
    /// <returns>List of parameter groups with their settings.</returns>
    public static List<ParameterGroup> GetParameterGroups(this Optimizer optimizer, Module model)
    {
        if (optimizer == null)
            throw new ArgumentNullException(nameof(optimizer));

        if (model == null)
            throw new ArgumentNullException(nameof(model));

        var groups = new Dictionary<string, ParameterGroup>();
        var parameters = optimizer.Parameters;

        // Get parameter multipliers if they exist
        var multipliers = optimizer.GetParameterGroupMultipliers();

        foreach (var (paramName, param) in model.GetNamedParameters())
        {
            if (!parameters.ContainsKey(paramName))
                continue;

            // Determine which group this parameter belongs to
            string groupName = GetGroupName(paramName);
            float lr = optimizer.BaseLearningRate;

            if (multipliers != null && multipliers.ContainsKey(paramName))
            {
                lr = optimizer.BaseLearningRate * multipliers[paramName];
            }

            // Create or update the group
            if (!groups.ContainsKey(groupName))
            {
                groups[groupName] = new ParameterGroup(groupName, lr);
            }

            groups[groupName].AddParameter(param, paramName);
        }

        return groups.Values.ToList();
    }

    /// <summary>
    /// Applies parameter groups to the optimizer.
    /// Each group will have its own learning rate and potentially other hyperparameters.
    /// </summary>
    /// <param name="optimizer">The optimizer to configure.</param>
    /// <param name="parameterGroups">List of parameter groups to apply.</param>
    public static void ApplyParameterGroups(this Optimizer optimizer, List<ParameterGroup> parameterGroups)
    {
        if (optimizer == null)
            throw new ArgumentNullException(nameof(optimizer));

        if (parameterGroups == null)
            throw new ArgumentNullException(nameof(parameterGroups));

        var multipliers = new Dictionary<string, float>();
        float baseLr = optimizer.BaseLearningRate;

        foreach (var group in parameterGroups)
        {
            foreach (var paramName in group.ParameterNames)
            {
                float multiplier = group.LearningRate / baseLr;
                multipliers[paramName] = multiplier;
            }
        }

        optimizer.SetParameterGroupMultipliers(multipliers);
    }

    /// <summary>
    /// Builds parameter groups from a parameter group builder and applies them to the optimizer.
    /// </summary>
    /// <param name="optimizer">The optimizer to configure.</param>
    /// <param name="builder">The parameter group builder.</param>
    public static void ApplyParameterGroups(this Optimizer optimizer, ParameterGroupBuilder builder)
    {
        if (optimizer == null)
            throw new ArgumentNullException(nameof(optimizer));

        if (builder == null)
            throw new ArgumentNullException(nameof(builder));

        var groups = builder.Build();
        optimizer.ApplyParameterGroups(groups);
    }

    /// <summary>
    /// Creates a parameter group builder configured for discriminative fine-tuning.
    /// Early layers have lower learning rates, later layers have higher.
    /// </summary>
    /// <param name="optimizer">The optimizer being configured (used for base LR).</param>
    /// <param name="model">The model to configure.</param>
    /// <param name="numGroups">Number of parameter groups to create.</param>
    /// <param name="baseMultiplier">Multiplier for the first group (early layers).</param>
    /// <param name="finalMultiplier">Multiplier for the last group (late layers).</param>
    /// <returns>ParameterGroupBuilder with configured groups.</returns>
    public static ParameterGroupBuilder CreateDiscriminativeGroups(
        this Optimizer optimizer,
        Module model,
        int numGroups,
        float baseMultiplier = 0.1f,
        float finalMultiplier = 1.0f)
    {
        if (optimizer == null)
            throw new ArgumentNullException(nameof(optimizer));

        if (model == null)
            throw new ArgumentNullException(nameof(model));

        if (numGroups <= 0)
            throw new ArgumentOutOfRangeException(nameof(numGroups), "Number of groups must be positive.");

        var allLayers = GetAllLayers(model).ToList();
        int layersPerGroup = Math.Max(1, allLayers.Count / numGroups);
        float baseLr = optimizer.BaseLearningRate;

        var builder = new ParameterGroupBuilder();

        for (int i = 0; i < numGroups; i++)
        {
            // Calculate learning rate for this group (geometric progression)
            float t = i / (float)(numGroups - 1);
            float multiplier = baseMultiplier + (finalMultiplier - baseMultiplier) * t;
            float groupLr = baseLr * multiplier;

            string groupName = $"group_{i}";
            builder.NewGroup(groupName, groupLr);

            // Add layers to this group
            int startIdx = i * layersPerGroup;
            int endIdx = Math.Min((i + 1) * layersPerGroup - 1, allLayers.Count - 1);

            for (int j = startIdx; j <= endIdx && j < allLayers.Count; j++)
            {
                foreach (var (paramName, param) in allLayers[j].GetNamedParameters())
                {
                    builder.AddParameter(param, paramName);
                }
            }
        }

        return builder;
    }

    /// <summary>
    /// Helper method to get all layers from a module.
    /// </summary>
    private static IEnumerable<Module> GetAllLayers(Module module)
    {
        yield return module;

        if (module is SequentialModule sequential)
        {
            for (int i = 0; i < sequential.Count; i++)
            {
                var child = sequential.GetModule(i);
                foreach (var descendant in GetAllLayers(child))
                {
                    yield return descendant;
                }
            }
        }
    }

    /// <summary>
    /// Helper method to extract group name from parameter name.
    /// </summary>
    private static string GetGroupName(string paramName)
    {
        // Extract the layer name from the parameter name
        // Assumes parameter names are formatted like "layer_name.param_name"
        int lastDot = paramName.LastIndexOf('.');
        if (lastDot > 0)
        {
            return paramName.Substring(0, lastDot);
        }
        return "default";
    }

    #region Parameter Group Multipliers Storage

    /// <summary>
    /// Storage for parameter group multipliers.
    /// Key: Optimizer instance, Value: Dictionary of parameter multipliers.
    /// </summary>
    private static readonly Dictionary<Optimizer, Dictionary<string, float>> _parameterGroupMultipliers =
        new Dictionary<Optimizer, Dictionary<string, float>>();

    /// <summary>
    /// Sets the parameter group multipliers for an optimizer.
    /// </summary>
    private static void SetParameterGroupMultipliers(this Optimizer optimizer, Dictionary<string, float> multipliers)
    {
        _parameterGroupMultipliers[optimizer] = multipliers;
    }

    /// <summary>
    /// Gets the parameter group multipliers for an optimizer.
    /// </summary>
    private static Dictionary<string, float> GetParameterGroupMultipliers(this Optimizer optimizer)
    {
        if (_parameterGroupMultipliers.TryGetValue(optimizer, out var multipliers))
        {
            return multipliers;
        }
        return null;
    }

    /// <summary>
    /// Clears the parameter group multipliers for an optimizer.
    /// </summary>
    public static void ClearLayerWiseLrs(this Optimizer optimizer)
    {
        if (optimizer == null)
            throw new ArgumentNullException(nameof(optimizer));

        _parameterGroupMultipliers.Remove(optimizer);
    }

    #endregion
}
