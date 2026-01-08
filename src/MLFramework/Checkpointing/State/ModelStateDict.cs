namespace MachineLearning.Checkpointing;

/// <summary>
/// Model-specific state dictionary that organizes model parameters by layer
/// </summary>
public class ModelStateDict : StateDict
{
    /// <summary>
    /// Type of model
    /// </summary>
    public string ModelType { get; set; } = string.Empty;

    /// <summary>
    /// Number of layers in the model
    /// </summary>
    public int LayerCount { get; set; }

    /// <summary>
    /// Creates an empty model state dict
    /// </summary>
    public ModelStateDict()
    {
    }

    /// <summary>
    /// Creates a model state dict for a specific model type
    /// </summary>
    /// <param name="modelType">Type of the model</param>
    /// <param name="layerCount">Number of layers</param>
    /// <returns>A new ModelStateDict instance</returns>
    public static ModelStateDict Create(string modelType, int layerCount)
    {
        return new ModelStateDict
        {
            ModelType = modelType,
            LayerCount = layerCount
        };
    }

    /// <summary>
    /// Gets state for a specific layer
    /// </summary>
    /// <param name="layerName">Name of the layer</param>
    /// <returns>State dictionary containing the layer's parameters</returns>
    public StateDict GetLayerState(string layerName)
    {
        if (string.IsNullOrWhiteSpace(layerName))
            throw new ArgumentException("Layer name cannot be empty", nameof(layerName));

        var layerState = new StateDict();
        var prefix = $"{layerName}.";

        foreach (var (key, value) in this)
        {
            if (key.StartsWith(prefix))
            {
                var newKey = key.Substring(prefix.Length);
                layerState[newKey] = value;
            }
        }

        return layerState;
    }

    /// <summary>
    /// Sets state for a specific layer
    /// </summary>
    /// <param name="layerName">Name of the layer</param>
    /// <param name="layerState">State dictionary containing the layer's parameters</param>
    public void SetLayerState(string layerName, StateDict layerState)
    {
        if (string.IsNullOrWhiteSpace(layerName))
            throw new ArgumentException("Layer name cannot be empty", nameof(layerName));

        if (layerState == null)
            throw new ArgumentNullException(nameof(layerState));

        var prefix = $"{layerName}.";

        // Remove old layer state
        var keysToRemove = Keys.Where(k => k.StartsWith(prefix)).ToList();
        foreach (var key in keysToRemove)
        {
            Remove(key);
        }

        // Add new layer state
        foreach (var (key, value) in layerState)
        {
            this[$"{prefix}{key}"] = value;
        }
    }

    /// <summary>
    /// Gets all layer names in the model
    /// </summary>
    /// <returns>Collection of unique layer names</returns>
    public IEnumerable<string> GetLayerNames()
    {
        var layerNames = new HashSet<string>();

        foreach (var key in Keys)
        {
            var dotIndex = key.IndexOf('.');
            if (dotIndex > 0)
            {
                var layerName = key.Substring(0, dotIndex);
                layerNames.Add(layerName);
            }
        }

        return layerNames;
    }

    /// <summary>
    /// Gets model-level parameters (not associated with any specific layer)
    /// </summary>
    /// <returns>State dictionary containing model-level parameters</returns>
    public StateDict GetModelLevelState()
    {
        var modelState = new StateDict();

        foreach (var (key, value) in this)
        {
            // Parameters without a dot separator are model-level
            if (!key.Contains('.'))
            {
                modelState[key] = value;
            }
        }

        return modelState;
    }

    /// <summary>
    /// Sets model-level parameters
    /// </summary>
    /// <param name="modelState">State dictionary containing model-level parameters</param>
    public void SetModelLevelState(StateDict modelState)
    {
        if (modelState == null)
            throw new ArgumentNullException(nameof(modelState));

        // Remove old model-level state
        var keysToRemove = Keys.Where(k => !k.Contains('.')).ToList();
        foreach (var key in keysToRemove)
        {
            Remove(key);
        }

        // Add new model-level state
        foreach (var (key, value) in modelState)
        {
            this[key] = value;
        }
    }
}
