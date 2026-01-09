namespace MLFramework.ModelZoo.Progressive;

/// <summary>
/// Defines the optimal loading order for different model architectures.
/// </summary>
public class LayerLoadOrder
{
    /// <summary>
    /// Gets or sets the architecture type.
    /// </summary>
    public ModelArchitectureType ArchitectureType { get; set; }

    /// <summary>
    /// Gets or sets the ordered list of layer names to load.
    /// </summary>
    public List<string> OrderedLayers { get; set; } = new();

    /// <summary>
    /// Gets or sets a dictionary mapping layer names to their dependencies.
    /// Layers must be loaded before their dependencies.
    /// </summary>
    public Dictionary<string, List<string>> Dependencies { get; set; } = new();

    /// <summary>
    /// Creates a LayerLoadOrder for the specified architecture type.
    /// </summary>
    /// <param name="architectureType">The architecture type.</param>
    public LayerLoadOrder(ModelArchitectureType architectureType)
    {
        ArchitectureType = architectureType;
    }

    /// <summary>
    /// Creates a default load order for a CNN architecture.
    /// Layers are loaded from input to output.
    /// </summary>
    public static LayerLoadOrder ForCNN()
    {
        return new LayerLoadOrder(ModelArchitectureType.CNN);
    }

    /// <summary>
    /// Creates a default load order for a Transformer architecture.
    /// Embeddings are loaded first, then attention layers.
    /// </summary>
    public static LayerLoadOrder ForTransformer()
    {
        return new LayerLoadOrder(ModelArchitectureType.Transformer);
    }

    /// <summary>
    /// Creates a default load order for an RNN architecture.
    /// Embeddings are loaded first, then recurrent layers.
    /// </summary>
    public static LayerLoadOrder ForRNN()
    {
        return new LayerLoadOrder(ModelArchitectureType.RNN);
    }

    /// <summary>
    /// Adds a layer to the load order.
    /// </summary>
    /// <param name="layerName">The layer name.</param>
    public void AddLayer(string layerName)
    {
        OrderedLayers.Add(layerName);
    }

    /// <summary>
    /// Adds a dependency for a layer.
    /// </summary>
    /// <param name="layerName">The layer name.</param>
    /// <param name="dependencyName">The dependency layer name.</param>
    public void AddDependency(string layerName, string dependencyName)
    {
        if (!Dependencies.ContainsKey(layerName))
        {
            Dependencies[layerName] = new List<string>();
        }
        Dependencies[layerName].Add(dependencyName);
    }

    /// <summary>
    /// Gets the order in which layers should be loaded, respecting dependencies.
    /// </summary>
    /// <returns>The ordered list of layer names.</returns>
    public List<string> GetLoadOrder()
    {
        var result = new List<string>();
        var visited = new HashSet<string>();

        foreach (var layer in OrderedLayers)
        {
            Visit(layer, visited, result);
        }

        return result;
    }

    private void Visit(string layer, HashSet<string> visited, List<string> result)
    {
        if (visited.Contains(layer))
        {
            return;
        }

        visited.Add(layer);

        if (Dependencies.TryGetValue(layer, out var deps))
        {
            foreach (var dep in deps)
            {
                Visit(dep, visited, result);
            }
        }

        result.Add(layer);
    }
}

/// <summary>
/// Supported model architecture types.
/// </summary>
public enum ModelArchitectureType
{
    /// <summary>
    /// Convolutional Neural Network.
    /// </summary>
    CNN,

    /// <summary>
    /// Transformer architecture.
    /// </summary>
    Transformer,

    /// <summary>
    /// Recurrent Neural Network.
    /// </summary>
    RNN,

    /// <summary>
    /// Multi-layer perceptron / Fully connected network.
    /// </summary>
    MLP,

    /// <summary>
    /// Custom or unknown architecture.
    /// </summary>
    Custom
}
