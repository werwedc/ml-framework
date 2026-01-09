using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using MLFramework.NN;
using RitterFramework.Core.Tensor;

namespace MLFramework.ModelZoo.TransferLearning;

/// <summary>
/// Fluent API builder for creating parameter groups with shared optimizer settings.
/// </summary>
public class ParameterGroupBuilder
{
    private readonly List<ParameterGroup> _groups;
    private ParameterGroup _currentGroup;

    /// <summary>
    /// Gets the number of parameter groups built so far.
    /// </summary>
    public int GroupCount => _groups.Count;

    /// <summary>
    /// Initializes a new instance of the ParameterGroupBuilder class.
    /// </summary>
    public ParameterGroupBuilder()
    {
        _groups = new List<ParameterGroup>();
    }

    /// <summary>
    /// Starts a new parameter group with the specified name and learning rate.
    /// </summary>
    /// <param name="groupName">Name for the parameter group.</param>
    /// <param name="learningRate">Learning rate for this group.</param>
    /// <returns>This builder for method chaining.</returns>
    public ParameterGroupBuilder NewGroup(string groupName, float learningRate)
    {
        if (string.IsNullOrEmpty(groupName))
            throw new ArgumentException("Group name cannot be null or empty.", nameof(groupName));

        // Save the current group if it exists
        if (_currentGroup != null)
        {
            _groups.Add(_currentGroup);
        }

        // Start a new group
        _currentGroup = new ParameterGroup(groupName, learningRate);
        return this;
    }

    /// <summary>
    /// Starts a new parameter group with full configuration.
    /// </summary>
    /// <param name="groupName">Name for the parameter group.</param>
    /// <param name="learningRate">Learning rate for this group.</param>
    /// <param name="weightDecay">Weight decay for this group.</param>
    /// <param name="momentum">Momentum value for this group (optional).</param>
    /// <returns>This builder for method chaining.</returns>
    public ParameterGroupBuilder NewGroup(string groupName, float learningRate, float weightDecay, float? momentum = null)
    {
        if (string.IsNullOrEmpty(groupName))
            throw new ArgumentException("Group name cannot be null or empty.", nameof(groupName));

        // Save the current group if it exists
        if (_currentGroup != null)
        {
            _groups.Add(_currentGroup);
        }

        // Start a new group
        _currentGroup = new ParameterGroup(groupName, learningRate, weightDecay, momentum);
        return this;
    }

    /// <summary>
    /// Adds all frozen parameters (those with RequiresGrad = false) to the current group.
    /// </summary>
    /// <param name="module">The module to extract frozen parameters from.</param>
    /// <returns>This builder for method chaining.</returns>
    public ParameterGroupBuilder AddFrozenParameters(Module module)
    {
        if (module == null)
            throw new ArgumentNullException(nameof(module));

        if (_currentGroup == null)
            throw new InvalidOperationException("No active group. Call NewGroup() first.");

        foreach (var (paramName, param) in module.GetNamedParameters())
        {
            if (!param.RequiresGrad)
            {
                _currentGroup.AddParameter(param, paramName);
            }
        }

        return this;
    }

    /// <summary>
    /// Adds all unfrozen parameters (those with RequiresGrad = true) to the current group.
    /// </summary>
    /// <param name="module">The module to extract unfrozen parameters from.</param>
    /// <returns>This builder for method chaining.</returns>
    public ParameterGroupBuilder AddUnfrozenParameters(Module module)
    {
        if (module == null)
            throw new ArgumentNullException(nameof(module));

        if (_currentGroup == null)
            throw new InvalidOperationException("No active group. Call NewGroup() first.");

        foreach (var (paramName, param) in module.GetNamedParameters())
        {
            if (param.RequiresGrad)
            {
                _currentGroup.AddParameter(param, paramName);
            }
        }

        return this;
    }

    /// <summary>
    /// Adds all parameters from a specific layer to the current group.
    /// </summary>
    /// <param name="layerName">Name of the layer to add parameters from.</param>
    /// <param name="module">The module containing the layer.</param>
    /// <returns>This builder for method chaining.</returns>
    public ParameterGroupBuilder AddLayer(string layerName, Module module)
    {
        if (string.IsNullOrEmpty(layerName))
            throw new ArgumentException("Layer name cannot be null or empty.", nameof(layerName));

        if (module == null)
            throw new ArgumentNullException(nameof(module));

        if (_currentGroup == null)
            throw new InvalidOperationException("No active group. Call NewGroup() first.");

        var layer = LayerSelectionHelper.SelectByName(module, layerName).FirstOrDefault();
        if (layer == null)
            throw new ArgumentException($"Layer '{layerName}' not found in module.", nameof(layerName));

        foreach (var (paramName, param) in layer.GetNamedParameters())
        {
            _currentGroup.AddParameter(param, paramName);
        }

        return this;
    }

    /// <summary>
    /// Adds all parameters from layers matching a regex pattern to the current group.
    /// </summary>
    /// <param name="pattern">Regular expression pattern to match layer names.</param>
    /// <param name="module">The module containing the layers.</param>
    /// <returns>This builder for method chaining.</returns>
    public ParameterGroupBuilder AddLayersByPattern(string pattern, Module module)
    {
        if (string.IsNullOrEmpty(pattern))
            throw new ArgumentException("Pattern cannot be null or empty.", nameof(pattern));

        if (module == null)
            throw new ArgumentNullException(nameof(module));

        if (_currentGroup == null)
            throw new InvalidOperationException("No active group. Call NewGroup() first.");

        var matchingLayers = LayerSelectionHelper.SelectByPattern(module, pattern);
        foreach (var layer in matchingLayers)
        {
            foreach (var (paramName, param) in layer.GetNamedParameters())
            {
                _currentGroup.AddParameter(param, paramName);
            }
        }

        return this;
    }

    /// <summary>
    /// Adds parameters from the first N layers in the module hierarchy.
    /// </summary>
    /// <param name="count">Number of layers from the beginning to include.</param>
    /// <param name="module">The module to extract layers from.</param>
    /// <returns>This builder for method chaining.</returns>
    public ParameterGroupBuilder AddFirstNLayers(int count, Module module)
    {
        if (count <= 0)
            throw new ArgumentOutOfRangeException(nameof(count), "Count must be positive.");

        if (module == null)
            throw new ArgumentNullException(nameof(module));

        if (_currentGroup == null)
            throw new InvalidOperationException("No active group. Call NewGroup() first.");

        var layers = LayerSelectionHelper.SelectFirstN(module, count);
        foreach (var layer in layers)
        {
            foreach (var (paramName, param) in layer.GetNamedParameters())
            {
                _currentGroup.AddParameter(param, paramName);
            }
        }

        return this;
    }

    /// <summary>
    /// Adds parameters from the last N layers in the module hierarchy.
    /// </summary>
    /// <param name="count">Number of layers from the end to include.</param>
    /// <param name="module">The module to extract layers from.</param>
    /// <returns>This builder for method chaining.</returns>
    public ParameterGroupBuilder AddLastNLayers(int count, Module module)
    {
        if (count <= 0)
            throw new ArgumentOutOfRangeException(nameof(count), "Count must be positive.");

        if (module == null)
            throw new ArgumentNullException(nameof(module));

        if (_currentGroup == null)
            throw new InvalidOperationException("No active group. Call NewGroup() first.");

        var layers = LayerSelectionHelper.SelectLastN(module, count);
        foreach (var layer in layers)
        {
            foreach (var (paramName, param) in layer.GetNamedParameters())
            {
                _currentGroup.AddParameter(param, paramName);
            }
        }

        return this;
    }

    /// <summary>
    /// Adds parameters from layers within a range of indices.
    /// </summary>
    /// <param name="startIndex">Starting index (inclusive).</param>
    /// <param name="endIndex">Ending index (inclusive).</param>
    /// <param name="module">The module to extract layers from.</param>
    /// <returns>This builder for method chaining.</returns>
    public ParameterGroupBuilder AddLayerRange(int startIndex, int endIndex, Module module)
    {
        if (startIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(startIndex), "Start index cannot be negative.");

        if (module == null)
            throw new ArgumentNullException(nameof(module));

        if (_currentGroup == null)
            throw new InvalidOperationException("No active group. Call NewGroup() first.");

        var layers = LayerSelectionHelper.SelectByRange(module, startIndex, endIndex);
        foreach (var layer in layers)
        {
            foreach (var (paramName, param) in layer.GetNamedParameters())
            {
                _currentGroup.AddParameter(param, paramName);
            }
        }

        return this;
    }

    /// <summary>
    /// Adds a specific parameter to the current group.
    /// </summary>
    /// <param name="parameter">The parameter to add.</param>
    /// <param name="parameterName">The name of the parameter.</param>
    /// <returns>This builder for method chaining.</returns>
    public ParameterGroupBuilder AddParameter(Tensor parameter, string parameterName)
    {
        if (parameter == null)
            throw new ArgumentNullException(nameof(parameter));

        if (string.IsNullOrEmpty(parameterName))
            throw new ArgumentException("Parameter name cannot be null or empty.", nameof(parameterName));

        if (_currentGroup == null)
            throw new InvalidOperationException("No active group. Call NewGroup() first.");

        _currentGroup.AddParameter(parameter, parameterName);
        return this;
    }

    /// <summary>
    /// Adds multiple parameters to the current group.
    /// </summary>
    /// <param name="parameters">Dictionary of parameter names to tensors.</param>
    /// <returns>This builder for method chaining.</returns>
    public ParameterGroupBuilder AddParameters(IDictionary<string, Tensor> parameters)
    {
        if (parameters == null)
            throw new ArgumentNullException(nameof(parameters));

        if (_currentGroup == null)
            throw new InvalidOperationException("No active group. Call NewGroup() first.");

        _currentGroup.AddParameters(parameters);
        return this;
    }

    /// <summary>
    /// Sets the learning rate for the current group.
    /// </summary>
    /// <param name="learningRate">The learning rate to set.</param>
    /// <returns>This builder for method chaining.</returns>
    public ParameterGroupBuilder WithLearningRate(float learningRate)
    {
        if (_currentGroup == null)
            throw new InvalidOperationException("No active group. Call NewGroup() first.");

        _currentGroup.LearningRate = learningRate;
        return this;
    }

    /// <summary>
    /// Sets the weight decay for the current group.
    /// </summary>
    /// <param name="weightDecay">The weight decay to set.</param>
    /// <returns>This builder for method chaining.</returns>
    public ParameterGroupBuilder WithWeightDecay(float weightDecay)
    {
        if (_currentGroup == null)
            throw new InvalidOperationException("No active group. Call NewGroup() first.");

        _currentGroup.WeightDecay = weightDecay;
        return this;
    }

    /// <summary>
    /// Sets the momentum for the current group.
    /// </summary>
    /// <param name="momentum">The momentum value to set.</param>
    /// <returns>This builder for method chaining.</returns>
    public ParameterGroupBuilder WithMomentum(float momentum)
    {
        if (_currentGroup == null)
            throw new InvalidOperationException("No active group. Call NewGroup() first.");

        _currentGroup.Momentum = momentum;
        return this;
    }

    /// <summary>
    /// Builds and returns the list of parameter groups.
    /// </summary>
    /// <returns>List of parameter groups.</returns>
    public List<ParameterGroup> Build()
    {
        // Add the current group if it exists
        if (_currentGroup != null)
        {
            _groups.Add(_currentGroup);
            _currentGroup = null;
        }

        return _groups;
    }

    /// <summary>
    /// Builds the parameter groups and returns them as a dictionary.
    /// </summary>
    /// <returns>Dictionary mapping group names to parameter groups.</returns>
    public Dictionary<string, ParameterGroup> BuildDictionary()
    {
        var groups = Build();
        var dict = new Dictionary<string, ParameterGroup>();

        foreach (var group in groups)
        {
            dict[group.Name] = group;
        }

        return dict;
    }

    /// <summary>
    /// Resets the builder, clearing all groups.
    /// </summary>
    /// <returns>This builder for method chaining.</returns>
    public ParameterGroupBuilder Reset()
    {
        _groups.Clear();
        _currentGroup = null;
        return this;
    }
}
