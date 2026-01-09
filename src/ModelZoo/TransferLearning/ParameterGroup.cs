using System.Collections.Generic;
using RitterFramework.Core.Tensor;

namespace MLFramework.ModelZoo.TransferLearning;

/// <summary>
/// Represents a group of parameters with shared optimizer settings.
/// Used for layer-wise learning rate scheduling during fine-tuning.
/// </summary>
public class ParameterGroup
{
    private readonly List<Tensor> _parameters;
    private readonly List<string> _parameterNames;
    private readonly string _name;
    private float _learningRate;
    private float _weightDecay;
    private float? _momentum;

    /// <summary>
    /// Gets the list of parameters in this group.
    /// </summary>
    public IReadOnlyList<Tensor> Parameters => _parameters.AsReadOnly();

    /// <summary>
    /// Gets the list of parameter names in this group.
    /// </summary>
    public IReadOnlyList<string> ParameterNames => _parameterNames.AsReadOnly();

    /// <summary>
    /// Gets or sets the learning rate for this group.
    /// </summary>
    public float LearningRate
    {
        get => _learningRate;
        set => _learningRate = value;
    }

    /// <summary>
    /// Gets or sets the weight decay for this group.
    /// </summary>
    public float WeightDecay
    {
        get => _weightDecay;
        set => _weightDecay = value;
    }

    /// <summary>
    /// Gets or sets the momentum value for this group (if applicable).
    /// </summary>
    public float? Momentum
    {
        get => _momentum;
        set => _momentum = value;
    }

    /// <summary>
    /// Gets the name/identifier of this group for debugging.
    /// </summary>
    public string Name => _name;

    /// <summary>
    /// Gets the total number of parameters in this group.
    /// </summary>
    public int ParameterCount => _parameters.Count;

    /// <summary>
    /// Gets the total number of elements across all parameters in this group.
    /// </summary>
    public int TotalElements
    {
        get
        {
            int total = 0;
            foreach (var param in _parameters)
            {
                total += param.Size;
            }
            return total;
        }
    }

    /// <summary>
    /// Initializes a new instance of the ParameterGroup class.
    /// </summary>
    /// <param name="name">Name/identifier for this group.</param>
    /// <param name="learningRate">Learning rate for this group.</param>
    /// <param name="weightDecay">Weight decay for this group (default: 0).</param>
    /// <param name="momentum">Momentum value for this group (optional).</param>
    public ParameterGroup(string name, float learningRate, float weightDecay = 0.0f, float? momentum = null)
    {
        if (string.IsNullOrEmpty(name))
            throw new System.ArgumentException("Group name cannot be null or empty.", nameof(name));

        if (learningRate <= 0)
            throw new System.ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");

        _name = name;
        _learningRate = learningRate;
        _weightDecay = weightDecay;
        _momentum = momentum;
        _parameters = new List<Tensor>();
        _parameterNames = new List<string>();
    }

    /// <summary>
    /// Adds a parameter to this group.
    /// </summary>
    /// <param name="parameter">The parameter to add.</param>
    /// <param name="parameterName">The name of the parameter.</param>
    public void AddParameter(Tensor parameter, string parameterName)
    {
        if (parameter == null)
            throw new System.ArgumentNullException(nameof(parameter));

        if (string.IsNullOrEmpty(parameterName))
            throw new System.ArgumentException("Parameter name cannot be null or empty.", nameof(parameterName));

        _parameters.Add(parameter);
        _parameterNames.Add(parameterName);
    }

    /// <summary>
    /// Adds multiple parameters to this group.
    /// </summary>
    /// <param name="parameters">Dictionary of parameter names to tensors.</param>
    public void AddParameters(System.Collections.Generic.IDictionary<string, Tensor> parameters)
    {
        if (parameters == null)
            throw new System.ArgumentNullException(nameof(parameters));

        foreach (var kvp in parameters)
        {
            AddParameter(kvp.Value, kvp.Key);
        }
    }

    /// <summary>
    /// Removes all parameters from this group.
    /// </summary>
    public void Clear()
    {
        _parameters.Clear();
        _parameterNames.Clear();
    }

    /// <summary>
    /// Creates a dictionary mapping parameter names to tensors for all parameters in this group.
    /// </summary>
    /// <returns>Dictionary of parameter names to tensors.</returns>
    public Dictionary<string, Tensor> ToParameterDictionary()
    {
        var dict = new Dictionary<string, Tensor>();
        for (int i = 0; i < _parameters.Count; i++)
        {
            dict[_parameterNames[i]] = _parameters[i];
        }
        return dict;
    }

    /// <summary>
    /// Gets a string representation of this parameter group.
    /// </summary>
    /// <returns>String representation.</returns>
    public override string ToString()
    {
        return $"ParameterGroup '{_name}': {_parameters.Count} parameters, LR={_learningRate}, " +
               $"WD={_weightDecay}" + (_momentum.HasValue ? $", Momentum={_momentum}" : "");
    }
}
