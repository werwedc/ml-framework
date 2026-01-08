using System;
using System.Collections.Generic;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using MLFramework.Schedulers;

namespace MLFramework.Optimizers.SecondOrder;

/// <summary>
/// Natural Gradient optimizer that uses the Fisher information matrix
/// instead of the standard Hessian for computing update directions.
/// Natural gradient: Δp = -α * F^(-1) * ∇L
/// where F is the Fisher information matrix.
/// </summary>
public class NaturalGradientOptimizer : SecondOrderOptimizer
{
    private float _learningRate;
    private int _maxCGIterations;
    private float _cgTolerance;

    /// <summary>
    /// Gets or sets the learning rate for natural gradient descent.
    /// </summary>
    public new float LearningRate
    {
        get => _learningRate;
        set => _learningRate = value;
    }

    /// <summary>
    /// Gets or sets the maximum number of iterations for conjugate gradient solver.
    /// </summary>
    public int MaxCGIterations
    {
        get => _maxCGIterations;
        set => _maxCGIterations = value;
    }

    /// <summary>
    /// Gets or sets the tolerance for conjugate gradient solver.
    /// </summary>
    public float CGTolerance
    {
        get => _cgTolerance;
        set => _cgTolerance = value;
    }

    /// <summary>
    /// Gets or sets whether to use empirical Fisher approximation.
    /// When true, uses outer product of gradients: F ≈ ∇L * ∇L^T
    /// When false, uses expected Fisher (requires distribution).
    /// </summary>
    public bool UseEmpiricalFisher { get; set; } = true;

    /// <summary>
    /// Initializes a new instance of the NaturalGradientOptimizer class.
    /// </summary>
    /// <param name="parameters">Dictionary mapping parameter names to tensors.</param>
    /// <param name="learningRate">Learning rate (default: 1e-3).</param>
    public NaturalGradientOptimizer(
        Dictionary<string, Tensor> parameters,
        float learningRate = 1e-3f)
        : base(parameters)
    {
        _learningRate = learningRate;
        _maxCGIterations = 100;
        _cgTolerance = 1e-10f;
    }

    /// <inheritdoc/>
    public override float BaseLearningRate => _learningRate;

    /// <inheritdoc/>
    public override void SetLearningRate(float lr)
    {
        _learningRate = lr;
    }

    /// <inheritdoc/>
    public override void Step(Dictionary<string, Tensor> gradients)
    {
        UpdateLearningRate();

        var parameters = GetParameterArray();
        var gradientArray = GetGradientArray(gradients);

        // Create a dummy loss tensor for Hessian/Fisher computation
        var loss = new Tensor(new[] { 0.0f }, new[] { 1 });

        // Compute natural gradient update
        var updateStep = ComputeUpdateStep(loss, gradientArray, parameters);

        // Apply update step
        for (int i = 0; i < parameters.Length; i++)
        {
            var paramData = TensorAccessor.GetData(parameters[i]);
            var updateData = TensorAccessor.GetData(updateStep[i]);

            for (int j = 0; j < paramData.Length; j++)
            {
                paramData[j] += updateData[j];
            }
        }

        _stepCount++;
        StepScheduler();
    }

    /// <inheritdoc/>
    public override void StepParameter(string parameterName, Tensor gradient)
    {
        if (!_parameters.ContainsKey(parameterName))
        {
            throw new ArgumentException($"Parameter {parameterName} not found in optimizer.");
        }

        var param = _parameters[parameterName];
        var gradients = new Dictionary<string, Tensor> { { parameterName, gradient } };
        Step(gradients);
    }

    /// <inheritdoc/>
    public override void ZeroGrad()
    {
        foreach (var param in _parameters.Values)
        {
            if (param.Gradient != null)
            {
                var gradData = TensorAccessor.GetData(param.Gradient);
                for (int i = 0; i < gradData.Length; i++)
                {
                    gradData[i] = 0.0f;
                }
            }
        }
    }

    /// <inheritdoc/>
    protected override Tensor[] ComputeUpdateStep(
        Tensor loss,
        Tensor[] gradients,
        Tensor[] parameters)
    {
        // Flatten gradients
        var flattenedGradients = FlattenParameters(gradients);

        // Solve F * x = g where F is Fisher information matrix
        var fisherInverseGradient = SolveFisherSystem(loss, parameters, flattenedGradients);

        // Negate and scale by learning rate: Δp = -α * F^(-1) * ∇L
        var solutionData = TensorAccessor.GetData(fisherInverseGradient);
        for (int i = 0; i < solutionData.Length; i++)
        {
            solutionData[i] = -LearningRate * solutionData[i];
        }

        // Unflatten to match original parameter shapes
        return UnflattenParameters(fisherInverseGradient, parameters);
    }

    /// <summary>
    /// Solves the linear system F * x = g using conjugate gradient,
    /// where F is the Fisher information matrix.
    /// </summary>
    private Tensor SolveFisherSystem(Tensor loss, Tensor[] parameters, Tensor gradients)
    {
        if (UseEmpiricalFisher)
        {
            // Use empirical Fisher: F ≈ ∇L * ∇L^T
            // Then F * v = (∇L * ∇L^T) * v = ∇L * (∇L^T * v)
            return ConjugateGradientSolver.Solve(
                b: gradients,
                matrixVectorProduct: v =>
                {
                    // Compute (∇L * ∇L^T) * v = ∇L * (∇L^T * v)
                    var gradData = TensorAccessor.GetData(gradients);
                    var vData = TensorAccessor.GetData(v);

                    // Compute ∇L^T * v (scalar)
                    float gradDotV = 0.0f;
                    for (int i = 0; i < gradData.Length; i++)
                    {
                        gradDotV += gradData[i] * vData[i];
                    }

                    // Compute ∇L * (∇L^T * v)
                    var resultData = new float[gradData.Length];
                    for (int i = 0; i < gradData.Length; i++)
                    {
                        resultData[i] = gradData[i] * gradDotV;
                    }

                    return new Tensor(resultData, v.Shape);
                },
                maxIterations: MaxCGIterations,
                tolerance: CGTolerance
            );
        }
        else
        {
            // Use expected Fisher (approximated as Hessian for expected loss)
            // This requires computing the Hessian, which is more expensive
            return ConjugateGradientSolver.Solve(
                b: gradients,
                matrixVectorProduct: v => ComputeHVP(loss, parameters, v),
                maxIterations: MaxCGIterations,
                tolerance: CGTolerance
            );
        }
    }

    /// <summary>
    /// Gets an array of parameters in the order expected by the optimizer.
    /// </summary>
    private Tensor[] GetParameterArray()
    {
        var result = new Tensor[_parameters.Count];
        int index = 0;

        foreach (var kvp in _parameters)
        {
            result[index++] = kvp.Value;
        }

        return result;
    }

    /// <summary>
    /// Gets an array of gradients in the same order as parameters.
    /// </summary>
    private Tensor[] GetGradientArray(Dictionary<string, Tensor> gradients)
    {
        var result = new Tensor[_parameters.Count];
        int index = 0;

        foreach (var paramName in _parameters.Keys)
        {
            if (gradients.ContainsKey(paramName))
            {
                result[index++] = gradients[paramName];
            }
            else
            {
                // Initialize zero gradient if not provided
                var param = _parameters[paramName];
                result[index++] = new Tensor(new float[param.Size], param.Shape);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public override StateDict GetState()
    {
        var state = base.GetState();
        state.Set("learning_rate", _learningRate);
        state.Set("max_cg_iterations", _maxCGIterations);
        state.Set("cg_tolerance", _cgTolerance);
        state.Set("use_empirical_fisher", UseEmpiricalFisher);
        return state;
    }

    /// <inheritdoc/>
    public override void LoadState(StateDict state)
    {
        base.LoadState(state);
        _learningRate = state.Get<float>("learning_rate", 1e-3f);
        _maxCGIterations = state.Get<int>("max_cg_iterations", 100);
        _cgTolerance = state.Get<float>("cg_tolerance", 1e-10f);
        UseEmpiricalFisher = state.Get<bool>("use_empirical_fisher", true);
    }
}
