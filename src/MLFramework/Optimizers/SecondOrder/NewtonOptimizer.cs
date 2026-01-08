using System;
using System.Collections.Generic;
using RitterFramework.Core.Tensor;
using MLFramework.Schedulers;
using MLFramework.Autograd;

namespace MLFramework.Optimizers.SecondOrder;

/// <summary>
/// Newton's method optimizer that uses second-order (Hessian) information for optimization.
/// Solves (H + λI) * Δp = -g where H is the Hessian, g is the gradient, and λ is damping.
/// Uses conjugate gradient to solve the linear system without materializing the Hessian.
/// </summary>
public class NewtonOptimizer : SecondOrderOptimizer
{
    private float _learningRate;
    private float _damping;

    /// <summary>
    /// Gets or sets the learning rate for Newton's method.
    /// Typically a smaller value (e.g., 0.1 - 1.0) for stability.
    /// </summary>
    public float LearningRate
    {
        get => _learningRate;
        set => _learningRate = value;
    }

    /// <summary>
    /// Gets or sets the damping factor to avoid singular Hessian.
    /// Adds λI to the Hessian for numerical stability.
    /// </summary>
    public float Damping
    {
        get => _damping;
        set => _damping = value;
    }

    /// <summary>
    /// Gets or sets the maximum number of iterations for conjugate gradient solver.
    /// </summary>
    public int MaxCGIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the tolerance for conjugate gradient solver.
    /// </summary>
    public float CGTolerance { get; set; } = 1e-10f;

    /// <summary>
    /// Initializes a new instance of the NewtonOptimizer class.
    /// </summary>
    /// <param name="parameters">Dictionary mapping parameter names to tensors.</param>
    /// <param name="learningRate">Learning rate (default: 1.0).</param>
    /// <param name="damping">Damping factor for numerical stability (default: 1e-4).</param>
    public NewtonOptimizer(
        Dictionary<string, Tensor> parameters,
        float learningRate = 1.0f,
        float damping = 1e-4f)
        : base(parameters)
    {
        _learningRate = learningRate;
        _damping = damping;
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

        // Create a dummy loss tensor for Hessian computation
        // In practice, this would be computed during the forward pass
        var loss = new Tensor(new[] { 0.0f }, new[] { 1 });

        // Compute Newton step using second-order information
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
        // Flatten gradients to solve the linear system
        var flattenedGradients = FlattenParameters(gradients);

        // Solve (H + λI) * x = -g using conjugate gradient
        var solution = ConjugateGradientSolver.Solve(
            b: flattenedGradients,
            matrixVectorProduct: v =>
            {
                // Compute HVP
                var hvp = ComputeHVP(loss, parameters, v);

                // Add damping: (H + λI)v = Hv + λv
                var vData = TensorAccessor.GetData(v);
                var hvpData = TensorAccessor.GetData(hvp);
                var resultData = new float[vData.Length];

                for (int i = 0; i < vData.Length; i++)
                {
                    resultData[i] = hvpData[i] + Damping * vData[i];
                }

                return new Tensor(resultData, v.Shape);
            },
            maxIterations: MaxCGIterations,
            tolerance: CGTolerance
        );

        // Negate and scale by learning rate: Δp = -α * H^(-1)g
        var solutionData = TensorAccessor.GetData(solution);
        for (int i = 0; i < solutionData.Length; i++)
        {
            solutionData[i] = -LearningRate * solutionData[i];
        }

        // Unflatten to match original parameter shapes
        return UnflattenParameters(solution, parameters);
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
        state.Set("damping", _damping);
        state.Set("max_cg_iterations", MaxCGIterations);
        state.Set("cg_tolerance", CGTolerance);
        return state;
    }

    /// <inheritdoc/>
    public override void LoadState(StateDict state)
    {
        base.LoadState(state);
        _learningRate = state.Get<float>("learning_rate", 1.0f);
        _damping = state.Get<float>("damping", 1e-4f);
        MaxCGIterations = state.Get<int>("max_cg_iterations", 100);
        CGTolerance = state.Get<float>("cg_tolerance", 1e-10f);
    }
}
