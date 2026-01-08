using System;
using System.Collections.Generic;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using MLFramework.Schedulers;

namespace MLFramework.Optimizers.SecondOrder;

/// <summary>
/// Hessian-aware Adam optimizer that adapts the learning rate based on the
/// curvature of the loss landscape, as estimated by the maximum eigenvalue of the Hessian.
/// This helps improve stability and convergence on problems with varying curvature.
/// </summary>
public class HessianAwareAdam : Adam
{
    private int _eigenvalueEstimationIterations;
    private bool _usePerParameterAdaptation;
    private float _minLearningRate;
    private float _maxLearningRate;

    /// <summary>
    /// Gets or sets the number of power iteration steps for eigenvalue estimation (default: 10).
    /// </summary>
    public int EigenvalueEstimationIterations
    {
        get => _eigenvalueEstimationIterations;
        set => _eigenvalueEstimationIterations = value;
    }

    /// <summary>
    /// Gets or sets whether to adapt learning rate per-parameter (default: false).
    /// When true, estimates eigenvalue per-parameter, which is more accurate but more expensive.
    /// </summary>
    public bool UsePerParameterAdaptation
    {
        get => _usePerParameterAdaptation;
        set => _usePerParameterAdaptation = value;
    }

    /// <summary>
    /// Gets or sets the minimum allowed learning rate after adaptation (default: 1e-6).
    /// </summary>
    public float MinLearningRate
    {
        get => _minLearningRate;
        set => _minLearningRate = value;
    }

    /// <summary>
    /// Gets or sets the maximum allowed learning rate after adaptation (default: 1.0).
    /// </summary>
    public float MaxLearningRate
    {
        get => _maxLearningRate;
        set => _maxLearningRate = value;
    }

    /// <summary>
    /// Initializes a new instance of the HessianAwareAdam optimizer.
    /// </summary>
    /// <param name="parameters">Dictionary mapping parameter names to tensors.</param>
    /// <param name="learningRate">Base learning rate (default: 1e-3).</param>
    /// <param name="beta1">Exponential decay rate for first moment (default: 0.9).</param>
    /// <param name="beta2">Exponential decay rate for second moment (default: 0.999).</param>
    /// <param name="epsilon">Small constant for numerical stability (default: 1e-8).</param>
    /// <param name="weightDecay">Weight decay coefficient (default: 0).</param>
    /// <param name="amsgrad">Whether to use AMSGrad variant (default: false).</param>
    /// <param name="eigenvalueEstimationIterations">Iterations for eigenvalue estimation (default: 10).</param>
    /// <param name="usePerParameterAdaptation">Whether to adapt per-parameter (default: false).</param>
    public HessianAwareAdam(
        Dictionary<string, Tensor> parameters,
        float learningRate = 1e-3f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float epsilon = 1e-8f,
        float weightDecay = 0.0f,
        bool amsgrad = false,
        int eigenvalueEstimationIterations = 10,
        bool usePerParameterAdaptation = false)
        : base(parameters, learningRate, beta1, beta2, epsilon, weightDecay, amsgrad)
    {
        _eigenvalueEstimationIterations = eigenvalueEstimationIterations;
        _usePerParameterAdaptation = usePerParameterAdaptation;
        _minLearningRate = 1e-6f;
        _maxLearningRate = 1.0f;
    }

    /// <inheritdoc/>
    public override void Step(Dictionary<string, Tensor> gradients)
    {
        // Estimate maximum eigenvalue of Hessian
        float maxEigenvalue = EstimateMaxEigenvalue(gradients);

        // Adapt learning rate based on curvature
        float adaptedLearningRate = AdaptLearningRate(maxEigenvalue);

        // Temporarily override learning rate
        float originalLearningRate = LearningRate;
        SetLearningRate(adaptedLearningRate);

        // Perform standard Adam step
        base.Step(gradients);

        // Restore original learning rate
        SetLearningRate(originalLearningRate);
    }

    /// <summary>
    /// Estimates the maximum eigenvalue of the Hessian using power iteration.
    /// </summary>
    /// <param name="gradients">Dictionary mapping parameter names to gradient tensors.</param>
    /// <returns>Estimated maximum eigenvalue.</returns>
    private float EstimateMaxEigenvalue(Dictionary<string, Tensor> gradients)
    {
        if (_usePerParameterAdaptation)
        {
            // Estimate eigenvalue per parameter and take maximum
            float maxEigenvalue = 0.0f;
            foreach (var kvp in Parameters)
            {
                var paramName = kvp.Key;
                var param = kvp.Value;

                if (!gradients.ContainsKey(paramName))
                    continue;

                var gradient = gradients[paramName];

                // Create a dummy loss tensor for Hessian computation
                var loss = new Tensor(new[] { 0.0f }, new[] { 1 });

                float paramEigenvalue = PowerIteration.EstimateMaxEigenvalue(
                    matrixVectorProduct: v => ComputeHVP(loss, new[] { param }, v),
                    dimension: param.Size,
                    numIterations: _eigenvalueEstimationIterations
                );

                maxEigenvalue = Math.Max(maxEigenvalue, paramEigenvalue);
            }

            return maxEigenvalue;
        }
        else
        {
            // Estimate global eigenvalue using all parameters
            var parameters = GetParameterArray();
            var gradientArray = GetGradientArray(gradients);
            var flattenedParams = FlattenParameters(parameters);
            var flattenedGradients = FlattenParameters(gradientArray);

            // Create a dummy loss tensor for Hessian computation
            var loss = new Tensor(new[] { 0.0f }, new[] { 1 });

            return PowerIteration.EstimateMaxEigenvalue(
                matrixVectorProduct: v => ComputeHVP(loss, parameters, v),
                dimension: flattenedParams.Size,
                numIterations: _eigenvalueEstimationIterations
            );
        }
    }

    /// <summary>
    /// Computes the Hessian-Vector Product.
    /// </summary>
    private Tensor ComputeHVP(Tensor loss, Tensor[] parameters, Tensor vector)
    {
        return Hessian.ComputeVectorHessianProduct(
            f: t => loss.Data[0],
            x: FlattenParameters(parameters),
            v: vector
        );
    }

    /// <summary>
    /// Adapts the learning rate based on the estimated maximum eigenvalue.
    /// Uses the formula: lr_adapted = lr_base / (1 + max_eigenvalue)
    /// This automatically reduces learning rate in regions of high curvature.
    /// </summary>
    /// <param name="maxEigenvalue">Estimated maximum eigenvalue.</param>
    /// <returns>Adapted learning rate.</returns>
    private float AdaptLearningRate(float maxEigenvalue)
    {
        float baseLearningRate = LearningRate;

        // Adapt based on curvature: higher eigenvalue -> smaller learning rate
        float adaptedLearningRate = baseLearningRate / (1.0f + maxEigenvalue);

        // Clamp to [min, max] range
        adaptedLearningRate = Math.Max(_minLearningRate, adaptedLearningRate);
        adaptedLearningRate = Math.Min(_maxLearningRate, adaptedLearningRate);

        return adaptedLearningRate;
    }

    /// <summary>
    /// Gets an array of parameters in the order expected by the optimizer.
    /// </summary>
    private Tensor[] GetParameterArray()
    {
        var result = new Tensor[Parameters.Count];
        int index = 0;
        foreach (var kvp in Parameters)
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
        var result = new Tensor[Parameters.Count];
        int index = 0;
        foreach (var paramName in Parameters.Keys)
        {
            if (gradients.ContainsKey(paramName))
            {
                result[index++] = gradients[paramName];
            }
            else
            {
                var param = Parameters[paramName];
                result[index++] = new Tensor(new float[param.Size], param.Shape);
            }
        }
        return result;
    }

    /// <summary>
    /// Flattens a set of parameters into a single 1D tensor.
    /// </summary>
    private Tensor FlattenParameters(Tensor[] parameters)
    {
        int totalSize = 0;
        foreach (var param in parameters)
        {
            totalSize += param.Size;
        }

        var flattened = new float[totalSize];
        int offset = 0;

        foreach (var param in parameters)
        {
            var paramData = TensorAccessor.GetData(param);
            Array.Copy(paramData, 0, flattened, offset, param.Size);
            offset += param.Size;
        }

        return new Tensor(flattened, new[] { totalSize });
    }

    /// <inheritdoc/>
    public override StateDict GetState()
    {
        var state = base.GetState();
        state.Set("eigenvalue_estimation_iterations", _eigenvalueEstimationIterations);
        state.Set("use_per_parameter_adaptation", _usePerParameterAdaptation);
        state.Set("min_learning_rate", _minLearningRate);
        state.Set("max_learning_rate", _maxLearningRate);
        return state;
    }

    /// <inheritdoc/>
    public override void LoadState(StateDict state)
    {
        base.LoadState(state);
        _eigenvalueEstimationIterations = state.Get<int>("eigenvalue_estimation_iterations", 10);
        _usePerParameterAdaptation = state.Get<bool>("use_per_parameter_adaptation", false);
        _minLearningRate = state.Get<float>("min_learning_rate", 1e-6f);
        _maxLearningRate = state.Get<float>("max_learning_rate", 1.0f);
    }
}
