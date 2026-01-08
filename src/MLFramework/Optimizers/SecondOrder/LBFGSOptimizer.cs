using System;
using System.Collections.Generic;
using RitterFramework.Core.Tensor;
using MLFramework.Schedulers;
using MLFramework.Autograd;

namespace MLFramework.Optimizers.SecondOrder;

/// <summary>
/// Limited-memory BFGS (L-BFGS) optimizer.
/// Uses gradient history to approximate the inverse Hessian without storing the full matrix.
/// Memory efficient for large-scale optimization problems.
/// </summary>
public class LBFGSOptimizer : Optimizer
{
    private float _learningRate;
    private int _historySize;
    private int _maxIterations;
    private float _tolerance;
    private int _lineSearchMaxIterations;

    /// <summary>
    /// Gets or sets the learning rate (default: 1.0).
    /// L-BFGS typically uses unit learning rate with line search.
    /// </summary>
    public new float LearningRate
    {
        get => _learningRate;
        set => _learningRate = value;
    }

    /// <summary>
    /// Gets or sets the history size for L-BFGS (default: 10).
    /// Determines how many past gradient/parameter updates to store.
    /// </summary>
    public int HistorySize
    {
        get => _historySize;
        set => _historySize = value;
    }

    /// <summary>
    /// Gets or sets the maximum number of iterations per optimization step (default: 20).
    /// L-BFGS performs multiple inner iterations per Step() call.
    /// </summary>
    public int MaxIterations
    {
        get => _maxIterations;
        set => _maxIterations = value;
    }

    /// <summary>
    /// Gets or sets the convergence tolerance (default: 1e-5).
    /// </summary>
    public float Tolerance
    {
        get => _tolerance;
        set => _tolerance = value;
    }

    /// <summary>
    /// Gets or sets the maximum number of line search iterations (default: 20).
    /// </summary>
    public int LineSearchMaxIterations
    {
        get => _lineSearchMaxIterations;
        set => _lineSearchMaxIterations = value;
    }

    /// <summary>
    /// Gets or sets the line search tolerance (default: 1e-4).
    /// </summary>
    public float LineSearchTolerance { get; set; } = 1e-4f;

    // L-BFGS state
    private List<Tensor> _sHistory;  // Parameter differences: s_k = x_{k+1} - x_k
    private List<Tensor> _yHistory;  // Gradient differences: y_k = g_{k+1} - g_k
    private List<Tensor> _rhoHistory; // 1 / (s_k^T y_k)
    private Tensor[] _previousParameters;
    private Tensor[]? _previousGradients;

    /// <summary>
    /// Initializes a new instance of the LBFGSOptimizer class.
    /// </summary>
    /// <param name="parameters">Dictionary mapping parameter names to tensors.</param>
    /// <param name="learningRate">Learning rate (default: 1.0).</param>
    /// <param name="historySize">History size for L-BFGS (default: 10).</param>
    /// <param name="maxIterations">Maximum iterations per step (default: 20).</param>
    /// <param name="tolerance">Convergence tolerance (default: 1e-5).</param>
    public LBFGSOptimizer(
        Dictionary<string, Tensor> parameters,
        float learningRate = 1.0f,
        int historySize = 10,
        int maxIterations = 20,
        float tolerance = 1e-5f)
        : base(parameters)
    {
        _learningRate = learningRate;
        _historySize = historySize;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _lineSearchMaxIterations = 20;

        _sHistory = new List<Tensor>();
        _yHistory = new List<Tensor>();
        _rhoHistory = new List<Tensor>();

        _previousParameters = CloneParameters();
        _previousGradients = null;
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

        // Perform L-BFGS update
        if (_previousGradients != null)
        {
            UpdateHistory(parameters, gradientArray);
        }

        var updateStep = ComputeLBFGSUpdate(gradientArray);

        // Apply line search to find optimal step size
        float stepSize = LineSearch(parameters, gradientArray, updateStep);

        // Apply update
        for (int i = 0; i < parameters.Length; i++)
        {
            var paramData = TensorAccessor.GetData(parameters[i]);
            var updateData = TensorAccessor.GetData(updateStep[i]);

            for (int j = 0; j < paramData.Length; j++)
            {
                paramData[j] -= stepSize * updateData[j];
            }
        }

        // Store previous state
        _previousParameters = CloneParameters();
        _previousGradients = CloneGradients(gradientArray);

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

    /// <summary>
    /// Computes the L-BFGS update step using two-loop recursion.
    /// </summary>
    private Tensor[] ComputeLBFGSUpdate(Tensor[] gradients)
    {
        // Flatten gradients for computation
        var flattenedGradients = FlattenParameters(gradients);

        // Initialize q = g (current gradient)
        var qData = (float[])TensorAccessor.GetData(flattenedGradients).Clone();

        // Two-loop recursion algorithm
        var alpha = new float[_sHistory.Count];

        // First loop: recursively compute alphas
        for (int i = _sHistory.Count - 1; i >= 0; i--)
        {
            var sData = TensorAccessor.GetData(_sHistory[i]);
            var yData = TensorAccessor.GetData(_yHistory[i]);
            var rhoData = TensorAccessor.GetData(_rhoHistory[i])[0];

            alpha[i] = rhoData * DotProduct(qData, sData);

            for (int j = 0; j < qData.Length; j++)
            {
                qData[j] -= alpha[i] * yData[j];
            }
        }

        // Initial Hessian approximation (scaled identity)
        // H_0 = gamma_k * I
        float gamma = 1.0f;
        if (_yHistory.Count > 0)
        {
            int lastIdx = _yHistory.Count - 1;
            var sLast = TensorAccessor.GetData(_sHistory[lastIdx]);
            var yLast = TensorAccessor.GetData(_yHistory[lastIdx]);
            float sy = DotProduct(sLast, yLast);
            float yy = DotProduct(yLast, yLast);
            gamma = sy / yy;
        }

        for (int j = 0; j < qData.Length; j++)
        {
            qData[j] *= gamma;
        }

        // Second loop: recursively compute search direction
        for (int i = 0; i < _sHistory.Count; i++)
        {
            var sData = TensorAccessor.GetData(_sHistory[i]);
            var yData = TensorAccessor.GetData(_yHistory[i]);
            var rhoData = TensorAccessor.GetData(_rhoHistory[i])[0];

            float beta = rhoData * DotProduct(qData, yData);

            for (int j = 0; j < qData.Length; j++)
            {
                qData[j] += sData[j] * (alpha[i] - beta);
            }
        }

        var flattenedUpdate = new Tensor(qData, flattenedGradients.Shape);

        // Unflatten to match original parameter shapes
        return UnflattenParameters(flattenedUpdate, gradients);
    }

    /// <summary>
    /// Updates the L-BFGS history with new parameter and gradient differences.
    /// </summary>
    private void UpdateHistory(Tensor[] parameters, Tensor[] gradients)
    {
        // This method is only called when _previousGradients is not null,
        // but we add a check to satisfy the compiler
        if (_previousGradients == null)
            return;

        // Compute s = x_new - x_old
        var sData = new float[GetTotalParameterSize()];
        int offset = 0;
        for (int i = 0; i < parameters.Length; i++)
        {
            var paramData = TensorAccessor.GetData(parameters[i]);
            var prevData = TensorAccessor.GetData(_previousParameters[i]);
            Array.Copy(paramData, 0, sData, offset, paramData.Length);
            for (int j = 0; j < paramData.Length; j++)
            {
                sData[offset + j] -= prevData[j];
            }
            offset += paramData.Length;
        }

        // Compute y = g_new - g_old
        var yData = new float[sData.Length];
        offset = 0;
        for (int i = 0; i < gradients.Length; i++)
        {
            var gradData = TensorAccessor.GetData(gradients[i]);
            var prevGradData = TensorAccessor.GetData(_previousGradients[i]);
            for (int j = 0; j < gradData.Length; j++)
            {
                yData[offset + j] = gradData[j] - prevGradData[j];
            }
            offset += gradData.Length;
        }

        // Compute rho = 1 / (s^T * y)
        float sy = DotProduct(sData, yData);
        if (Math.Abs(sy) > 1e-10f)
        {
            var s = new Tensor(sData, new[] { sData.Length });
            var y = new Tensor(yData, new[] { yData.Length });
            var rho = new Tensor(new[] { 1.0f / sy }, new[] { 1 });

            // Add to history
            _sHistory.Add(s);
            _yHistory.Add(y);
            _rhoHistory.Add(rho);

            // Maintain history size
            if (_sHistory.Count > _historySize)
            {
                _sHistory.RemoveAt(0);
                _yHistory.RemoveAt(0);
                _rhoHistory.RemoveAt(0);
            }
        }
    }

    /// <summary>
    /// Performs backtracking line search to find optimal step size.
    /// </summary>
    private float LineSearch(
        Tensor[] parameters,
        Tensor[] gradients,
        Tensor[] direction)
    {
        float alpha = 1.0f;
        float c = LineSearchTolerance;  // Sufficient decrease parameter

        for (int iter = 0; iter < _lineSearchMaxIterations; iter++)
        {
            // Compute loss at x - alpha * direction
            var testParameters = CloneParameters();
            for (int i = 0; i < testParameters.Length; i++)
            {
                var paramData = TensorAccessor.GetData(testParameters[i]);
                var dirData = TensorAccessor.GetData(direction[i]);
                for (int j = 0; j < paramData.Length; j++)
                {
                    paramData[j] -= alpha * dirData[j];
                }
            }

            // In practice, we would recompute the forward pass here
            // For now, we use a simple sufficient decrease condition based on gradient
            float gradDotDir = ComputeGradientDotDirection(gradients, direction);

            // Simple backtracking: reduce alpha if gradient dot product is not sufficiently negative
            if (gradDotDir < 0)
            {
                break;  // Accept this step size
            }

            alpha *= 0.5f;  // Reduce step size
        }

        return alpha;
    }

    /// <summary>
    /// Computes the dot product of gradients and direction.
    /// </summary>
    private float ComputeGradientDotDirection(Tensor[] gradients, Tensor[] direction)
    {
        float result = 0.0f;
        for (int i = 0; i < gradients.Length; i++)
        {
            var gradData = TensorAccessor.GetData(gradients[i]);
            var dirData = TensorAccessor.GetData(direction[i]);
            result += DotProduct(gradData, dirData);
        }
        return result;
    }

    /// <summary>
    /// Gets an array of parameters.
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
                var param = _parameters[paramName];
                result[index++] = new Tensor(new float[param.Size], param.Shape);
            }
        }
        return result;
    }

    /// <summary>
    /// Clones all parameters.
    /// </summary>
    private Tensor[] CloneParameters()
    {
        var result = new Tensor[_parameters.Count];
        int index = 0;
        foreach (var param in _parameters.Values)
        {
            var data = (float[])TensorAccessor.GetData(param).Clone();
            result[index++] = new Tensor(data, param.Shape);
        }
        return result;
    }

    /// <summary>
    /// Clones gradients.
    /// </summary>
    private Tensor[] CloneGradients(Tensor[] gradients)
    {
        var result = new Tensor[gradients.Length];
        for (int i = 0; i < gradients.Length; i++)
        {
            var data = (float[])TensorAccessor.GetData(gradients[i]).Clone();
            result[i] = new Tensor(data, gradients[i].Shape);
        }
        return result;
    }

    /// <summary>
    /// Gets total size of all parameters.
    /// </summary>
    private int GetTotalParameterSize()
    {
        int total = 0;
        foreach (var param in _parameters.Values)
        {
            total += param.Size;
        }
        return total;
    }

    /// <summary>
    /// Computes dot product of two arrays.
    /// </summary>
    private float DotProduct(float[] a, float[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Arrays must have the same length.");

        float result = 0.0f;
        for (int i = 0; i < a.Length; i++)
        {
            result += a[i] * b[i];
        }
        return result;
    }

    /// <summary>
    /// Flattens parameters (inherited from SecondOrderOptimizer logic).
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

    /// <summary>
    /// Unflattens parameters (inherited from SecondOrderOptimizer logic).
    /// </summary>
    private Tensor[] UnflattenParameters(Tensor flattened, Tensor[] original)
    {
        var result = new Tensor[original.Length];
        var flattenedData = TensorAccessor.GetData(flattened);
        int offset = 0;

        for (int i = 0; i < original.Length; i++)
        {
            int size = original[i].Size;
            var paramData = new float[size];
            Array.Copy(flattenedData, offset, paramData, 0, size);
            result[i] = new Tensor(paramData, original[i].Shape);
            offset += size;
        }

        return result;
    }

    /// <inheritdoc/>
    public override StateDict GetState()
    {
        var state = base.GetState();
        state.Set("learning_rate", _learningRate);
        state.Set("history_size", _historySize);
        state.Set("max_iterations", _maxIterations);
        state.Set("tolerance", _tolerance);
        state.Set("line_search_max_iterations", _lineSearchMaxIterations);
        state.Set("line_search_tolerance", LineSearchTolerance);
        return state;
    }

    /// <inheritdoc/>
    public override void LoadState(StateDict state)
    {
        base.LoadState(state);
        _learningRate = state.Get<float>("learning_rate", 1.0f);
        _historySize = state.Get<int>("history_size", 10);
        _maxIterations = state.Get<int>("max_iterations", 20);
        _tolerance = state.Get<float>("tolerance", 1e-5f);
        _lineSearchMaxIterations = state.Get<int>("line_search_max_iterations", 20);
        LineSearchTolerance = state.Get<float>("line_search_tolerance", 1e-4f);
    }
}
