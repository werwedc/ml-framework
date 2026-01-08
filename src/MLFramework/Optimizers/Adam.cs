using System;
using System.Collections.Generic;
using RitterFramework.Core.Tensor;
using MLFramework.Schedulers;
using MLFramework.Autograd;

namespace MLFramework.Optimizers;

/// <summary>
/// Adam (Adaptive Moment Estimation) optimizer.
/// Combines ideas from RMSProp and momentum.
/// Keeps track of running averages of gradients and their squares.
/// </summary>
public class Adam : Optimizer
{
    private float _learningRate;
    private float _beta1;
    private float _beta2;
    private float _epsilon;
    private float _weightDecay;
    private bool _amsgrad;

    // Running averages
    private Dictionary<string, Tensor> _m;  // First moment (mean)
    private Dictionary<string, Tensor> _v;  // Second moment (uncentered variance)
    private Dictionary<string, Tensor> _vMax;  // For AMSGrad

    private int _t;  // Time step

    /// <summary>
    /// Gets or sets the learning rate (default: 1e-3).
    /// </summary>
    public float LearningRate
    {
        get => _learningRate;
        set => _learningRate = value;
    }

    /// <summary>
    /// Gets or sets the exponential decay rate for the first moment estimates (default: 0.9).
    /// </summary>
    public float Beta1
    {
        get => _beta1;
        set => _beta1 = value;
    }

    /// <summary>
    /// Gets or sets the exponential decay rate for the second moment estimates (default: 0.999).
    /// </summary>
    public float Beta2
    {
        get => _beta2;
        set => _beta2 = value;
    }

    /// <summary>
    /// Gets or sets the small constant for numerical stability (default: 1e-8).
    /// </summary>
    public float Epsilon
    {
        get => _epsilon;
        set => _epsilon = value;
    }

    /// <summary>
    /// Gets or sets the weight decay coefficient (L2 regularization) (default: 0).
    /// </summary>
    public float WeightDecay
    {
        get => _weightDecay;
        set => _weightDecay = value;
    }

    /// <summary>
    /// Gets or sets whether to use the AMSGrad variant (default: false).
    /// When true, keeps the maximum of all second moment running averages.
    /// </summary>
    public bool Amsgrad
    {
        get => _amsgrad;
        set => _amsgrad = value;
    }

    /// <summary>
    /// Initializes a new instance of the Adam optimizer.
    /// </summary>
    /// <param name="parameters">Dictionary mapping parameter names to tensors.</param>
    /// <param name="learningRate">Learning rate (default: 1e-3).</param>
    /// <param name="beta1">Exponential decay rate for first moment (default: 0.9).</param>
    /// <param name="beta2">Exponential decay rate for second moment (default: 0.999).</param>
    /// <param name="epsilon">Small constant for numerical stability (default: 1e-8).</param>
    /// <param name="weightDecay">Weight decay coefficient (default: 0).</param>
    /// <param name="amsgrad">Whether to use AMSGrad variant (default: false).</param>
    public Adam(
        Dictionary<string, Tensor> parameters,
        float learningRate = 1e-3f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float epsilon = 1e-8f,
        float weightDecay = 0.0f,
        bool amsgrad = false)
        : base(parameters)
    {
        _learningRate = learningRate;
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _weightDecay = weightDecay;
        _amsgrad = amsgrad;

        _t = 0;
        _m = new Dictionary<string, Tensor>();
        _v = new Dictionary<string, Tensor>();
        _vMax = new Dictionary<string, Tensor>();

        // Initialize running averages
        foreach (var kvp in _parameters)
        {
            var paramName = kvp.Key;
            var param = kvp.Value;
            _m[paramName] = new Tensor(new float[param.Size], param.Shape);
            _v[paramName] = new Tensor(new float[param.Size], param.Shape);

            if (_amsgrad)
            {
                _vMax[paramName] = new Tensor(new float[param.Size], param.Shape);
            }
        }
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
        _t++;

        float biasCorrection1 = 1.0f - (float)Math.Pow(_beta1, _t);
        float biasCorrection2 = 1.0f - (float)Math.Pow(_beta2, _t);

        foreach (var kvp in _parameters)
        {
            var paramName = kvp.Key;
            var param = kvp.Value;

            if (!gradients.ContainsKey(paramName))
            {
                continue;
            }

            var gradient = gradients[paramName];
            var paramData = TensorAccessor.GetData(param);
            var gradData = TensorAccessor.GetData(gradient);
            var mData = TensorAccessor.GetData(_m[paramName]);
            var vData = TensorAccessor.GetData(_v[paramName]);

            // Update biased first moment estimate
            for (int i = 0; i < param.Size; i++)
            {
                mData[i] = _beta1 * mData[i] + (1.0f - _beta1) * gradData[i];
            }

            // Update biased second raw moment estimate
            for (int i = 0; i < param.Size; i++)
            {
                vData[i] = _beta2 * vData[i] + (1.0f - _beta2) * gradData[i] * gradData[i];
            }

            // Compute bias-corrected first moment estimate
            var mHatData = new float[param.Size];
            for (int i = 0; i < param.Size; i++)
            {
                mHatData[i] = mData[i] / biasCorrection1;
            }

            // Compute bias-corrected second raw moment estimate
            var vHatData = new float[param.Size];
            for (int i = 0; i < param.Size; i++)
            {
                if (_amsgrad)
                {
                    // Update vMax: vMax[i] = max(vMax[i], v[i])
                    var vMaxData = TensorAccessor.GetData(_vMax[paramName]);
                    vMaxData[i] = Math.Max(vMaxData[i], vData[i]);
                    vHatData[i] = vMaxData[i] / biasCorrection2;
                }
                else
                {
                    vHatData[i] = vData[i] / biasCorrection2;
                }
            }

            // Update parameters with weight decay
            float lr = GetCurrentLearningRate();
            for (int i = 0; i < param.Size; i++)
            {
                // Apply weight decay
                if (_weightDecay != 0.0f)
                {
                    paramData[i] -= lr * _weightDecay * paramData[i];
                }

                // Adam update
                paramData[i] -= lr * mHatData[i] / ((float)Math.Sqrt(vHatData[i]) + _epsilon);
            }
        }

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
    public override StateDict GetState()
    {
        var state = base.GetState();
        state.Set("learning_rate", _learningRate);
        state.Set("beta1", _beta1);
        state.Set("beta2", _beta2);
        state.Set("epsilon", _epsilon);
        state.Set("weight_decay", _weightDecay);
        state.Set("amsgrad", _amsgrad);
        state.Set("t", _t);

        // Save moment tensors
        var mState = new StateDict();
        foreach (var kvp in _m)
        {
            mState.Set(kvp.Key, TensorAccessor.GetData(kvp.Value));
        }
        state.Set("m", mState);

        var vState = new StateDict();
        foreach (var kvp in _v)
        {
            vState.Set(kvp.Key, TensorAccessor.GetData(kvp.Value));
        }
        state.Set("v", vState);

        if (_amsgrad)
        {
            var vMaxState = new StateDict();
            foreach (var kvp in _vMax)
            {
                vMaxState.Set(kvp.Key, TensorAccessor.GetData(kvp.Value));
            }
            state.Set("v_max", vMaxState);
        }

        return state;
    }

    /// <inheritdoc/>
    public override void LoadState(StateDict state)
    {
        base.LoadState(state);
        _learningRate = state.Get<float>("learning_rate", 1e-3f);
        _beta1 = state.Get<float>("beta1", 0.9f);
        _beta2 = state.Get<float>("beta2", 0.999f);
        _epsilon = state.Get<float>("epsilon", 1e-8f);
        _weightDecay = state.Get<float>("weight_decay", 0.0f);
        _amsgrad = state.Get<bool>("amsgrad", false);
        _t = state.Get<int>("t", 0);

        // Load moment tensors
        var mState = state.Get<StateDict>("m");
        if (mState != null)
        {
            foreach (var kvp in _m)
            {
                var data = mState.Get<float[]>(kvp.Key);
                if (data != null && data.Length == kvp.Value.Size)
                {
                    Array.Copy(data, TensorAccessor.GetData(kvp.Value), data.Length);
                }
            }
        }

        var vState = state.Get<StateDict>("v");
        if (vState != null)
        {
            foreach (var kvp in _v)
            {
                var data = vState.Get<float[]>(kvp.Key);
                if (data != null && data.Length == kvp.Value.Size)
                {
                    Array.Copy(data, TensorAccessor.GetData(kvp.Value), data.Length);
                }
            }
        }

        if (_amsgrad)
        {
            var vMaxState = state.Get<StateDict>("v_max");
            if (vMaxState != null)
            {
                foreach (var kvp in _vMax)
                {
                    var data = vMaxState.Get<float[]>(kvp.Key);
                    if (data != null && data.Length == kvp.Value.Size)
                    {
                        Array.Copy(data, TensorAccessor.GetData(kvp.Value), data.Length);
                    }
                }
            }
        }
    }
}
