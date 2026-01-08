using System.Collections.Generic;
using RitterFramework.Core.Tensor;
using MLFramework.Schedulers;

namespace MLFramework.Optimizers;

/// <summary>
/// Abstract base class providing common functionality for all optimizers.
/// Supports learning rate scheduler integration.
/// </summary>
public abstract class Optimizer : IOptimizer
{
    private ILearningRateScheduler _scheduler;
    protected float _baseLearningRate;
    protected Dictionary<string, Tensor> _parameters;
    protected int _stepCount;

    /// <summary>
    /// Gets the current learning rate from the scheduler or base learning rate.
    /// </summary>
    public abstract float BaseLearningRate { get; }

    /// <summary>
    /// Current learning rate (may be modified by scheduler).
    /// </summary>
    public float LearningRate => GetCurrentLearningRate();

    /// <summary>
    /// Gets the learning rate scheduler for this optimizer.
    /// </summary>
    public ILearningRateScheduler Scheduler => _scheduler;

    /// <summary>
    /// Gets the current step count.
    /// </summary>
    public int StepCount => _stepCount;

    /// <summary>
    /// Gets the parameters being optimized.
    /// </summary>
    public Dictionary<string, Tensor> Parameters => _parameters;

    /// <summary>
    /// Initializes a new instance of the Optimizer class.
    /// </summary>
    /// <param name="parameters">Dictionary mapping parameter names to tensors.</param>
    protected Optimizer(Dictionary<string, Tensor> parameters)
    {
        _parameters = parameters ?? new Dictionary<string, Tensor>();
        _stepCount = 0;
    }

    /// <summary>
    /// Sets the parameters to optimize.
    /// </summary>
    /// <param name="parameters">Dictionary mapping parameter names to tensors.</param>
    public void SetParameters(Dictionary<string, Tensor> parameters)
    {
        _parameters = parameters ?? new Dictionary<string, Tensor>();
    }

    /// <summary>
    /// Sets a learning rate scheduler for this optimizer.
    /// </summary>
    /// <param name="scheduler">The scheduler to use, or null to disable scheduling.</param>
    public void SetScheduler(ILearningRateScheduler scheduler)
    {
        _scheduler = scheduler;
        if (_scheduler != null)
        {
            // Store the current learning rate as the base for the scheduler
            _baseLearningRate = BaseLearningRate;
        }
    }

    /// <summary>
    /// Gets the current learning rate from the scheduler or base learning rate.
    /// </summary>
    /// <returns>Current learning rate.</returns>
    protected float GetCurrentLearningRate()
    {
        if (_scheduler != null)
        {
            return _scheduler.GetLearningRate(_stepCount, _baseLearningRate);
        }
        return BaseLearningRate;
    }

    /// <summary>
    /// Updates the learning rate based on the scheduler (if any) before applying gradients.
    /// This should be called at the start of Step().
    /// </summary>
    protected virtual void UpdateLearningRate()
    {
        if (_scheduler != null)
        {
            float newLR = _scheduler.GetLearningRate(_stepCount, _baseLearningRate);
            SetLearningRate(newLR);
        }
    }

    /// <summary>
    /// Steps the scheduler forward (if any) after applying gradients.
    /// This should be called at the end of Step().
    /// </summary>
    protected virtual void StepScheduler()
    {
        _scheduler?.Step();
    }

    /// <summary>
    /// Performs an optimizer step with the given gradients.
    /// Concrete implementations should call UpdateLearningRate() at the start
    /// and StepScheduler() at the end.
    /// </summary>
    /// <param name="gradients">Dictionary mapping parameter names to gradient tensors.</param>
    public abstract void Step(Dictionary<string, Tensor> gradients);

    /// <summary>
    /// Applies a specific gradient to a specific parameter.
    /// </summary>
    /// <param name="parameterName">Name of the parameter to update.</param>
    /// <param name="gradient">Gradient tensor for the parameter.</param>
    public abstract void StepParameter(string parameterName, Tensor gradient);

    /// <summary>
    /// Zeroes out all gradients.
    /// </summary>
    public abstract void ZeroGrad();

    /// <summary>
    /// Sets the learning rate.
    /// </summary>
    /// <param name="lr">New learning rate.</param>
    public abstract void SetLearningRate(float lr);

    /// <summary>
    /// Gets the current state of the optimizer for checkpointing.
    /// </summary>
    /// <returns>State dictionary containing the optimizer state.</returns>
    public virtual StateDict GetState()
    {
        var state = new StateDict();
        state.Set("step_count", _stepCount);
        state.Set("base_lr", _baseLearningRate);

        if (_scheduler != null)
        {
            state.Set("scheduler_state", _scheduler.GetState());
        }

        return state;
    }

    /// <summary>
    /// Loads the optimizer state from a checkpoint.
    /// </summary>
    /// <param name="state">State dictionary to load from.</param>
    public virtual void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("step_count", 0);
        _baseLearningRate = state.Get<float>("base_lr", BaseLearningRate);

        if (_scheduler != null)
        {
            var schedulerState = state.Get<StateDict>("scheduler_state");
            if (schedulerState != null)
            {
                _scheduler.LoadState(schedulerState);
            }
        }
    }
}
