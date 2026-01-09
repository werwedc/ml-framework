using System;
using System.Collections.Generic;
using MLFramework.Schedulers;

namespace MLFramework.ModelZoo.TransferLearning;

/// <summary>
/// Scheduler for layer-wise learning rate updates during fine-tuning.
/// Supports different schedules per layer and multipliers for base learning rates.
/// </summary>
public class LayerWiseLrScheduler : BaseScheduler, IStepScheduler
{
    #region Factory Methods for Common Fine-Tuning Schedules

    /// <summary>
    /// Creates a discriminative learning rate schedule with decreasing LRs from early to late layers.
    /// This is useful for transfer learning where early layers should change more slowly.
    /// </summary>
    /// <param name="baseLr">Base learning rate.</param>
    /// <param name="multipliers">Array of multipliers for each layer (from early to late).</param>
    /// <returns>A LayerWiseLrScheduler configured with discriminative LRs.</returns>
    public static LayerWiseLrScheduler CreateDiscriminative(float baseLr, float[] multipliers)
    {
        if (baseLr <= 0)
            throw new ArgumentOutOfRangeException(nameof(baseLr), "Base learning rate must be positive.");

        if (multipliers == null || multipliers.Length == 0)
            throw new ArgumentException("Multipliers array must not be empty.", nameof(multipliers));

        var scheduler = new LayerWiseLrScheduler(baseLr);

        for (int i = 0; i < multipliers.Length; i++)
        {
            string layerName = $"layer_{i}";
            scheduler.SetMultiplierByLayerIndex(i, multipliers[i]);
        }

        return scheduler;
    }

    /// <summary>
    /// Creates a triangular learning rate schedule for cyclical learning rates.
    /// Useful for one-cycle training and cyclical learning rate strategies.
    /// </summary>
    /// <param name="baseLr">Base learning rate.</param>
    /// <param name="minLr">Minimum learning rate in the triangular cycle.</param>
    /// <param name="maxLr">Maximum learning rate in the triangular cycle.</param>
    /// <param name="cycleSteps">Number of steps in one complete cycle (default: 1000).</param>
    /// <returns>A LayerWiseLrScheduler configured with triangular LR.</returns>
    public static LayerWiseLrScheduler CreateTriangular(float baseLr, float minLr, float maxLr, int cycleSteps = 1000)
    {
        if (baseLr <= 0)
            throw new ArgumentOutOfRangeException(nameof(baseLr), "Base learning rate must be positive.");

        if (minLr <= 0)
            throw new ArgumentOutOfRangeException(nameof(minLr), "Minimum learning rate must be positive.");

        if (maxLr <= minLr)
            throw new ArgumentOutOfRangeException(nameof(maxLr), "Maximum learning rate must be greater than minimum.");

        if (cycleSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(cycleSteps), "Cycle steps must be positive.");

        var scheduler = new LayerWiseLrScheduler(baseLr);

        // Configure triangular schedule for all layers
        scheduler.SetLayerSchedule(
            pattern: "",
            scheduleType: ScheduleType.Cosine,
            initialMultiplier: minLr / baseLr,
            finalMultiplier: maxLr / baseLr,
            totalSteps: cycleSteps / 2);

        return scheduler;
    }

    /// <summary>
    /// Creates a warmup followed by cosine annealing schedule.
    /// This is commonly used in modern transformer training and fine-tuning.
    /// </summary>
    /// <param name="baseLr">Base learning rate.</param>
    /// <param name="warmupLr">Target learning rate after warmup.</param>
    /// <param name="warmupSteps">Number of warmup steps.</param>
    /// <param name="totalSteps">Total number of training steps (including warmup).</param>
    /// <returns>A LayerWiseLrScheduler configured with warmup and cosine annealing.</returns>
    public static LayerWiseLrScheduler CreateWarmupCosine(float baseLr, float warmupLr, int warmupSteps, int totalSteps)
    {
        if (baseLr <= 0)
            throw new ArgumentOutOfRangeException(nameof(baseLr), "Base learning rate must be positive.");

        if (warmupLr <= 0)
            throw new ArgumentOutOfRangeException(nameof(warmupLr), "Warmup learning rate must be positive.");

        if (warmupSteps < 0)
            throw new ArgumentOutOfRangeException(nameof(warmupSteps), "Warmup steps cannot be negative.");

        if (totalSteps <= warmupSteps)
            throw new ArgumentOutOfRangeException(nameof(totalSteps), "Total steps must be greater than warmup steps.");

        var scheduler = new LayerWiseLrScheduler(baseLr);

        // Configure warmup phase
        scheduler.SetLayerSchedule(
            pattern: "warmup",
            scheduleType: ScheduleType.Linear,
            initialMultiplier: baseLr / baseLr,
            finalMultiplier: warmupLr / baseLr,
            totalSteps: warmupSteps);

        // Configure cosine annealing phase
        scheduler.SetLayerSchedule(
            pattern: "annealing",
            scheduleType: ScheduleType.Cosine,
            initialMultiplier: warmupLr / baseLr,
            finalMultiplier: 0.01f, // Final LR is 1% of warmup LR
            totalSteps: totalSteps - warmupSteps);

        return scheduler;
    }

    /// <summary>
    /// Creates a layered discriminative schedule with automatic multiplier calculation.
    /// Multipliers are calculated using a geometric progression.
    /// </summary>
    /// <param name="baseLr">Base learning rate.</param>
    /// <param name="numLayers">Number of layers.</param>
    /// <param name="firstLayerMultiplier">Multiplier for the first (earliest) layer.</param>
    /// <param name="lastLayerMultiplier">Multiplier for the last (latest) layer.</param>
    /// <returns>A LayerWiseLrScheduler configured with geometric discriminative LRs.</returns>
    public static LayerWiseLrScheduler CreateGeometricDiscriminative(
        float baseLr,
        int numLayers,
        float firstLayerMultiplier = 0.1f,
        float lastLayerMultiplier = 1.0f)
    {
        if (baseLr <= 0)
            throw new ArgumentOutOfRangeException(nameof(baseLr), "Base learning rate must be positive.");

        if (numLayers <= 0)
            throw new ArgumentOutOfRangeException(nameof(numLayers), "Number of layers must be positive.");

        var scheduler = new LayerWiseLrScheduler(baseLr);

        // Calculate geometric progression of multipliers
        float ratio = (float)Math.Pow(lastLayerMultiplier / firstLayerMultiplier, 1.0 / (numLayers - 1));

        for (int i = 0; i < numLayers; i++)
        {
            float multiplier = firstLayerMultiplier * (float)Math.Pow(ratio, i);
            scheduler.SetMultiplierByLayerIndex(i, multiplier);
        }

        return scheduler;
    }

    /// <summary>
    /// Creates a schedule for gradual unfreezing with increasing learning rates.
    /// Earlier (frozen) layers have lower LRs, later (unfrozen) layers have higher LRs.
    /// </summary>
    /// <param name="baseLr">Base learning rate.</param>
    /// <param name="numGroups">Number of parameter groups.</param>
    /// <param name="frozenLr">Learning rate for frozen layers.</param>
    /// <param name="unfrozenLr">Learning rate for unfrozen layers.</param>
    /// <returns>A LayerWiseLrScheduler configured for gradual unfreezing.</returns>
    public static LayerWiseLrScheduler CreateGradualUnfreezing(
        float baseLr,
        int numGroups,
        float frozenLr,
        float unfrozenLr)
    {
        if (baseLr <= 0)
            throw new ArgumentOutOfRangeException(nameof(baseLr), "Base learning rate must be positive.");

        if (numGroups <= 0)
            throw new ArgumentOutOfRangeException(nameof(numGroups), "Number of groups must be positive.");

        if (frozenLr <= 0)
            throw new ArgumentOutOfRangeException(nameof(frozenLr), "Frozen learning rate must be positive.");

        if (unfrozenLr <= 0)
            throw new ArgumentOutOfRangeException(nameof(unfrozenLr), "Unfrozen learning rate must be positive.");

        if (unfrozenLr < frozenLr)
            throw new ArgumentOutOfRangeException(nameof(unfrozenLr), "Unfrozen LR must be greater than or equal to frozen LR.");

        var scheduler = new LayerWiseLrScheduler(baseLr);

        // Create linear progression from frozen to unfrozen LR
        for (int i = 0; i < numGroups; i++)
        {
            float t = numGroups > 1 ? i / (float)(numGroups - 1) : 0;
            float lr = frozenLr + (unfrozenLr - frozenLr) * t;
            scheduler.SetMultiplierByLayerIndex(i, lr / baseLr);
        }

        return scheduler;
    }

    #endregion
    private readonly float _baseLearningRate;
    private readonly Dictionary<string, LayerSchedule> _layerSchedules;
    private readonly Dictionary<string, float> _multipliers;

    /// <summary>
    /// Gets the base learning rate for this scheduler.
    /// </summary>
    public float BaseLearningRate => _baseLearningRate;

    /// <summary>
    /// Gets the number of layers with custom schedules.
    /// </summary>
    public int ScheduledLayerCount => _layerSchedules.Count;

    /// <summary>
    /// Initializes a new instance of the LayerWiseLrScheduler class.
    /// </summary>
    /// <param name="baseLearningRate">Base learning rate.</param>
    public LayerWiseLrScheduler(float baseLearningRate)
    {
        if (baseLearningRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(baseLearningRate), "Learning rate must be positive.");

        _baseLearningRate = baseLearningRate;
        _layerSchedules = new Dictionary<string, LayerSchedule>();
        _multipliers = new Dictionary<string, float>();
    }

    /// <summary>
    /// Sets a multiplier for layers matching a pattern.
    /// </summary>
    /// <param name="layerPattern">Regex pattern to match layer names.</param>
    /// <param name="multiplier">Multiplier to apply to the base LR.</param>
    public void SetMultiplier(string layerPattern, float multiplier)
    {
        if (string.IsNullOrEmpty(layerPattern))
            throw new ArgumentException("Pattern cannot be null or empty.", nameof(layerPattern));

        _multipliers[layerPattern] = multiplier;
    }

    /// <summary>
    /// Sets a multiplier for a specific layer by index.
    /// </summary>
    /// <param name="layerIndex">Index of the layer (0-based).</param>
    /// <param name="multiplier">Multiplier to apply to the base LR.</param>
    public void SetMultiplierByLayerIndex(int layerIndex, float multiplier)
    {
        string layerName = $"layer_{layerIndex}";
        _multipliers[layerName] = multiplier;
    }

    /// <summary>
    /// Sets a custom schedule for a specific layer.
    /// </summary>
    /// <param name="layerPattern">Regex pattern to match layer names.</param>
    /// <param name="scheduleType">Type of schedule to use.</param>
    /// <param name="initialMultiplier">Initial learning rate multiplier.</param>
    /// <param name="finalMultiplier">Final learning rate multiplier.</param>
    /// <param name="totalSteps">Total number of steps for the schedule.</param>
    public void SetLayerSchedule(
        string layerPattern,
        ScheduleType scheduleType,
        float initialMultiplier,
        float finalMultiplier,
        int totalSteps)
    {
        if (string.IsNullOrEmpty(layerPattern))
            throw new ArgumentException("Pattern cannot be null or empty.", nameof(layerPattern));

        _layerSchedules[layerPattern] = new LayerSchedule
        {
            Pattern = layerPattern,
            Type = scheduleType,
            InitialMultiplier = initialMultiplier,
            FinalMultiplier = finalMultiplier,
            TotalSteps = totalSteps
        };
    }

    /// <summary>
    /// Gets the current learning rates for all parameter groups.
    /// </summary>
    /// <returns>Dictionary mapping layer patterns to current learning rates.</returns>
    public Dictionary<string, float> GetCurrentLrs()
    {
        var currentLrs = new Dictionary<string, float>();

        // Get base LR from scheduler
        float currentBaseLr = GetLearningRate(_stepCount, _baseLearningRate);

        // Apply multipliers for layers with fixed multipliers
        foreach (var kvp in _multipliers)
        {
            currentLrs[kvp.Key] = currentBaseLr * kvp.Value;
        }

        // Apply schedules for layers with custom schedules
        foreach (var kvp in _layerSchedules)
        {
            float multiplier = GetScheduleMultiplier(kvp.Value, _stepCount);
            currentLrs[kvp.Key] = currentBaseLr * multiplier;
        }

        return currentLrs;
    }

    /// <summary>
    /// Gets the current learning rate for a specific parameter group.
    /// </summary>
    /// <param name="parameterName">Name of the parameter.</param>
    /// <returns>Learning rate for this parameter.</returns>
    public float GetParameterLearningRate(string parameterName)
    {
        if (string.IsNullOrEmpty(parameterName))
            throw new ArgumentException("Parameter name cannot be null or empty.", nameof(parameterName));

        // Get base LR from scheduler
        float currentBaseLr = GetLearningRate(_stepCount, _baseLearningRate);

        // Check if parameter matches any layer schedule
        foreach (var kvp in _layerSchedules)
        {
            if (parameterName.Contains(kvp.Key) || MatchesPattern(parameterName, kvp.Key))
            {
                float multiplier = GetScheduleMultiplier(kvp.Value, _stepCount);
                return currentBaseLr * multiplier;
            }
        }

        // Check if parameter matches any multiplier pattern
        foreach (var kvp in _multipliers)
        {
            if (parameterName.Contains(kvp.Key) || MatchesPattern(parameterName, kvp.Key))
            {
                return currentBaseLr * kvp.Value;
            }
        }

        // Default to base LR
        return currentBaseLr;
    }

    /// <summary>
    /// Gets the learning rate for a specific parameter group.
    /// </summary>
    /// <param name="groupIndex">Index of the parameter group.</param>
    /// <param name="layerName">Name of the layer (optional).</param>
    /// <returns>Learning rate for this parameter group.</returns>
    public float GetGroupLearningRate(int groupIndex, string layerName = null)
    {
        if (layerName != null)
        {
            return GetParameterLearningRate(layerName);
        }

        // Default to base LR if no layer name provided
        return GetLearningRate(_stepCount, _baseLearningRate);
    }

    public override float GetLearningRate(int step, float baseLearningRate)
    {
        // Return the base learning rate by default
        // Use GetParameterLearningRate for layer-specific rates
        return baseLearningRate;
    }

    /// <summary>
    /// Gets the schedule multiplier for a given step.
    /// </summary>
    private float GetScheduleMultiplier(LayerSchedule schedule, int step)
    {
        float t = Math.Min(1.0f, step / (float)schedule.TotalSteps);

        switch (schedule.Type)
        {
            case ScheduleType.Linear:
                return schedule.InitialMultiplier + (schedule.FinalMultiplier - schedule.InitialMultiplier) * t;

            case ScheduleType.Cosine:
                return schedule.FinalMultiplier +
                       (schedule.InitialMultiplier - schedule.FinalMultiplier) *
                       (1.0f + (float)Math.Cos(t * Math.PI)) / 2.0f;

            case ScheduleType.Step:
                // Step schedule: use initial multiplier for first half, final for second half
                return t < 0.5f ? schedule.InitialMultiplier : schedule.FinalMultiplier;

            case ScheduleType.Constant:
            default:
                return schedule.InitialMultiplier;
        }
    }

    /// <summary>
    /// Checks if a parameter name matches a pattern.
    /// Simple implementation that checks for substring match or starts with.
    /// </summary>
    private bool MatchesPattern(string parameterName, string pattern)
    {
        if (parameterName.StartsWith(pattern))
            return true;

        if (parameterName.Contains(pattern))
            return true;

        return false;
    }

    public override StateDict GetState()
    {
        var state = new StateDict();
        state.Set("base_lr", _baseLearningRate);
        state.Set("step_count", _stepCount);
        state.Set("epoch_count", _epochCount);

        // Save multipliers
        var multipliersState = new StateDict();
        foreach (var kvp in _multipliers)
        {
            multipliersState.Set(kvp.Key, kvp.Value);
        }
        state.Set("multipliers", multipliersState);

        // Save layer schedules
        var schedulesState = new StateDict();
        int idx = 0;
        foreach (var kvp in _layerSchedules)
        {
            var scheduleState = new StateDict();
            scheduleState.Set("pattern", kvp.Value.Pattern);
            scheduleState.Set("type", (int)kvp.Value.Type);
            scheduleState.Set("initial_multiplier", kvp.Value.InitialMultiplier);
            scheduleState.Set("final_multiplier", kvp.Value.FinalMultiplier);
            scheduleState.Set("total_steps", kvp.Value.TotalSteps);
            schedulesState.Set($"schedule_{idx}", scheduleState);
            idx++;
        }
        state.Set("layer_schedules", schedulesState);

        return state;
    }

    public override void LoadState(StateDict state)
    {
        _stepCount = state.Get<int>("step_count", 0);
        _epochCount = state.Get<int>("epoch_count", 0);

        // Load multipliers
        var multipliersState = state.Get<StateDict>("multipliers");
        if (multipliersState != null)
        {
            _multipliers.Clear();
            foreach (var key in multipliersState.Keys)
            {
                _multipliers[key] = multipliersState.Get<float>(key);
            }
        }

        // Load layer schedules
        var schedulesState = state.Get<StateDict>("layer_schedules");
        if (schedulesState != null)
        {
            _layerSchedules.Clear();
            foreach (var key in schedulesState.Keys)
            {
                var scheduleState = schedulesState.Get<StateDict>(key);
                if (scheduleState != null)
                {
                    var schedule = new LayerSchedule
                    {
                        Pattern = scheduleState.Get<string>("pattern"),
                        Type = (ScheduleType)scheduleState.Get<int>("type"),
                        InitialMultiplier = scheduleState.Get<float>("initial_multiplier"),
                        FinalMultiplier = scheduleState.Get<float>("final_multiplier"),
                        TotalSteps = scheduleState.Get<int>("total_steps")
                    };
                    _layerSchedules[schedule.Pattern] = schedule;
                }
            }
        }
    }

    /// <summary>
    /// Types of schedules supported for layer-wise learning rates.
    /// </summary>
    public enum ScheduleType
    {
        /// <summary>
        /// Constant multiplier throughout training.
        /// </summary>
        Constant,

        /// <summary>
        /// Linear interpolation between initial and final multipliers.
        /// </summary>
        Linear,

        /// <summary>
        /// Cosine annealing between initial and final multipliers.
        /// </summary>
        Cosine,

        /// <summary>
        /// Step function: initial multiplier then final multiplier at midpoint.
        /// </summary>
        Step
    }

    /// <summary>
    /// Internal class representing a schedule for a layer.
    /// </summary>
    private class LayerSchedule
    {
        public string Pattern { get; set; }
        public ScheduleType Type { get; set; }
        public float InitialMultiplier { get; set; }
        public float FinalMultiplier { get; set; }
        public int TotalSteps { get; set; }
    }
}
