using System;
using System.Collections.Generic;
using System.Linq;
using MLFramework.NN;

namespace MLFramework.ModelZoo.TransferLearning
{
    /// <summary>
    /// Scheduler for progressive layer unfreezing during fine-tuning.
    /// Implements the "gradual unfreezing" technique from ULMFiT.
    /// </summary>
    public class GradualUnfreezingScheduler
    {
        private readonly Module _module;
        private readonly double[] _unfreezeThresholds;
        private int _currentStage;
        private bool _isInitialized;

        /// <summary>
        /// Gets the module being managed by this scheduler.
        /// </summary>
        public Module Module => _module;

        /// <summary>
        /// Gets the unfreeze thresholds for each stage.
        /// </summary>
        public IReadOnlyList<double> UnfreezeThresholds => _unfreezeThresholds;

        /// <summary>
        /// Gets the current unfreezing stage (0-based).
        /// </summary>
        public int CurrentStage => _currentStage;

        /// <summary>
        /// Gets the total number of unfreezing stages.
        /// </summary>
        public int TotalStages => _unfreezeThresholds.Length;

        /// <summary>
        /// Event fired when layers are unfrozen.
        /// </summary>
        public event Action<int, int>? OnUnfreezeStage;

        /// <summary>
        /// Creates a new gradual unfreezing scheduler.
        /// </summary>
        /// <param name="module">The module to manage.</param>
        /// <param name="unfreezeThresholds">Array of thresholds (0.0-1.0) for each unfreezing stage.</param>
        public GradualUnfreezingScheduler(Module module, double[] unfreezeThresholds)
        {
            _module = module ?? throw new ArgumentNullException(nameof(module));
            _unfreezeThresholds = unfreezeThresholds ?? throw new ArgumentNullException(nameof(unfreezeThresholds));

            if (_unfreezeThresholds.Length == 0)
                throw new ArgumentException("At least one unfreeze threshold must be provided.", nameof(unfreezeThresholds));

            // Validate thresholds
            for (int i = 0; i < _unfreezeThresholds.Length; i++)
            {
                if (_unfreezeThresholds[i] < 0.0 || _unfreezeThresholds[i] > 1.0)
                    throw new ArgumentOutOfRangeException(nameof(unfreezeThresholds),
                        $"Threshold at index {i} is {_unfreezeThresholds[i]}, but must be between 0.0 and 1.0.");
            }

            // Ensure thresholds are sorted
            for (int i = 1; i < _unfreezeThresholds.Length; i++)
            {
                if (_unfreezeThresholds[i] < _unfreezeThresholds[i - 1])
                    throw new ArgumentException("Unfreeze thresholds must be in ascending order.", nameof(unfreezeThresholds));
            }

            _currentStage = -1;
            _isInitialized = false;
        }

        /// <summary>
        /// Creates a new gradual unfreezing scheduler with evenly distributed thresholds.
        /// </summary>
        /// <param name="module">The module to manage.</param>
        /// <param name="numStages">Number of unfreezing stages.</param>
        /// <returns>A new scheduler instance.</returns>
        public static GradualUnfreezingScheduler CreateEvenlyDistributed(Module module, int numStages)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));
            if (numStages <= 0)
                throw new ArgumentOutOfRangeException(nameof(numStages), "Number of stages must be positive.");

            var thresholds = new double[numStages];
            double step = 1.0 / numStages;
            for (int i = 0; i < numStages; i++)
            {
                thresholds[i] = (i + 1) * step;
            }

            return new GradualUnfreezingScheduler(module, thresholds);
        }

        /// <summary>
        /// Initializes the scheduler by freezing all layers in the model.
        /// </summary>
        public void Initialize()
        {
            _module.Freeze();
            _currentStage = -1;
            _isInitialized = true;
        }

        /// <summary>
        /// Updates unfreezing based on training progress.
        /// </summary>
        /// <param name="epochProgress">Progress through training (0.0 to 1.0).</param>
        /// <returns>True if a new unfreezing stage was triggered, false otherwise.</returns>
        public bool UpdateUnfreezing(double epochProgress)
        {
            if (!_isInitialized)
                Initialize();

            if (epochProgress < 0.0 || epochProgress > 1.0)
                throw new ArgumentOutOfRangeException(nameof(epochProgress),
                    "Epoch progress must be between 0.0 and 1.0.");

            // Determine which stage we should be at
            int targetStage = -1;
            for (int i = 0; i < _unfreezeThresholds.Length; i++)
            {
                if (epochProgress >= _unfreezeThresholds[i])
                {
                    targetStage = i;
                }
                else
                {
                    break;
                }
            }

            // If we need to advance to a new stage
            if (targetStage > _currentStage && targetStage < _unfreezeThresholds.Length)
            {
                _currentStage = targetStage;
                UnfreezeStage(_currentStage);
                OnUnfreezeStage?.Invoke(_currentStage, _unfreezeThresholds.Length);
                return true;
            }

            return false;
        }

        /// <summary>
        /// Unfreezes layers up to the specified stage.
        /// Stage 0 unfreezes only the last layer, Stage 1 unfreezes the last 2 layers, etc.
        /// </summary>
        /// <param name="stage">The unfreezing stage (0-based).</param>
        private void UnfreezeStage(int stage)
        {
            int numLayersToUnfreeze = stage + 1;
            var layersToUnfreeze = LayerSelectionHelper.SelectLastN(_module, numLayersToUnfreeze);

            foreach (var layer in layersToUnfreeze)
            {
                layer.Unfreeze();
            }
        }

        /// <summary>
        /// Manually advances to the next unfreezing stage.
        /// </summary>
        /// <returns>True if advanced, false if already at final stage.</returns>
        public bool AdvanceToNextStage()
        {
            if (_currentStage >= _unfreezeThresholds.Length - 1)
                return false;

            _currentStage++;
            UnfreezeStage(_currentStage);
            OnUnfreezeStage?.Invoke(_currentStage, _unfreezeThresholds.Length);
            return true;
        }

        /// <summary>
        /// Manually sets the unfreezing stage.
        /// </summary>
        /// <param name="stage">The target stage (0-based).</param>
        public void SetStage(int stage)
        {
            if (stage < -1 || stage >= _unfreezeThresholds.Length)
                throw new ArgumentOutOfRangeException(nameof(stage),
                    $"Stage must be between -1 and {_unfreezeThresholds.Length - 1}.");

            // If going back, freeze everything first
            if (stage < _currentStage)
            {
                _module.Freeze();
            }

            // Then unfreeze up to the target stage
            _currentStage = stage;
            if (_currentStage >= 0)
            {
                UnfreezeStage(_currentStage);
            }
        }

        /// <summary>
        /// Resets the scheduler, freezing all layers.
        /// </summary>
        public void Reset()
        {
            _module.Freeze();
            _currentStage = -1;
        }

        /// <summary>
        /// Gets the number of layers that will be unfrozen at a given stage.
        /// </summary>
        /// <param name="stage">The stage to query.</param>
        /// <returns>Number of unfrozen layers at that stage.</returns>
        public int GetUnfrozenLayerCount(int stage)
        {
            if (stage < -1 || stage >= _unfreezeThresholds.Length)
                throw new ArgumentOutOfRangeException(nameof(stage));

            if (stage < 0)
                return 0;

            return stage + 1;
        }

        /// <summary>
        /// Gets a summary of the current state.
        /// </summary>
        /// <returns>A string describing the current state.</returns>
        public string GetStateSummary()
        {
            int totalLayers = _module.GetAllModules().Count();
            int unfrozenLayers = _module.GetUnfrozenLayers().Count();

            return $"Gradual Unfreezing Scheduler - Stage {_currentStage + 1}/{_unfreezeThresholds.Length}, " +
                   $"Unfrozen: {unfrozenLayers}/{totalLayers} layers";
        }
    }
}
