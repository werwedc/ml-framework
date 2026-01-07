using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using RitterFramework.Core.Tensor;
using MLFramework.Pipeline;

namespace MLFramework.Pipeline
{
    /// <summary>
    /// Optimizer for pipeline parallelism with gradient synchronization
    /// </summary>
    public class PipelineOptimizer : IDisposable
    {
        private readonly List<PipelineStage> _stages;
        private readonly IPipelineCommunicator _communicator;
        private float _learningRate;
        private int _disposed;

        /// <summary>
        /// Gets the learning rate
        /// </summary>
        public float LearningRate
        {
            get => _learningRate;
            set
            {
                if (value <= 0)
                    throw new ArgumentOutOfRangeException(nameof(value), "Learning rate must be positive");
                _learningRate = value;
            }
        }

        /// <summary>
        /// Gets the number of pipeline stages
        /// </summary>
        public int NumStages => _stages.Count;

        public PipelineOptimizer(
            List<PipelineStage> stages,
            IPipelineCommunicator communicator,
            float learningRate = 0.001f)
        {
            if (stages == null || stages.Count == 0)
                throw new ArgumentException("Stages cannot be null or empty", nameof(stages));
            if (communicator == null)
                throw new ArgumentNullException(nameof(communicator));
            if (learningRate <= 0)
                throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive");

            _stages = stages;
            _communicator = communicator;
            _learningRate = learningRate;
        }

        /// <summary>
        /// Perform an optimizer step with gradient synchronization
        /// </summary>
        public async Task StepAsync()
        {
            ThrowIfDisposed();

            // Synchronize gradients across all stages
            await SynchronizeGradientsAsync();

            // Update parameters on all stages
            foreach (var stage in _stages)
            {
                UpdateParameters(stage, _learningRate);
            }
        }

        /// <summary>
        /// Update parameters for a specific stage
        /// </summary>
        private void UpdateParameters(PipelineStage stage, float lr)
        {
            foreach (var parameter in stage.GetParameters())
            {
                if (parameter.Gradient != null)
                {
                    // SGD update: param = param - lr * grad
                    for (int i = 0; i < parameter.Data.Length; i++)
                    {
                        parameter.Data[i] -= lr * parameter.Gradient.Data[i];
                    }

                    // Clear gradient
                    if (parameter.Gradient != null)
                    {
                        Array.Clear(parameter.Gradient.Data, 0, parameter.Gradient.Data.Length);
                    }
                }
            }
        }

        /// <summary>
        /// Synchronize gradients across all pipeline stages
        /// </summary>
        public async Task SynchronizeGradientsAsync()
        {
            ThrowIfDisposed();

            // Wait for all stages to reach this point
            await _communicator.BarrierAsync();

            // In a real implementation, this would average gradients across stages
            // For now, this is a stub
        }

        /// <summary>
        /// Broadcast parameters from a specific stage to all other stages
        /// </summary>
        public async Task BroadcastParametersAsync(int rootStage)
        {
            ThrowIfDisposed();

            if (rootStage < 0 || rootStage >= _stages.Count)
                throw new ArgumentOutOfRangeException(nameof(rootStage), "Invalid root stage");

            // Wait for all stages to reach this point
            await _communicator.BarrierAsync();

            // In a real implementation, this would broadcast parameters
            // For now, this is a stub
        }

        /// <summary>
        /// Zero out gradients for all stages
        /// </summary>
        public void ZeroGradients()
        {
            ThrowIfDisposed();

            foreach (var stage in _stages)
            {
                foreach (var parameter in stage.GetParameters())
                {
                    if (parameter.Gradient != null)
                    {
                        Array.Clear(parameter.Gradient.Data, 0, parameter.Gradient.Data.Length);
                    }
                }
            }
        }

        /// <summary>
        /// Set learning rate
        /// </summary>
        public void SetLearningRate(float lr)
        {
            if (lr <= 0)
                throw new ArgumentOutOfRangeException(nameof(lr), "Learning rate must be positive");
            _learningRate = lr;
        }

        /// <summary>
        /// Get current learning rate
        /// </summary>
        public float GetLearningRate() => _learningRate;

        private void ThrowIfDisposed()
        {
            if (_disposed == 1)
                throw new ObjectDisposedException(nameof(PipelineOptimizer));
        }

        public void Dispose()
        {
            if (_disposed == 1)
                return;

            _disposed = 1;
        }
    }
}
