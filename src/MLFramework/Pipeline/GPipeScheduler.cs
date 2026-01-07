using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using RitterFramework.Core.Tensor;
using MLFramework.Pipeline;

namespace MLFramework.Pipeline
{
    /// <summary>
    /// GPipe scheduler for pipeline parallelism with gradient checkpointing
    /// </summary>
    public class GPipeScheduler : IDisposable
    {
        private readonly List<PipelineStage> _stages;
        private readonly IPipelineCommunicator _communicator;
        private readonly MicroBatchManager _microBatchManager;
        private readonly PipelineConfig _config;
        private readonly ActivationCheckpointManager _checkpointManager;
        private int _disposed;

        /// <summary>
        /// Gets the number of pipeline stages
        /// </summary>
        public int NumStages => _stages.Count;

        /// <summary>
        /// Gets the pipeline configuration
        /// </summary>
        public PipelineConfig Config => _config;

        public GPipeScheduler(
            List<PipelineStage> stages,
            IPipelineCommunicator communicator,
            MicroBatchManager microBatchManager,
            PipelineConfig config,
            ActivationCheckpointManager checkpointManager)
        {
            if (stages == null || stages.Count == 0)
                throw new ArgumentException("Stages cannot be null or empty", nameof(stages));
            if (communicator == null)
                throw new ArgumentNullException(nameof(communicator));
            if (microBatchManager == null)
                throw new ArgumentNullException(nameof(microBatchManager));
            if (config == null)
                throw new ArgumentNullException(nameof(config));
            if (checkpointManager == null)
                throw new ArgumentNullException(nameof(checkpointManager));

            _stages = stages;
            _communicator = communicator;
            _microBatchManager = microBatchManager;
            _config = config;
            _checkpointManager = checkpointManager;
        }

        /// <summary>
        /// Execute forward pass through the pipeline
        /// </summary>
        public async Task<Tensor> ForwardAsync(Tensor input, int microBatchIdx)
        {
            ThrowIfDisposed();

            if (input == null)
                throw new ArgumentNullException(nameof(input));

            Tensor current = input;

            foreach (var stage in _stages)
            {
                // First stage receives from previous micro-batch or input
                if (stage.Rank == 0)
                {
                    current = await Task.Run(() => stage.Forward(current));
                }
                else
                {
                    // Receive from previous stage
                    current = await _communicator.ReceiveAsync(stage.Rank - 1);
                    current = await Task.Run(() => stage.Forward(current));
                }

                // Store activation if checkpointing is enabled
                if (_checkpointManager.Strategy != CheckpointStrategy.RecomputeAll)
                {
                    _checkpointManager.StoreActivation(microBatchIdx, current);
                }

                // Send to next stage (except last)
                if (stage.Rank < _stages.Count - 1)
                {
                    await _communicator.SendAsync(current, stage.Rank + 1);
                }
            }

            return current;
        }

        /// <summary>
        /// Execute backward pass through the pipeline
        /// </summary>
        public async Task<List<Tensor>> BackwardAsync(Tensor gradient, int microBatchIdx)
        {
            ThrowIfDisposed();

            if (gradient == null)
                throw new ArgumentNullException(nameof(gradient));

            Tensor current = gradient;
            var gradients = new List<Tensor>();

            // Backward pass goes in reverse
            for (int i = _stages.Count - 1; i >= 0; i--)
            {
                var stage = _stages[i];

                // Last stage receives gradient from loss
                if (stage.Rank == _stages.Count - 1)
                {
                    // Simulated backward - in real implementation, this would compute gradients
                    gradients.Add(current);
                }
                else
                {
                    // Receive gradient from next stage
                    current = await _communicator.ReceiveAsync(stage.Rank + 1);
                    gradients.Add(current);
                }

                // Send gradient to previous stage (except first)
                if (stage.Rank > 0)
                {
                    await _communicator.SendAsync(current, stage.Rank - 1);
                }
            }

            return gradients;
        }

        /// <summary>
        /// Execute a single training iteration
        /// </summary>
        public async Task TrainIterationAsync(Tensor input, Tensor targets, Func<Tensor, Tensor, Tensor> lossFunction)
        {
            ThrowIfDisposed();

            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (targets == null)
                throw new ArgumentNullException(nameof(targets));
            if (lossFunction == null)
                throw new ArgumentNullException(nameof(lossFunction));

            // Split input into micro-batches
            var microBatches = _microBatchManager.SplitBatch(input);

            // Forward pass for all micro-batches
            var outputs = new List<Tensor>();
            for (int i = 0; i < microBatches.Count; i++)
            {
                var output = await ForwardAsync(microBatches[i], i);
                outputs.Add(output);
            }

            // Backward pass for all micro-batches
            for (int i = outputs.Count - 1; i >= 0; i--)
            {
                var loss = lossFunction(outputs[i], targets);
                var gradient = loss; // In real implementation, compute gradient of loss
                var gradients = await BackwardAsync(gradient, i);
                _microBatchManager.AccumulateGradients(gradients);
            }

            // Get averaged gradients
            var averagedGradients = _microBatchManager.GetAccumulatedGradients();
        }

        /// <summary>
        /// Reset the scheduler state
        /// </summary>
        public void Reset()
        {
            ThrowIfDisposed();
            _microBatchManager.ResetGradients();
            _checkpointManager.Clear();
        }

        private void ThrowIfDisposed()
        {
            if (_disposed == 1)
                throw new ObjectDisposedException(nameof(GPipeScheduler));
        }

        public void Dispose()
        {
            if (_disposed == 1)
                return;

            _communicator?.Dispose();
            _microBatchManager?.Dispose();
            _disposed = 1;
        }
    }
}
