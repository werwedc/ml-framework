using System;
using System.Collections.Generic;
using System.Linq;
using RitterFramework.Core.Tensor;
using MLFramework.NN;

namespace MLFramework.Pipeline
{
    /// <summary>
    /// Manages activation checkpointing for pipeline stages
    /// </summary>
    public class ActivationCheckpointManager : IDisposable
    {
        private readonly CheckpointStrategy _strategy;
        private readonly int _checkpointInterval;
        private readonly long _memoryThreshold;
        private readonly Dictionary<int, Tensor> _checkpoints;
        private readonly Dictionary<int, CheckpointMetadata> _metadata;
        private readonly PipelineStage _stage;
        private bool _disposed;

        /// <summary>
        /// Current checkpoint strategy
        /// </summary>
        public CheckpointStrategy Strategy => _strategy;

        /// <summary>
        /// Get number of stored checkpoints
        /// </summary>
        public int CheckpointCount => _checkpoints.Count;

        public ActivationCheckpointManager(
            CheckpointStrategy strategy,
            PipelineStage stage,
            int checkpointInterval = 1,
            long memoryThreshold = 1024 * 1024 * 1024) // 1GB default
        {
            _strategy = strategy;
            _stage = stage ?? throw new ArgumentNullException(nameof(stage));
            _checkpointInterval = checkpointInterval;
            _memoryThreshold = memoryThreshold;
            _checkpoints = new Dictionary<int, Tensor>();
            _metadata = new Dictionary<int, CheckpointMetadata>();
        }

        /// <summary>
        /// Check if activation should be stored for a given micro-batch
        /// </summary>
        public bool ShouldCheckpoint(int microBatchIndex)
        {
            switch (_strategy)
            {
                case CheckpointStrategy.StoreAll:
                    return true;

                case CheckpointStrategy.RecomputeAll:
                    return false;

                case CheckpointStrategy.Selective:
                    // Always checkpoint first and last, and every Nth activation
                    var interval = _checkpointInterval > 0 ? _checkpointInterval : 1;
                    if (microBatchIndex == 0) return true;
                    return microBatchIndex % interval == 0;

                case CheckpointStrategy.MemoryBased:
                    // We'll check memory threshold during StoreActivation
                    // For ShouldCheckpoint, always return true to allow decision during store
                    return true;

                default:
                    throw new ArgumentOutOfRangeException($"Unknown strategy: {_strategy}");
            }
        }

        /// <summary>
        /// Store an activation for later backward pass
        /// </summary>
        public void StoreActivation(int microBatchIndex, Tensor activation)
        {
            if (activation == null)
                throw new ArgumentNullException(nameof(activation));

            // Check if we should checkpoint for this micro-batch
            if (!ShouldCheckpoint(microBatchIndex))
                return;

            // Handle MemoryBased strategy
            if (_strategy == CheckpointStrategy.MemoryBased)
            {
                var currentMemory = EstimateMemoryUsage();
                var activationMemory = EstimateTensorMemory(activation);

                // Remove oldest checkpoints if we would exceed threshold
                while (currentMemory + activationMemory > _memoryThreshold && _checkpoints.Count > 2)
                {
                    // Always keep first and last (we don't know if last is stored yet)
                    var oldestKey = _metadata
                        .OrderBy(m => m.Value.Timestamp)
                        .First(m => m.Key != 0 && m.Key != _metadata.Keys.Max())
                        .Key;

                    RemoveCheckpoint(oldestKey);
                    currentMemory = EstimateMemoryUsage();
                }

                // Still can't store? Skip this activation unless it's first or last
                if (currentMemory + activationMemory > _memoryThreshold && _checkpoints.Count > 0)
                {
                    // Only skip if we already have at least one checkpoint
                    return;
                }
            }

            // Clone the activation to avoid modification
            var clonedActivation = activation.Clone();

            // Store the activation
            if (_checkpoints.ContainsKey(microBatchIndex))
            {
                // Replace existing checkpoint
                RemoveCheckpoint(microBatchIndex);
            }

            _checkpoints[microBatchIndex] = clonedActivation;

            // Create and store metadata
            var memorySize = EstimateTensorMemory(clonedActivation);
            var shape = clonedActivation.Shape.Select(s => (long)s).ToArray();
            _metadata[microBatchIndex] = new CheckpointMetadata(microBatchIndex, memorySize, shape);
        }

        /// <summary>
        /// Retrieve a stored activation
        /// </summary>
        public Tensor? GetActivation(int microBatchIndex)
        {
            _checkpoints.TryGetValue(microBatchIndex, out var activation);
            return activation;
        }

        /// <summary>
        /// Check if activation is available (stored)
        /// </summary>
        public bool HasActivation(int microBatchIndex)
        {
            return _checkpoints.ContainsKey(microBatchIndex);
        }

        /// <summary>
        /// Recompute activation for a micro-batch
        /// </summary>
        /// <param name="input">Input to the stage</param>
        /// <param name="microBatchIndex">Micro-batch index</param>
        /// <returns>Recomputed activation</returns>
        public Tensor RecomputeActivation(Tensor input, int microBatchIndex)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            // Run forward pass of stage module
            var activation = _stage.Forward(input);

            // Optionally cache result if strategy changed mid-execution
            if (ShouldCheckpoint(microBatchIndex) && !HasActivation(microBatchIndex))
            {
                StoreActivation(microBatchIndex, activation);
            }

            return activation;
        }

        /// <summary>
        /// Get or compute activation (handles both cases)
        /// </summary>
        public Tensor GetOrComputeActivation(Tensor input, int microBatchIndex)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            // Check if activation is stored
            if (HasActivation(microBatchIndex))
            {
                return GetActivation(microBatchIndex)!;
            }

            // If not stored, recompute it
            return RecomputeActivation(input, microBatchIndex);
        }

        /// <summary>
        /// Clear all checkpoints
        /// </summary>
        public void Clear()
        {
            var keys = _checkpoints.Keys.ToList();
            foreach (var key in keys)
            {
                RemoveCheckpoint(key);
            }
        }

        /// <summary>
        /// Estimate memory used by checkpoints
        /// </summary>
        public long EstimateMemoryUsage()
        {
            return _metadata.Values.Sum(m => m.MemorySize);
        }

        private void RemoveCheckpoint(int microBatchIndex)
        {
            _checkpoints.Remove(microBatchIndex);
            _metadata.Remove(microBatchIndex);
        }

        private long EstimateTensorMemory(Tensor tensor)
        {
            // Use Tensor.Size * sizeof(float) for estimation
            return tensor.Size * sizeof(float);
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                Clear();
                _disposed = true;
            }

            GC.SuppressFinalize(this);
        }

        ~ActivationCheckpointManager()
        {
            Dispose();
        }
    }
}
