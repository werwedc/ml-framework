using System;
using System.Collections.Generic;

namespace MLFramework.Pipeline
{
    /// <summary>
    /// Result of model partitioning
    /// </summary>
    public class PartitionResult
    {
        /// <summary>
        /// List of pipeline stages
        /// </summary>
        public List<PipelineStage> Stages { get; }

        /// <summary>
        /// Layer indices assigned to each stage
        /// </summary>
        public List<List<int>> StageLayerIndices { get; }

        /// <summary>
        /// Estimated memory per stage (in bytes)
        /// </summary>
        public long[] MemoryPerStage { get; }

        /// <summary>
        /// Estimated computation per stage (in FLOPs)
        /// </summary>
        public float[] ComputationPerStage { get; }

        /// <summary>
        /// Load balance metric (lower is better, 1.0 is perfectly balanced)
        /// Calculated as max(memory) / avg(memory) for memory balance
        /// </summary>
        public float LoadBalance { get; }

        /// <summary>
        /// Creates a new partition result
        /// </summary>
        public PartitionResult(
            List<PipelineStage> stages,
            List<List<int>> stageLayerIndices,
            long[] memoryPerStage,
            float[] computationPerStage,
            float loadBalance)
        {
            Stages = stages ?? throw new ArgumentNullException(nameof(stages));
            StageLayerIndices = stageLayerIndices ?? throw new ArgumentNullException(nameof(stageLayerIndices));
            MemoryPerStage = memoryPerStage ?? throw new ArgumentNullException(nameof(memoryPerStage));
            ComputationPerStage = computationPerStage ?? throw new ArgumentNullException(nameof(computationPerStage));

            if (Stages.Count != StageLayerIndices.Count)
                throw new ArgumentException("Stages and StageLayerIndices must have the same count");

            if (Stages.Count != MemoryPerStage.Length)
                throw new ArgumentException("Stages and MemoryPerStage must have the same count");

            if (Stages.Count != ComputationPerStage.Length)
                throw new ArgumentException("Stages and ComputationPerStage must have the same count");

            LoadBalance = loadBalance;
        }
    }
}
