using System;
using System.Collections.Generic;
using System.Linq;
using RitterFramework.Core.Tensor;
using MLFramework.HAL;
using MLFramework.NN;

namespace MLFramework.Pipeline
{
    /// <summary>
    /// Partitions a model into pipeline stages
    /// </summary>
    public class LayerPartitioner
    {
        private readonly PartitionMode _mode;
        private readonly int _numStages;
        private readonly IDevice[] _devices;

        /// <summary>
        /// Gets the partition mode
        /// </summary>
        public PartitionMode Mode => _mode;

        /// <summary>
        /// Gets the number of stages
        /// </summary>
        public int NumStages => _numStages;

        /// <summary>
        /// Creates a new layer partitioner
        /// </summary>
        public LayerPartitioner(
            PartitionMode mode,
            int numStages,
            IDevice[] devices = null)
        {
            _mode = mode;
            _numStages = numStages;

            if (numStages <= 0)
                throw new ArgumentException("Number of stages must be greater than 0", nameof(numStages));

            if (devices != null && devices.Length != numStages)
                throw new ArgumentException("Number of devices must match number of stages", nameof(devices));

            _devices = devices ?? Enumerable.Range(0, numStages)
                .Select(_ => Device.CPU)
                .ToArray();
        }

        /// <summary>
        /// Partitions a model into pipeline stages
        /// </summary>
        /// <param name="model">The model to partition</param>
        /// <param name="manualPartitions">
        /// For manual mode: List of layer indices for each stage
        /// Example: [[0,1,2], [3,4,5], [6,7,8,9]] for 3 stages
        /// </param>
        /// <returns>PartitionResult containing the stages and metadata</returns>
        public PartitionResult Partition(Module model, List<List<int>> manualPartitions = null)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            // Get child modules to partition
            var childModules = GetChildModules(model);
            if (childModules.Count == 0)
                throw new ArgumentException("Model has no child modules to partition");

            List<List<int>> partitions;

            switch (_mode)
            {
                case PartitionMode.Manual:
                    if (manualPartitions == null)
                        throw new ArgumentException("Manual mode requires manualPartitions parameter");

                    partitions = manualPartitions;
                    ValidateManualPartitions(partitions, childModules.Count);
                    break;

                case PartitionMode.Uniform:
                    partitions = UniformPartition(childModules.Count);
                    break;

                case PartitionMode.Automatic:
                    partitions = AutomaticPartition(model, childModules);
                    break;

                default:
                    throw new ArgumentException($"Unknown partition mode: {_mode}");
            }

            // Create pipeline stages from partitions
            var stages = CreateStages(model, partitions);
            var memoryPerStage = EstimateMemoryPerStage(model, partitions);
            var computationPerStage = EstimateComputationPerStage(model, partitions);
            var loadBalance = CalculateLoadBalance(memoryPerStage);

            return new PartitionResult(
                stages: stages,
                stageLayerIndices: partitions,
                memoryPerStage: memoryPerStage,
                computationPerStage: computationPerStage,
                loadBalance: loadBalance);
        }

        /// <summary>
        /// Gets child modules from a model
        /// </summary>
        private List<Module> GetChildModules(Module model)
        {
            // If model is a SequentialModule, return its modules
            if (model is SequentialModule sequential)
            {
                return sequential.GetModules().ToList();
            }

            // If model is a HierarchicalModule, get its children
            if (model is IHierarchicalModule hierarchical)
            {
                return hierarchical.Children().Cast<Module>().ToList();
            }

            // Otherwise, return empty (cannot partition non-hierarchical models)
            return new List<Module>();
        }

        /// <summary>
        /// Automatic partitioning based on memory estimation
        /// Uses greedy algorithm to balance memory usage across stages
        /// </summary>
        private List<List<int>> AutomaticPartition(Module model, List<Module> childModules)
        {
            int numLayers = childModules.Count;
            var partitions = new List<List<int>>();

            // Estimate memory for each layer
            var layerMemory = new long[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                layerMemory[i] = EstimateMemoryUsage(childModules[i]);
            }

            // Greedy partitioning: assign layers to stages to minimize max memory
            var stageMemory = new long[_numStages];
            for (int i = 0; i < _numStages; i++)
            {
                partitions.Add(new List<int>());
            }

            // Assign each layer to the stage with current minimum memory
            // But preserve order: layers must be assigned in sequence
            var currentStage = 0;
            var currentLayers = partitions[0];

            for (int i = 0; i < numLayers; i++)
            {
                // Check if moving to next stage would improve balance
                long memoryWithCurrentLayer = stageMemory[currentStage] + layerMemory[i];
                long memoryInNextStage = currentStage < _numStages - 1 ? stageMemory[currentStage + 1] : long.MaxValue;

                // If adding this layer would make current stage much larger than next, consider moving
                if (currentStage < _numStages - 1 && currentLayers.Count > 0 &&
                    memoryWithCurrentLayer > memoryInNextStage * 1.5)
                {
                    currentStage++;
                    currentLayers = partitions[currentStage];
                }

                // Add layer to current stage
                currentLayers.Add(i);
                stageMemory[currentStage] += layerMemory[i];

                // Move to next stage if current stage has enough layers
                // or if we're close to the end of the layers
                int minLayersPerStage = numLayers / _numStages;
                int maxLayersPerStage = minLayersPerStage + 1;
                if (currentStage < _numStages - 1 &&
                    currentLayers.Count >= maxLayersPerStage &&
                    remainingLayers(i, numLayers) >= (_numStages - currentStage - 1))
                {
                    currentStage++;
                    currentLayers = partitions[currentStage];
                }
            }

            // Ensure all stages have at least one layer if possible
            FixEmptyStages(partitions, numLayers);

            return partitions;
        }

        /// <summary>
        /// Gets remaining layers after current index
        /// </summary>
        private int remainingLayers(int currentIndex, int totalLayers)
        {
            return totalLayers - currentIndex - 1;
        }

        /// <summary>
        /// Ensures no stage is empty by redistributing layers
        /// </summary>
        private void FixEmptyStages(List<List<int>> partitions, int numLayers)
        {
            // Find empty stages and non-empty stages
            var emptyStages = new List<int>();
            var nonEmptyStages = new List<int>();

            for (int i = 0; i < partitions.Count; i++)
            {
                if (partitions[i].Count == 0)
                    emptyStages.Add(i);
                else
                    nonEmptyStages.Add(i);
            }

            // Redistribute one layer from non-empty stages to empty stages
            foreach (var emptyStage in emptyStages)
            {
                // Find the nearest non-empty stage
                int bestStage = -1;
                int bestDistance = int.MaxValue;

                foreach (var nonEmptyStage in nonEmptyStages)
                {
                    int distance = Math.Abs(nonEmptyStage - emptyStage);
                    if (distance < bestDistance)
                    {
                        bestDistance = distance;
                        bestStage = nonEmptyStage;
                    }
                }

                if (bestStage != -1 && partitions[bestStage].Count > 1)
                {
                    // Move one layer from bestStage to emptyStage
                    if (emptyStage > bestStage)
                    {
                        // Move the last layer of bestStage to emptyStage
                        int layerIndex = partitions[bestStage].Last();
                        partitions[bestStage].RemoveAt(partitions[bestStage].Count - 1);
                        partitions[emptyStage].Add(layerIndex);
                    }
                    else
                    {
                        // Move the first layer of bestStage to emptyStage
                        int layerIndex = partitions[bestStage][0];
                        partitions[bestStage].RemoveAt(0);
                        partitions[emptyStage].Add(layerIndex);
                    }
                }
            }
        }

        /// <summary>
        /// Uniform partitioning: evenly distribute layers
        /// </summary>
        private List<List<int>> UniformPartition(int numLayers)
        {
            var partitions = new List<List<int>>();

            int layersPerStage = numLayers / _numStages;
            int remainder = numLayers % _numStages;

            int currentLayer = 0;
            for (int stage = 0; stage < _numStages; stage++)
            {
                var stageLayers = new List<int>();
                int layersInThisStage = layersPerStage + (stage < remainder ? 1 : 0);

                for (int i = 0; i < layersInThisStage; i++)
                {
                    stageLayers.Add(currentLayer);
                    currentLayer++;
                }

                partitions.Add(stageLayers);
            }

            return partitions;
        }

        /// <summary>
        /// Creates pipeline stages from partition indices
        /// </summary>
        private List<PipelineStage> CreateStages(Module model, List<List<int>> partitions)
        {
            var stages = new List<PipelineStage>();
            var childModules = GetChildModules(model);

            for (int stageIndex = 0; stageIndex < partitions.Count; stageIndex++)
            {
                var layerIndices = partitions[stageIndex];
                var stageModule = CreateStageModule(model, layerIndices);
                var stage = new PipelineStage(
                    module: stageModule,
                    rank: stageIndex,
                    totalStages: _numStages,
                    device: _devices[stageIndex]);
                stages.Add(stage);
            }

            return stages;
        }

        /// <summary>
        /// Creates a SequentialModule from a list of layer indices
        /// </summary>
        private Module CreateStageModule(Module model, List<int> layerIndices)
        {
            var childModules = GetChildModules(model);
            var stageLayers = new List<Module>();

            foreach (int layerIndex in layerIndices)
            {
                if (layerIndex < 0 || layerIndex >= childModules.Count)
                    throw new ArgumentOutOfRangeException(nameof(layerIndex),
                        $"Layer index {layerIndex} is out of range [0, {childModules.Count - 1}]");

                stageLayers.Add(childModules[layerIndex]);
            }

            var sequentialModule = new SequentialModule($"Stage_{layerIndices[0]}-{layerIndices[layerIndices.Count - 1]}");
            foreach (var layer in stageLayers)
            {
                sequentialModule.Add(layer);
            }

            return sequentialModule;
        }

        /// <summary>
        /// Estimates memory usage for a set of layers
        /// </summary>
        private long EstimateMemoryUsage(Module module)
        {
            // Simple estimation: count parameters * 4 bytes (float32)
            // This is a rough approximation - actual memory depends on activations, gradients, etc.
            long parameterCount = 0;
            foreach (var param in module.GetParameters())
            {
                // Get tensor size
                var tensor = param as Tensor;
                if (tensor != null)
                {
                    parameterCount += tensor.Size;
                }
            }

            return parameterCount * 4; // 4 bytes per float32
        }

        /// <summary>
        /// Estimates memory per stage
        /// </summary>
        private long[] EstimateMemoryPerStage(Module model, List<List<int>> partitions)
        {
            var childModules = GetChildModules(model);
            var memoryPerStage = new long[partitions.Count];

            for (int stage = 0; stage < partitions.Count; stage++)
            {
                long stageMemory = 0;
                foreach (int layerIndex in partitions[stage])
                {
                    stageMemory += EstimateMemoryUsage(childModules[layerIndex]);
                }
                memoryPerStage[stage] = stageMemory;
            }

            return memoryPerStage;
        }

        /// <summary>
        /// Estimates computation time for a set of layers (in FLOPs)
        /// </summary>
        private float EstimateComputationTime(Module module)
        {
            // Simple estimation based on parameter count
            // This is a very rough approximation
            float flops = 0;
            foreach (var param in module.GetParameters())
            {
                var tensor = param as Tensor;
                if (tensor != null)
                {
                    flops += tensor.Size * 2; // Assume 2 FLOPs per parameter
                }
            }

            return flops;
        }

        /// <summary>
        /// Estimates computation per stage
        /// </summary>
        private float[] EstimateComputationPerStage(Module model, List<List<int>> partitions)
        {
            var childModules = GetChildModules(model);
            var computationPerStage = new float[partitions.Count];

            for (int stage = 0; stage < partitions.Count; stage++)
            {
                float stageComputation = 0;
                foreach (int layerIndex in partitions[stage])
                {
                    stageComputation += EstimateComputationTime(childModules[layerIndex]);
                }
                computationPerStage[stage] = stageComputation;
            }

            return computationPerStage;
        }

        /// <summary>
        /// Calculates load balance metric
        /// </summary>
        private float CalculateLoadBalance(long[] memoryPerStage)
        {
            if (memoryPerStage.Length == 0)
                return 1.0f;

            long maxMemory = memoryPerStage.Max();
            long avgMemory = memoryPerStage.Sum() / memoryPerStage.Length;

            if (avgMemory == 0)
                return 1.0f;

            return (float)maxMemory / avgMemory;
        }

        /// <summary>
        /// Validates manual partitions
        /// </summary>
        private void ValidateManualPartitions(List<List<int>> partitions, int numLayers)
        {
            if (partitions.Count != _numStages)
            {
                throw new ArgumentException(
                    $"Manual partitions must have exactly {_numStages} entries, got {partitions.Count}",
                    nameof(partitions));
            }

            // Check that all layers are assigned exactly once
            var assignedLayers = new HashSet<int>();
            for (int stage = 0; stage < partitions.Count; stage++)
            {
                var stageLayers = partitions[stage];

                // Check that layers are in order
                for (int i = 1; i < stageLayers.Count; i++)
                {
                    if (stageLayers[i] <= stageLayers[i - 1])
                    {
                        throw new ArgumentException(
                            $"Layers in manual partition must be in order, found {stageLayers[i - 1]} before {stageLayers[i]} in stage {stage}",
                            nameof(partitions));
                    }
                }

                // Check for gaps between stages
                if (stage > 0 && stageLayers.Count > 0 && partitions[stage - 1].Count > 0)
                {
                    int lastLayerInPrevStage = partitions[stage - 1].Last();
                    int firstLayerInThisStage = stageLayers[0];
                    if (firstLayerInThisStage != lastLayerInPrevStage + 1)
                    {
                        throw new ArgumentException(
                            $"Gap found between stages {stage - 1} and {stage}: layer {lastLayerInPrevStage + 1} is missing",
                            nameof(partitions));
                    }
                }

                // Check that layer indices are in valid range
                foreach (int layerIndex in stageLayers)
                {
                    if (layerIndex < 0 || layerIndex >= numLayers)
                    {
                        throw new ArgumentOutOfRangeException(
                            nameof(partitions),
                            $"Layer index {layerIndex} is out of range [0, {numLayers - 1}]");
                    }

                    if (!assignedLayers.Add(layerIndex))
                    {
                        throw new ArgumentException(
                            $"Layer {layerIndex} is assigned to multiple stages",
                            nameof(partitions));
                    }
                }
            }

            // Check that all layers are assigned
            for (int i = 0; i < numLayers; i++)
            {
                if (!assignedLayers.Contains(i))
                {
                    throw new ArgumentException(
                        $"Layer {i} is not assigned to any stage",
                        nameof(partitions));
                }
            }
        }
    }
}
