using System.Collections.Generic;

namespace MLFramework.Distributed.FSDP
{
    /// <summary>
    /// Information about a parameter for sharding planning.
    /// </summary>
    public class ParameterInfo
    {
        /// <summary>Parameter name</summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>Parameter shape</summary>
        public long[] Shape { get; set; } = System.Array.Empty<long>();

        /// <summary>Size in bytes</summary>
        public long SizeBytes { get; set; }

        /// <summary>Layer name this parameter belongs to</summary>
        public string LayerName { get; set; } = string.Empty;

        /// <summary>Whether this parameter should always be gathered</summary>
        public bool AlwaysGather { get; set; }
    }

    /// <summary>
    /// Sharding plan describing how parameters are distributed.
    /// </summary>
    public class ShardingPlan
    {
        /// <summary>Total number of shards</summary>
        public int TotalShards { get; set; }

        /// <summary>Parameter assignments to shards</summary>
        public Dictionary<string, ShardingAssignment> Assignments { get; set; } = new();

        /// <summary>Parameters that should always be gathered</summary>
        public HashSet<string> AlwaysGathered { get; set; } = new();
    }

    /// <summary>
    /// Assignment of a parameter to a shard.
    /// </summary>
    public class ShardingAssignment
    {
        /// <summary>Parameter name</summary>
        public string ParameterName { get; set; } = string.Empty;

        /// <summary>Shard index</summary>
        public int ShardIndex { get; set; }

        /// <summary>Owner rank</summary>
        public int OwnerRank { get; set; }

        /// <summary>Offset in the parameter</summary>
        public long Offset { get; set; }

        /// <summary>Size of this shard</summary>
        public long Size { get; set; }
    }

    /// <summary>
    /// Interface for sharding strategies.
    /// </summary>
    public interface IShardingStrategy
    {
        /// <summary>
        /// Calculate sharding plan for given parameters.
        /// </summary>
        /// <param name="parameters">List of parameters to shard</param>
        /// <param name="worldSize">Number of devices</param>
        /// <returns>Sharding plan</returns>
        ShardingPlan CalculateShardingPlan(List<ParameterInfo> parameters, int worldSize);
    }

    /// <summary>
    /// Full sharding strategy: shards all parameters across all devices.
    /// </summary>
    public class FullShardingStrategy : IShardingStrategy
    {
        /// <inheritdoc/>
        public ShardingPlan CalculateShardingPlan(List<ParameterInfo> parameters, int worldSize)
        {
            var plan = new ShardingPlan { TotalShards = worldSize };

            foreach (var param in parameters)
            {
                if (param.AlwaysGather)
                {
                    plan.AlwaysGathered.Add(param.Name);
                    continue;
                }

                var assignment = new ShardingAssignment
                {
                    ParameterName = param.Name,
                    ShardIndex = 0, // Simplified
                    OwnerRank = 0,
                    Offset = 0,
                    Size = param.SizeBytes
                };

                plan.Assignments[param.Name] = assignment;
            }

            return plan;
        }
    }

    /// <summary>
    /// Layer-wise sharding strategy: shards individual layers.
    /// </summary>
    public class LayerWiseShardingStrategy : IShardingStrategy
    {
        /// <inheritdoc/>
        public ShardingPlan CalculateShardingPlan(List<ParameterInfo> parameters, int worldSize)
        {
            var plan = new ShardingPlan { TotalShards = worldSize };

            // Group parameters by layer
            var layers = new Dictionary<string, List<ParameterInfo>>();
            foreach (var param in parameters)
            {
                if (!layers.ContainsKey(param.LayerName))
                {
                    layers[param.LayerName] = new List<ParameterInfo>();
                }
                layers[param.LayerName].Add(param);
            }

            // Assign each layer to a device
            int currentRank = 0;
            foreach (var layerPair in layers)
            {
                foreach (var param in layerPair.Value)
                {
                    if (param.AlwaysGather)
                    {
                        plan.AlwaysGathered.Add(param.Name);
                        continue;
                    }

                    var assignment = new ShardingAssignment
                    {
                        ParameterName = param.Name,
                        ShardIndex = currentRank,
                        OwnerRank = currentRank,
                        Offset = 0,
                        Size = param.SizeBytes
                    };

                    plan.Assignments[param.Name] = assignment;
                }

                currentRank = (currentRank + 1) % worldSize;
            }

            return plan;
        }
    }

    /// <summary>
    /// Hybrid sharding strategy: mix of full and layer-wise sharding.
    /// </summary>
    public class HybridShardingStrategy : IShardingStrategy
    {
        private readonly HashSet<string> _fullShardedLayers;
        private readonly HashSet<string> _layerWiseShardedLayers;

        /// <summary>
        /// Create a hybrid sharding strategy.
        /// </summary>
        /// <param name="fullShardedLayers">Layers to shard fully</param>
        /// <param name="layerWiseShardedLayers">Layers to shard layer-wise</param>
        public HybridShardingStrategy(List<string> fullShardedLayers, List<string> layerWiseShardedLayers)
        {
            _fullShardedLayers = new HashSet<string>(fullShardedLayers ?? new List<string>());
            _layerWiseShardedLayers = new HashSet<string>(layerWiseShardedLayers ?? new List<string>());
        }

        /// <inheritdoc/>
        public ShardingPlan CalculateShardingPlan(List<ParameterInfo> parameters, int worldSize)
        {
            var plan = new ShardingPlan { TotalShards = worldSize };

            foreach (var param in parameters)
            {
                if (param.AlwaysGather)
                {
                    plan.AlwaysGathered.Add(param.Name);
                    continue;
                }

                // Determine sharding mode based on layer
                var shardingMode = _fullShardedLayers.Contains(param.LayerName)
                    ? ShardingMode.Full
                    : (_layerWiseShardedLayers.Contains(param.LayerName)
                        ? ShardingMode.LayerWise
                        : ShardingMode.Full); // Default to full

                var assignment = new ShardingAssignment
                {
                    ParameterName = param.Name,
                    ShardIndex = 0,
                    OwnerRank = 0,
                    Offset = 0,
                    Size = param.SizeBytes
                };

                plan.Assignments[param.Name] = assignment;
            }

            return plan;
        }

        private enum ShardingMode { Full, LayerWise }
    }
}
