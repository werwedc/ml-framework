using System;

namespace MLFramework.Distributed.FSDP
{
    /// <summary>
    /// Sharding strategies for Fully Sharded Data Parallelism.
    /// </summary>
    public enum ShardingStrategy
    {
        /// <summary>Shard all parameters across all devices (maximum memory savings)</summary>
        Full,

        /// <summary>Shard individual layers sequentially (better for communication)</summary>
        LayerWise,

        /// <summary>Mix of full and layer-wise sharding</summary>
        Hybrid
    }

    /// <summary>
    /// Configuration class for FSDP (Fully Sharded Data Parallelism) settings.
    /// </summary>
    public class FSDPConfig
    {
        /// <summary>Sharding strategy to use</summary>
        public ShardingStrategy ShardingStrategy { get; set; } = ShardingStrategy.Full;

        /// <summary>Enable mixed precision (FP16/BF16)</summary>
        public bool MixedPrecision { get; set; } = true;

        /// <summary>Offload parameters/gradients to CPU when not in use</summary>
        public bool OffloadToCPU { get; set; } = false;

        /// <summary>Enable activation checkpointing</summary>
        public bool ActivationCheckpointing { get; set; } = false;

        /// <summary>Bucket size for gradient communication (in MB)</summary>
        public int BucketSizeMB { get; set; } = 25;

        /// <summary>Number of communication workers</summary>
        public int NumCommunicationWorkers { get; set; } = 2;

        /// <summary>
        /// Validate the configuration settings.
        /// </summary>
        /// <exception cref="ArgumentException">Thrown when configuration values are invalid.</exception>
        public void Validate()
        {
            if (BucketSizeMB <= 0 || BucketSizeMB > 1000)
            {
                throw new ArgumentException("BucketSizeMB must be between 1 and 1000", nameof(BucketSizeMB));
            }

            if (NumCommunicationWorkers <= 0 || NumCommunicationWorkers > 16)
            {
                throw new ArgumentException("NumCommunicationWorkers must be between 1 and 16", nameof(NumCommunicationWorkers));
            }
        }
    }

    /// <summary>
    /// State tracking class for sharded parameters in FSDP.
    /// </summary>
    public class FSDPState
    {
        /// <summary>Owner rank of this parameter shard</summary>
        public int OwnerRank { get; set; }

        /// <summary>Number of shards across all devices</summary>
        public int NumShards { get; set; }

        /// <summary>Local shard index</summary>
        public int ShardIndex { get; set; }

        /// <summary>Whether this shard is currently gathered on device</summary>
        public bool IsGathered { get; set; }

        /// <summary>Whether this shard is currently offloaded to CPU</summary>
        public bool IsOffloaded { get; set; }
    }
}
