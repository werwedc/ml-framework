using System;
using System.Collections.Generic;

namespace MLFramework.Distributed.FSDP
{
    /// <summary>
    /// Metadata class for FSDP checkpoints.
    /// Stores information about the checkpoint including model state, training progress, and configuration.
    /// </summary>
    public class FSDPCheckpointMetadata
    {
        /// <summary>Checkpoint version</summary>
        public int Version { get; set; } = 1;

        /// <summary>World size when checkpoint was created</summary>
        public int WorldSize { get; set; }

        /// <summary>Sharding strategy used</summary>
        public string ShardingStrategy { get; set; }

        /// <summary>Number of parameters</summary>
        public int NumParameters { get; set; }

        /// <summary>Timestamp of checkpoint</summary>
        public DateTime Timestamp { get; set; }

        /// <summary>Training epoch</summary>
        public int Epoch { get; set; }

        /// <summary>Training step</summary>
        public int Step { get; set; }

        /// <summary>Loss value</summary>
        public float Loss { get; set; }

        /// <summary>Mixed precision enabled</summary>
        public bool MixedPrecision { get; set; }

        /// <summary>CPU offloading enabled</summary>
        public bool CpuOffloading { get; set; }

        /// <summary>Parameter shapes</summary>
        public Dictionary<string, long[]> ParameterShapes { get; set; }

        /// <summary>
        /// Create a new checkpoint metadata instance.
        /// </summary>
        public FSDPCheckpointMetadata()
        {
            ParameterShapes = new Dictionary<string, long[]>();
        }
    }
}
