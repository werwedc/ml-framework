using System;

namespace MLFramework.Distributed.FSDP
{
    /// <summary>
    /// Configuration class for FSDP checkpointing.
    /// </summary>
    public class FSDPCheckpointConfig
    {
        /// <summary>Checkpoint directory</summary>
        public string CheckpointDir { get; set; }

        /// <summary>Checkpoint file prefix</summary>
        public string FilePrefix { get; set; } = "fsdp_checkpoint";

        /// <summary>Whether to save optimizer states</summary>
        public bool SaveOptimizerStates { get; set; } = true;

        /// <summary>Whether to save training state (epoch, step)</summary>
        public bool SaveTrainingState { get; set; } = true;

        /// <summary>Whether to use async checkpointing</summary>
        public bool AsyncCheckpoint { get; set; } = false;

        /// <summary>Checkpoint format (binary, json, torch)</summary>
        public string CheckpointFormat { get; set; } = "binary";

        /// <summary>Maximum number of checkpoints to keep</summary>
        public int MaxCheckpoints { get; set; } = 5;

        /// <summary>
        /// Validate configuration settings.
        /// </summary>
        /// <exception cref="ArgumentException">Thrown when configuration values are invalid.</exception>
        public void Validate()
        {
            if (string.IsNullOrEmpty(CheckpointDir))
                throw new ArgumentException("CheckpointDir cannot be empty", nameof(CheckpointDir));

            if (MaxCheckpoints < 1)
                throw new ArgumentException("MaxCheckpoints must be at least 1", nameof(MaxCheckpoints));
        }
    }
}
