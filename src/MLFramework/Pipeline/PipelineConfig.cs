using System;
using MLFramework.HAL;

namespace MLFramework.Pipeline
{
    /// <summary>
    /// Configuration for pipeline parallelism setup
    /// </summary>
    public class PipelineConfig
    {
        private int _numStages;
        private int _microBatches = 4;
        private IDevice[]? _devices;

        /// <summary>
        /// Number of pipeline stages (must be <= number of available devices)
        /// </summary>
        public int NumStages
        {
            get => _numStages;
            set => _numStages = value;
        }

        /// <summary>
        /// Number of micro-batches to split each batch into
        /// </summary>
        public int MicroBatches
        {
            get => _microBatches;
            set => _microBatches = value;
        }

        /// <summary>
        /// Devices to use for each stage (must be length NumStages)
        /// If null, uses first NumStages available devices
        /// </summary>
        public IDevice[]? Devices
        {
            get => _devices;
            set => _devices = value;
        }

        /// <summary>
        /// Validate configuration
        /// </summary>
        public void Validate()
        {
            if (_numStages <= 0)
            {
                throw new ArgumentException("NumStages must be greater than 0", nameof(NumStages));
            }

            if (_microBatches <= 0)
            {
                throw new ArgumentException("MicroBatches must be greater than 0", nameof(MicroBatches));
            }

            if (_devices != null && _devices.Length != _numStages)
            {
                throw new ArgumentException(
                    $"Devices array length ({_devices.Length}) must equal NumStages ({_numStages})",
                    nameof(Devices));
            }

            // Note: Validating device availability is beyond the scope of this basic implementation
            // In a production environment, you would check if devices are available here
        }
    }
}
