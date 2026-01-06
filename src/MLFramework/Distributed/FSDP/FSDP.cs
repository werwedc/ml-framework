using MLFramework.Distributed;
using System;
using System.Collections.Generic;

namespace MLFramework.Distributed.FSDP
{
    /// <summary>
    /// Interface for a model that can be wrapped by FSDP.
    /// </summary>
    public interface IModel
    {
        /// <summary>Model name</summary>
        string Name { get; }
    }

    /// <summary>
    /// FSDP (Fully Sharded Data Parallel) wrapper for models.
    /// Enables training of large models that don't fit in GPU memory.
    /// </summary>
    public class FSDP : IDisposable
    {
        private readonly IModel _model;
        private readonly FSDPConfig _config;
        private readonly IProcessGroup _processGroup;
        private readonly List<FSDPShardingUnit> _shardingUnits;
        private bool _disposed;

        /// <summary>
        /// The wrapped model.
        /// </summary>
        public IModel Model => _model;

        /// <summary>
        /// The FSDP configuration.
        /// </summary>
        public FSDPConfig Config => _config;

        /// <summary>
        /// The process group for distributed training.
        /// </summary>
        public IProcessGroup ProcessGroup => _processGroup;

        /// <summary>
        /// Create an FSDP wrapper with explicit process group.
        /// </summary>
        /// <param name="model">Model to wrap</param>
        /// <param name="config">FSDP configuration</param>
        /// <param name="processGroup">Process group for communication</param>
        public FSDP(IModel model, FSDPConfig config, IProcessGroup processGroup)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _processGroup = processGroup ?? throw new ArgumentNullException(nameof(processGroup));

            _config.Validate();

            if (_processGroup.WorldSize == 0)
            {
                throw new InvalidOperationException("Process group world size must be at least 1");
            }

            _shardingUnits = new List<FSDPShardingUnit>();

            // Sharding setup will be implemented in forward hook spec
        }

        /// <summary>
        /// Create an FSDP wrapper using default process group.
        /// </summary>
        /// <param name="model">Model to wrap</param>
        /// <param name="config">FSDP configuration</param>
        public FSDP(IModel model, FSDPConfig config)
        {
            var defaultProcessGroup = MLFramework.Distributed.ProcessGroup.Default;
            if (defaultProcessGroup == null)
            {
                throw new InvalidOperationException("No default process group initialized. Call ProcessGroup.Init() first or use the constructor that accepts an IProcessGroup.");
            }
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _processGroup = defaultProcessGroup;

            _config.Validate();

            if (_processGroup.WorldSize == 0)
            {
                throw new InvalidOperationException("Process group world size must be at least 1");
            }

            _shardingUnits = new List<FSDPShardingUnit>();

            // Sharding setup will be implemented in forward hook spec
        }

        /// <summary>
        /// Get all sharding units managed by this FSDP instance.
        /// </summary>
        /// <returns>List of sharding units</returns>
        public IReadOnlyList<FSDPShardingUnit> GetShardingUnits()
        {
            return _shardingUnits.AsReadOnly();
        }

        /// <summary>
        /// Dispose of resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Protected implementation of dispose pattern.
        /// </summary>
        /// <param name="disposing">Whether managed resources should be disposed</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Dispose all sharding units
                    foreach (var unit in _shardingUnits)
                    {
                        unit.Dispose();
                    }
                    _shardingUnits.Clear();
                }
                _disposed = true;
            }
        }
    }
}
