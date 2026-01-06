using MLFramework.Distributed;
using RitterFramework.Core.Tensor;
using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.Distributed.FSDP
{
    /// <summary>
    /// Interface for a model that can be wrapped by FSDP.
    /// </summary>
    public interface IModel
    {
        /// <summary>Model name</summary>
        string Name { get; }

        /// <summary>Forward pass</summary>
        Tensor Forward(Tensor input);

        /// <summary>Backward pass</summary>
        void Backward();

        /// <summary>Get model parameters</summary>
        List<NamedTensor> GetParameters();
    }

    /// <summary>
    /// FSDP (Fully Sharded Data Parallel) wrapper for models.
    /// Enables training of large models that don't fit in GPU memory.
    /// </summary>
    public class FSDP : IModel, IDisposable
    {
        private readonly IModel _model;
        private readonly IProcessGroup _processGroup;
        private readonly FSDPConfig _config;
        private readonly IShardingStrategy _shardingStrategy;
        private readonly Dictionary<string, FSDPShardingUnit> _shardingUnits;
        private readonly ShardingPlan _shardingPlan;
        private bool _disposed;

        /// <summary>
        /// The wrapped model.
        /// </summary>
        public IModel Model => _model;

        /// <summary>
        /// Model name.
        /// </summary>
        public string Name => _model.Name;

        /// <summary>
        /// The FSDP configuration.
        /// </summary>
        public FSDPConfig Config => _config;

        /// <summary>
        /// The process group for distributed training.
        /// </summary>
        public IProcessGroup ProcessGroup => _processGroup;

        /// <summary>
        /// Wrap a model with FSDP for distributed training.
        /// </summary>
        /// <param name="model">The model to wrap</param>
        /// <param name="config">FSDP configuration</param>
        /// <param name="processGroup">Process group for communication</param>
        public FSDP(IModel model, FSDPConfig config, IProcessGroup? processGroup = null)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _config = config ?? throw new ArgumentNullException(nameof(config));

            // Validate configuration
            _config.Validate();

            // Use default process group if not provided
            _processGroup = processGroup ?? MLFramework.Distributed.ProcessGroup.Default;
            if (_processGroup == null)
                throw new InvalidOperationException("No process group available. Call ProcessGroup.Init() first.");

            // Validate world size
            if (_processGroup.WorldSize == 1 && _config.ShardingStrategy == ShardingStrategy.Full)
            {
                // Warn but allow for single-device testing
                // In production, FSDP requires multiple devices
            }

            // Create sharding strategy
            _shardingStrategy = CreateShardingStrategy(_config.ShardingStrategy);

            // Collect parameter information
            var parameters = CollectParameterInfo(_model);

            // Calculate sharding plan
            _shardingPlan = _shardingStrategy.CalculateShardingPlan(parameters, _processGroup.WorldSize);

            // Create sharding units
            _shardingUnits = new Dictionary<string, FSDPShardingUnit>();
            foreach (var param in parameters)
            {
                if (!_shardingPlan.AlwaysGathered.Contains(param.Name))
                {
                    var shardingUnit = CreateShardingUnit(param, _model);
                    _shardingUnits[param.Name] = shardingUnit;
                }
            }

            // Register hooks
            RegisterForwardHooks();
            RegisterBackwardHooks();
        }

        /// <summary>
        /// Forward pass through the model.
        /// </summary>
        /// <param name="input">Input tensor</param>
        /// <returns>Output tensor</returns>
        public Tensor Forward(Tensor input)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(FSDP));

            // Forward pass will trigger hooks that gather parameters
            return _model.Forward(input);
        }

        /// <summary>
        /// Backward pass (computes gradients).
        /// </summary>
        public void Backward()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(FSDP));

            // Backward pass will trigger hooks that scatter gradients
            _model.Backward();
        }

        /// <summary>
        /// Get model parameters.
        /// </summary>
        /// <returns>List of parameter tensors</returns>
        public List<NamedTensor> GetParameters()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(FSDP));

            // Return sharded parameters
            var parameters = new List<NamedTensor>();
            foreach (var unit in _shardingUnits.Values)
            {
                if (unit.ShardedParameter != null)
                {
                    parameters.Add(new NamedTensor(unit.ParameterName, unit.ShardedParameter));
                }
            }
            return parameters;
        }

        /// <summary>
        /// Get model gradients.
        /// </summary>
        /// <returns>List of gradient tensors</returns>
        public List<Tensor> GetGradients()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(FSDP));

            // Return sharded gradients
            var gradients = new List<Tensor>();
            foreach (var unit in _shardingUnits.Values)
            {
                if (unit.LocalGradient != null)
                    gradients.Add(unit.LocalGradient);
            }
            return gradients;
        }

        /// <summary>
        /// Collect parameter information from the model.
        /// </summary>
        private List<ParameterInfo> CollectParameterInfo(IModel model)
        {
            var parameters = new List<ParameterInfo>();
            var modelParameters = model.GetParameters();

            foreach (var param in modelParameters)
            {
                var paramInfo = new ParameterInfo
                {
                    Name = param.Name ?? $"param_{parameters.Count}",
                    Shape = param.Tensor.Shape.Select(x => (long)x).ToArray(),
                    SizeBytes = param.Tensor.Size * 4, // Assume float32
                    LayerName = InferLayerName(param.Name),
                    AlwaysGather = ShouldAlwaysGather(param.Name)
                };
                parameters.Add(paramInfo);
            }

            return parameters;
        }

        /// <summary>
        /// Infer layer name from parameter name.
        /// </summary>
        private string InferLayerName(string paramName)
        {
            if (string.IsNullOrEmpty(paramName))
                return "layer_0";

            // Simple heuristic: extract layer name from parameter name
            // e.g., "transformer.layer1.weight" -> "transformer.layer1"
            var parts = paramName.Split('.');
            if (parts.Length >= 2)
            {
                return string.Join(".", parts.Take(parts.Length - 1));
            }
            return parts[0];
        }

        /// <summary>
        /// Determine if a parameter should always be gathered.
        /// </summary>
        private bool ShouldAlwaysGather(string paramName)
        {
            if (string.IsNullOrEmpty(paramName))
                return false;

            // Embeddings should always be gathered for simplicity
            if (paramName.Contains("embedding", StringComparison.OrdinalIgnoreCase))
                return true;

            return false;
        }

        /// <summary>
        /// Create a sharding unit for a parameter.
        /// </summary>
        private FSDPShardingUnit CreateShardingUnit(ParameterInfo paramInfo, IModel model)
        {
            var modelParams = model.GetParameters();
            var param = modelParams.FirstOrDefault(p => p.Name == paramInfo.Name);

            if (param == null)
                throw new ArgumentException($"Parameter {paramInfo.Name} not found in model");

            return new FSDPShardingUnit(paramInfo.Name, param.Tensor, _processGroup);
        }

        /// <summary>
        /// Create a sharding strategy based on the configuration.
        /// </summary>
        private IShardingStrategy CreateShardingStrategy(ShardingStrategy strategy)
        {
            return strategy switch
            {
                ShardingStrategy.Full => new FullShardingStrategy(),
                ShardingStrategy.LayerWise => new LayerWiseShardingStrategy(),
                ShardingStrategy.Hybrid => new HybridShardingStrategy(new List<string>(), new List<string>()),
                _ => throw new ArgumentException($"Unknown sharding strategy: {strategy}")
            };
        }

        /// <summary>
        /// Register forward hooks to gather parameters.
        /// </summary>
        private void RegisterForwardHooks()
        {
            // This will be implemented in spec_fsdp_forward_hook.md
            // For now, just mark as not implemented
        }

        /// <summary>
        /// Register backward hooks to scatter gradients.
        /// </summary>
        private void RegisterBackwardHooks()
        {
            // This will be implemented in spec_fsdp_backward_hook.md
            // For now, just mark as not implemented
        }

        /// <summary>
        /// Get all sharding units managed by this FSDP instance.
        /// </summary>
        /// <returns>List of sharding units</returns>
        public IReadOnlyList<FSDPShardingUnit> GetShardingUnits()
        {
            return _shardingUnits.Values.ToList().AsReadOnly();
        }

        /// <summary>
        /// Dispose of resources.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
                return;

            foreach (var unit in _shardingUnits.Values)
            {
                unit.Dispose();
            }

            _shardingUnits.Clear();
            _disposed = true;
        }
    }
}
