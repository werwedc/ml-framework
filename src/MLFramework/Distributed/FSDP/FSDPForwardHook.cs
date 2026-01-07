using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using MLFramework.Distributed;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MLFramework.Distributed.FSDP
{
    /// <summary>
    /// Context information passed to hooks.
    /// </summary>
    public class HookContext
    {
        /// <summary>Layer name</summary>
        public string LayerName { get; set; } = string.Empty;

        /// <summary>Input tensor</summary>
        public Tensor? Input { get; set; }

        /// <summary>Output tensor</summary>
        public Tensor? Output { get; set; }

        /// <summary>Whether this is the forward pass</summary>
        public bool IsForward { get; set; }

        /// <summary>Current step in the pass</summary>
        public int Step { get; set; }

        /// <summary>
        /// Create a new hook context.
        /// </summary>
        public HookContext(string layerName, Tensor input, bool isForward)
        {
            LayerName = layerName;
            Input = input;
            IsForward = isForward;
            Step = 0;
        }
    }

    /// <summary>
    /// Manages forward hooks for FSDP to automatically gather parameters
    /// before layer execution and release them after.
    /// </summary>
    public class FSDPForwardHook : IDisposable
    {
        private readonly FSDP _fsdp;
        private readonly IProcessGroup _processGroup;
        private readonly Dictionary<string, AllGatherOperation> _gatherOperations;
        private readonly Dictionary<string, Action> _preForwardHooks;
        private readonly Dictionary<string, Action> _postForwardHooks;
        private bool _disposed;

        /// <summary>
        /// Initialize forward hooks for FSDP.
        /// </summary>
        /// <param name="fsdp">FSDP wrapper instance</param>
        public FSDPForwardHook(FSDP fsdp)
        {
            _fsdp = fsdp ?? throw new ArgumentNullException(nameof(fsdp));
            _processGroup = fsdp.ProcessGroup;
            _gatherOperations = new Dictionary<string, AllGatherOperation>();
            _preForwardHooks = new Dictionary<string, Action>();
            _postForwardHooks = new Dictionary<string, Action>();
        }

        /// <summary>
        /// Register forward hooks for all sharding units.
        /// </summary>
        /// <param name="shardingUnits">Dictionary of parameter name to sharding unit</param>
        public void RegisterHooks(Dictionary<string, FSDPShardingUnit> shardingUnits)
        {
            if (shardingUnits == null || shardingUnits.Count == 0)
                return;

            foreach (var kvp in shardingUnits)
            {
                var paramName = kvp.Key;
                var shardingUnit = kvp.Value;

                // Register pre-forward hook to gather parameter
                RegisterPreForwardHook(paramName, shardingUnit);

                // Register post-forward hook to release parameter
                RegisterPostForwardHook(paramName, shardingUnit);
            }
        }

        /// <summary>
        /// Register a pre-forward hook to gather a parameter.
        /// </summary>
        private void RegisterPreForwardHook(string paramName, FSDPShardingUnit shardingUnit)
        {
            // Create All-Gather operation for this parameter
            var fullShape = new long[shardingUnit.Shape.Length];
            for (int i = 0; i < shardingUnit.Shape.Length; i++)
            {
                fullShape[i] = shardingUnit.Shape[i];
            }

            var allGatherOp = new AllGatherOperation(
                _processGroup,
                fullShape,
                shardingUnit.DataType,
                shardingUnit.State.ShardIndex
            );
            _gatherOperations[paramName] = allGatherOp;

            // Create pre-forward hook action
            var hook = () =>
            {
                GatherParameter(shardingUnit, allGatherOp);
            };
            _preForwardHooks[paramName] = hook;
        }

        /// <summary>
        /// Register a post-forward hook to release a parameter.
        /// </summary>
        private void RegisterPostForwardHook(string paramName, FSDPShardingUnit shardingUnit)
        {
            // Create post-forward hook action
            var hook = () =>
            {
                ReleaseParameter(shardingUnit);
            };
            _postForwardHooks[paramName] = hook;
        }

        /// <summary>
        /// Get the pre-forward hook for a parameter.
        /// </summary>
        public Action GetPreForwardHook(string paramName)
        {
            if (_preForwardHooks.TryGetValue(paramName, out var hook))
            {
                return hook;
            }
            throw new ArgumentException($"No pre-forward hook registered for parameter {paramName}", nameof(paramName));
        }

        /// <summary>
        /// Get the post-forward hook for a parameter.
        /// </summary>
        public Action GetPostForwardHook(string paramName)
        {
            if (_postForwardHooks.TryGetValue(paramName, out var hook))
            {
                return hook;
            }
            throw new ArgumentException($"No post-forward hook registered for parameter {paramName}", nameof(paramName));
        }

        /// <summary>
        /// Gather a parameter from all devices.
        /// </summary>
        private void GatherParameter(FSDPShardingUnit shardingUnit, AllGatherOperation allGatherOp)
        {
            if (shardingUnit.ShardedParameter == null)
                throw new InvalidOperationException($"Sharded parameter is null for {shardingUnit.ParameterName}");

            // Perform All-Gather
            var gatheredParam = allGatherOp.AllGather(shardingUnit.ShardedParameter);

            // Store gathered parameter in sharding unit
            shardingUnit.GatheredParameter = gatheredParam;
            shardingUnit.State.IsGathered = true;
        }

        /// <summary>
        /// Release a gathered parameter to free memory.
        /// </summary>
        private void ReleaseParameter(FSDPShardingUnit shardingUnit)
        {
            if (shardingUnit.GatheredParameter == null)
                return;

            // Release gathered parameter
            shardingUnit.ReleaseGatheredParameters();
        }

        /// <summary>
        /// Gather multiple parameters in parallel.
        /// </summary>
        /// <param name="shardingUnits">Sharding units to gather</param>
        public async Task GatherMultipleAsync(Dictionary<string, FSDPShardingUnit> shardingUnits)
        {
            if (shardingUnits == null || shardingUnits.Count == 0)
                return;

            var tasks = shardingUnits.Values.Select(unit =>
            {
                if (unit.ShardedParameter == null)
                    return Task.CompletedTask;

                if (!_gatherOperations.TryGetValue(unit.ParameterName, out var op))
                    return Task.CompletedTask;

                return Task.Run(() =>
                {
                    var gatheredParam = op.AllGather(unit.ShardedParameter);
                    unit.GatheredParameter = gatheredParam;
                    unit.State.IsGathered = true;
                });
            }).ToList();

            await Task.WhenAll(tasks);
        }

        /// <summary>
        /// Release multiple gathered parameters.
        /// </summary>
        /// <param name="shardingUnits">Sharding units to release</param>
        public void ReleaseMultiple(Dictionary<string, FSDPShardingUnit> shardingUnits)
        {
            if (shardingUnits == null || shardingUnits.Count == 0)
                return;

            foreach (var unit in shardingUnits.Values)
            {
                unit.ReleaseGatheredParameters();
            }
        }

        /// <summary>
        /// Get all registered gather operations.
        /// </summary>
        public IReadOnlyDictionary<string, AllGatherOperation> GetGatherOperations()
        {
            return _gatherOperations;
        }

        /// <summary>
        /// Check if a pre-forward hook is registered for a parameter.
        /// </summary>
        public bool HasPreForwardHook(string paramName)
        {
            return _preForwardHooks.ContainsKey(paramName);
        }

        /// <summary>
        /// Check if a post-forward hook is registered for a parameter.
        /// </summary>
        public bool HasPostForwardHook(string paramName)
        {
            return _postForwardHooks.ContainsKey(paramName);
        }

        /// <summary>
        /// Clear all registered hooks.
        /// </summary>
        public void ClearHooks()
        {
            _preForwardHooks.Clear();
            _postForwardHooks.Clear();
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
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    foreach (var op in _gatherOperations.Values)
                    {
                        // AllGatherOperation doesn't implement IDisposable, so no action needed
                    }
                    _gatherOperations.Clear();
                    ClearHooks();
                }
                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer for FSDPForwardHook.
        /// </summary>
        ~FSDPForwardHook()
        {
            Dispose(false);
        }
    }
}
