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
    /// Manages backward hooks for FSDP to automatically scatter gradients
    /// after gradient computation.
    /// </summary>
    public class FSDPBackwardHook : IDisposable
    {
        private readonly FSDP _fsdp;
        private readonly IProcessGroup _processGroup;
        private readonly Dictionary<string, ReduceScatterOperation> _scatterOperations;
        private readonly Dictionary<string, Func<Tensor, Tensor>> _backwardHooks;
        private bool _disposed;

        /// <summary>
        /// Initialize backward hooks for FSDP.
        /// </summary>
        /// <param name="fsdp">FSDP wrapper instance</param>
        public FSDPBackwardHook(FSDP fsdp)
        {
            _fsdp = fsdp ?? throw new ArgumentNullException(nameof(fsdp));
            _processGroup = fsdp.ProcessGroup;
            _scatterOperations = new Dictionary<string, ReduceScatterOperation>();
            _backwardHooks = new Dictionary<string, Func<Tensor, Tensor>>();
        }

        /// <summary>
        /// Register backward hooks for all sharding units.
        /// </summary>
        /// <param name="model">The model to register hooks on</param>
        /// <param name="shardingUnits">Dictionary of parameter name to sharding unit</param>
        public void RegisterHooks(IModel model, Dictionary<string, FSDPShardingUnit> shardingUnits)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            if (shardingUnits == null || shardingUnits.Count == 0)
                return;

            foreach (var kvp in shardingUnits)
            {
                var paramName = kvp.Key;
                var shardingUnit = kvp.Value;

                // Register backward hook to scatter gradients
                RegisterBackwardHook(model, paramName, shardingUnit);
            }
        }

        /// <summary>
        /// Register a backward hook to scatter gradients.
        /// </summary>
        private void RegisterBackwardHook(IModel model, string paramName, FSDPShardingUnit shardingUnit)
        {
            // Create Reduce-Scatter operation for this parameter
            var fullShape = new long[shardingUnit.Shape.Length];
            for (int i = 0; i < shardingUnit.Shape.Length; i++)
            {
                fullShape[i] = shardingUnit.Shape[i];
            }

            var reduceScatterOp = new ReduceScatterOperation(
                _processGroup,
                fullShape,
                shardingUnit.DataType,
                shardingUnit.State.ShardIndex,
                ReduceOp.Sum
            );
            _scatterOperations[paramName] = reduceScatterOp;

            // Create backward hook function
            var hook = new Func<Tensor, Tensor>((gradient) =>
            {
                // Scatter gradient after backward pass
                return ScatterGradient(shardingUnit, reduceScatterOp, gradient);
            });
            _backwardHooks[paramName] = hook;
        }

        /// <summary>
        /// Get the backward hook for a parameter.
        /// </summary>
        /// <param name="paramName">Parameter name</param>
        /// <returns>Backward hook function</returns>
        public Func<Tensor, Tensor>? GetBackwardHook(string paramName)
        {
            if (_backwardHooks.TryGetValue(paramName, out var hook))
            {
                return hook;
            }
            return null;
        }

        /// <summary>
        /// Scatter a gradient to the owning device.
        /// </summary>
        private Tensor ScatterGradient(FSDPShardingUnit shardingUnit, ReduceScatterOperation reduceScatterOp, Tensor fullGradient)
        {
            if (fullGradient == null)
                throw new InvalidOperationException($"Gradient is null for {shardingUnit.ParameterName}");

            // Perform Reduce-Scatter
            var scatteredGradient = reduceScatterOp.ReduceScatter(fullGradient);

            // Store scattered gradient in sharding unit
            shardingUnit.LocalGradient = scatteredGradient;

            // Return the scattered gradient (only the portion owned by this device)
            return scatteredGradient;
        }

        /// <summary>
        /// Scatter multiple gradients in parallel.
        /// </summary>
        /// <param name="shardingUnits">Sharding units to scatter</param>
        /// <param name="fullGradients">Full gradients from backward pass</param>
        public async Task ScatterMultipleAsync(
            Dictionary<string, FSDPShardingUnit> shardingUnits,
            Dictionary<string, Tensor> fullGradients)
        {
            if (shardingUnits == null || shardingUnits.Count == 0)
                return;

            if (fullGradients == null || fullGradients.Count == 0)
                return;

            if (shardingUnits.Count != fullGradients.Count)
                throw new ArgumentException("Sharding units and gradients must have the same count");

            var tasks = shardingUnits.Zip(fullGradients, (unitKvp, gradKvp) =>
            {
                var unit = unitKvp.Value;
                var grad = gradKvp.Value;

                if (grad == null || !_scatterOperations.TryGetValue(unit.ParameterName, out var op))
                {
                    return Task.CompletedTask;
                }

                return Task.Run(() =>
                {
                    var scatteredGrad = op.ReduceScatter(grad);
                    unit.LocalGradient = scatteredGrad;
                });
            }).ToList();

            await Task.WhenAll(tasks);
        }

        /// <summary>
        /// Accumulate gradients from multiple micro-batches.
        /// </summary>
        /// <param name="shardingUnits">Sharding units to accumulate</param>
        /// <param name="newGradients">New gradients to accumulate</param>
        public void AccumulateGradients(
            Dictionary<string, FSDPShardingUnit> shardingUnits,
            Dictionary<string, Tensor> newGradients)
        {
            if (shardingUnits == null || shardingUnits.Count == 0)
                return;

            if (newGradients == null || newGradients.Count == 0)
                return;

            foreach (var kvp in shardingUnits)
            {
                var paramName = kvp.Key;
                var shardingUnit = kvp.Value;

                if (newGradients.TryGetValue(paramName, out var newGrad))
                {
                    if (shardingUnit.LocalGradient == null)
                    {
                        // First gradient, just assign
                        shardingUnit.LocalGradient = newGrad.Clone();
                    }
                    else
                    {
                        // Accumulate
                        var localGradData = shardingUnit.LocalGradient.Data;
                        var newGradData = newGrad.Data;

                        for (int i = 0; i < Math.Min(localGradData.Length, newGradData.Length); i++)
                        {
                            localGradData[i] += newGradData[i];
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Clear accumulated gradients.
        /// </summary>
        /// <param name="shardingUnits">Sharding units to clear</param>
        public void ClearGradients(Dictionary<string, FSDPShardingUnit> shardingUnits)
        {
            if (shardingUnits == null || shardingUnits.Count == 0)
                return;

            foreach (var unit in shardingUnits.Values)
            {
                if (unit.LocalGradient != null)
                {
                    // Zero out the gradient
                    Array.Clear(unit.LocalGradient.Data, 0, unit.LocalGradient.Data.Length);
                }
            }
        }

        /// <summary>
        /// Verify that gradients are correctly scattered.
        /// </summary>
        /// <param name="shardingUnits">Sharding units to verify</param>
        /// <param name="fullGradients">Full gradients to compare against</param>
        public bool VerifyGradients(
            Dictionary<string, FSDPShardingUnit> shardingUnits,
            Dictionary<string, Tensor> fullGradients)
        {
            if (shardingUnits == null || shardingUnits.Count == 0)
                return true;

            if (fullGradients == null || fullGradients.Count == 0)
                return true;

            foreach (var kvp in shardingUnits)
            {
                var paramName = kvp.Key;
                var shardingUnit = kvp.Value;

                if (!fullGradients.TryGetValue(paramName, out var fullGrad))
                    continue;

                if (shardingUnit.LocalGradient == null)
                    return false;

                var worldSize = _processGroup.WorldSize;
                var shardIndex = shardingUnit.State.ShardIndex;

                // Calculate expected shard from full gradient
                var totalSize = fullGrad.Size;
                var chunkSize = (totalSize + worldSize - 1) / worldSize;
                var startOffset = shardIndex * chunkSize;
                var shardSize = Math.Min(chunkSize, totalSize - startOffset);

                // Compare
                for (int i = 0; i < shardSize; i++)
                {
                    var expected = fullGrad.Data[startOffset + i];
                    var actual = shardingUnit.LocalGradient.Data[i];

                    if (Math.Abs(expected - actual) > 1e-5)
                        return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Get all registered scatter operations.
        /// </summary>
        /// <returns>Dictionary of parameter names to scatter operations</returns>
        public IReadOnlyDictionary<string, ReduceScatterOperation> GetScatterOperations()
        {
            return _scatterOperations;
        }

        /// <summary>
        /// Check if a backward hook is registered for a parameter.
        /// </summary>
        /// <param name="paramName">Parameter name</param>
        /// <returns>True if a hook is registered, false otherwise</returns>
        public bool HasBackwardHook(string paramName)
        {
            return _backwardHooks.ContainsKey(paramName);
        }

        /// <summary>
        /// Clear all registered hooks.
        /// </summary>
        public void ClearHooks()
        {
            _backwardHooks.Clear();
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
                    foreach (var op in _scatterOperations.Values)
                    {
                        // ReduceScatterOperation doesn't implement IDisposable, so no action needed
                    }
                    _scatterOperations.Clear();
                    ClearHooks();
                }
                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer for FSDPBackwardHook.
        /// </summary>
        ~FSDPBackwardHook()
        {
            Dispose(false);
        }
    }
}
