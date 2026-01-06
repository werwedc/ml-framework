using MLFramework.NN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using RitterFramework.Core.Tensor;

namespace MLFramework.Distributed
{
    /// <summary>
    /// DistributedDataParallel (DDP) module that wraps a model and automatically
    /// handles gradient synchronization during backward pass.
    /// </summary>
    public class DistributedDataParallel : Module
    {
        private readonly Module _module;
        private readonly IProcessGroup _processGroup;
        private readonly bool _findUnusedParameters;
        private readonly GradientBucketManager _bucketManager;
        private readonly HashSet<string> _unusedParameters;
        private readonly List<GradientSynchronizationHook> _hooks;

        /// <summary>
        /// Gets the wrapped module.
        /// </summary>
        public Module Module => _module;

        /// <summary>
        /// Gets the process group.
        /// </summary>
        public IProcessGroup ProcessGroup => _processGroup;

        /// <summary>
        /// Gets whether findUnusedParameters mode is enabled.
        /// </summary>
        public bool FindUnusedParameters => _findUnusedParameters;

        /// <summary>
        /// Gets unused parameters from the last forward pass.
        /// Only populated when findUnusedParameters = true.
        /// </summary>
        public IReadOnlyCollection<string> UnusedParameters => _unusedParameters;

        /// <summary>
        /// Creates a new DistributedDataParallel wrapper.
        /// </summary>
        public DistributedDataParallel(
            Module module,
            IProcessGroup processGroup,
            bool findUnusedParameters = false)
            : base($"DDP({module.Name})")
        {
            _module = module ?? throw new ArgumentNullException(nameof(module));
            _processGroup = processGroup ?? throw new ArgumentNullException(nameof(processGroup));
            _findUnusedParameters = findUnusedParameters;
            _unusedParameters = new HashSet<string>();
            _hooks = new List<GradientSynchronizationHook>();

            // Initialize bucket manager with module parameters
            var parameters = _module.GetParameters().ToList();
            _bucketManager = new GradientBucketManager(_processGroup, parameters);

            // Register gradient hooks for automatic synchronization
            RegisterGradientHooks();

            // Broadcast initial weights from rank 0
            BroadcastParameters();
        }

        /// <summary>
        /// Forward pass delegates to the wrapped module.
        /// </summary>
        public override Tensor Forward(Tensor input)
        {
            if (_findUnusedParameters)
            {
                _unusedParameters.Clear();
                // Mark all parameters as unused initially
                foreach (var param in _module.GetParameters())
                {
                    _unusedParameters.Add(param.Name);
                }
            }

            return _module.Forward(input);
        }

        /// <summary>
        /// Gets all parameters from the wrapped module.
        /// </summary>
        public override IEnumerable<Parameter> GetParameters()
        {
            return _module.GetParameters();
        }

        /// <summary>
        /// Gets all named parameters from the wrapped module.
        /// </summary>
        public override IEnumerable<(string Name, Parameter Parameter)> GetNamedParameters()
        {
            return _module.GetNamedParameters();
        }

        /// <summary>
        /// Reduces all gradients asynchronously.
        /// Called automatically by gradient hooks, but can be called manually.
        /// </summary>
        public Task ReduceGradientsAsync(ReduceOp op = ReduceOp.Sum)
        {
            return _bucketManager.ReduceAllAsync(op);
        }

        /// <summary>
        /// Synchronizes parameters across all ranks.
        /// Broadcast from rank 0 to all other ranks.
        /// </summary>
        public void BroadcastParameters()
        {
            var parameters = _module.GetParameters();

            if (_processGroup.Rank == 0)
            {
                // Rank 0 broadcasts its parameters to all other ranks
                foreach (var param in parameters)
                {
                    _processGroup.Broadcast(param, root: 0);
                }
            }
            else
            {
                // Other ranks receive broadcast from rank 0
                foreach (var param in parameters)
                {
                    _processGroup.Broadcast(param, root: 0);
                }
            }
        }

        /// <summary>
        /// Waits for all gradient reductions to complete.
        /// </summary>
        public async Task WaitForReductionAsync()
        {
            await _bucketManager.WaitForAllAsync();
            _bucketManager.CopyBackAll();
        }

        /// <summary>
        /// Marks a parameter as used (for findUnusedParameters mode).
        /// </summary>
        internal void MarkParameterAsUsed(string parameterName)
        {
            if (_findUnusedParameters)
            {
                _unusedParameters.Remove(parameterName);
            }
        }

        /// <summary>
        /// Registers gradient hooks for automatic synchronization.
        /// </summary>
        private void RegisterGradientHooks()
        {
            foreach (var param in _module.GetParameters())
            {
                // Mark all parameters as unused initially if tracking unused parameters
                if (_findUnusedParameters)
                {
                    _unusedParameters.Add(param.Name);
                }

                // Create and register gradient hook
                var hook = new GradientSynchronizationHook(this, param, _bucketManager);
                param.RegisterGradHook(hook.OnGradient);
                _hooks.Add(hook);
            }
        }

        /// <summary>
        /// Disposes the DDP wrapper and releases resources.
        /// </summary>
        public void Dispose()
        {
            _bucketManager?.Dispose();

            // Unregister hooks
            foreach (var hook in _hooks)
            {
                // Hooks will be garbage collected when parameters are disposed
            }

            _hooks.Clear();
            _unusedParameters.Clear();

            GC.SuppressFinalize(this);
        }
    }
}
