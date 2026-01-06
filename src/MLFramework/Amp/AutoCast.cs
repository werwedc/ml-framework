using MLFramework.Core;
using RitterFramework.Core.Tensor;
using System;
using System.Threading;

namespace MLFramework.Amp
{
    /// <summary>
    /// Context manager for automatic mixed precision casting
    /// </summary>
    public class AutoCast : IDisposable
    {
        private readonly AutoCastMode _mode;
        private readonly AmpRegistry _registry;
        private readonly bool _enabled;
        private readonly Stack<AutoCast> _contextStack;

        /// <summary>
        /// Gets the current AutoCast mode
        /// </summary>
        public AutoCastMode Mode => _mode;

        /// <summary>
        /// Gets whether AutoCast is enabled
        /// </summary>
        public bool Enabled => _enabled;

        /// <summary>
        /// Gets the current active AutoCast context (thread-local)
        /// </summary>
        private static readonly AsyncLocal<AutoCast?> _current = new AsyncLocal<AutoCast?>();
        public static AutoCast? Current
        {
            get => _current.Value;
            private set => _current.Value = value;
        }

        /// <summary>
        /// Creates a new AutoCast context with BF16 mode (recommended)
        /// </summary>
        /// <param name="enabled">Whether to enable AutoCast (default: true)</param>
        /// <param name="registry">The operation precision registry (default: null for default registry)</param>
        public AutoCast(bool enabled = true, AmpRegistry? registry = null)
            : this(AutoCastMode.Bf16, enabled, registry)
        {
        }

        /// <summary>
        /// Creates a new AutoCast context with specified mode
        /// </summary>
        /// <param name="mode">The AutoCast mode (FP16, BF16, or None)</param>
        /// <param name="enabled">Whether to enable AutoCast (default: true)</param>
        /// <param name="registry">The operation precision registry (default: null for default registry)</param>
        public AutoCast(AutoCastMode mode, bool enabled = true, AmpRegistry? registry = null)
        {
            _mode = mode;
            _enabled = enabled;
            _registry = registry ?? new AmpRegistry(AmpConfig.CreateBf16());
            _contextStack = new Stack<AutoCast>();
        }

        /// <summary>
        /// Casts a tensor to the appropriate precision for the current operation
        /// Note: Currently a stub - actual casting implementation will be added when tensor storage supports dtypes
        /// </summary>
        /// <param name="tensor">The tensor to cast</param>
        /// <param name="operationType">The type of operation being performed</param>
        /// <returns>Casted tensor</returns>
        public Tensor Cast(Tensor tensor, Type operationType)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (operationType == null)
                throw new ArgumentNullException(nameof(operationType));

            if (!_enabled || _mode == AutoCastMode.None)
            {
                return tensor;
            }

            // TODO: Implement actual dtype casting when tensor storage supports it
            // For now, return the tensor as-is
            return tensor;
        }

        /// <summary>
        /// Casts a tensor to a specific dtype
        /// Note: Currently a stub - actual casting implementation will be added when tensor storage supports dtypes
        /// </summary>
        /// <param name="tensor">The tensor to cast</param>
        /// <param name="dtype">The target data type</param>
        /// <returns>Casted tensor</returns>
        public Tensor Cast(Tensor tensor, DataType dtype)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (!_enabled || _mode == AutoCastMode.None)
            {
                return tensor;
            }

            // TODO: Implement actual dtype casting when tensor storage supports it
            // For now, return the tensor as-is
            return tensor;
        }

        /// <summary>
        /// Gets the forward dtype for an operation
        /// </summary>
        /// <param name="operationType">The type of operation</param>
        /// <param name="inputDtype">The input tensor dtype</param>
        /// <returns>The target dtype for the operation</returns>
        public DataType GetForwardDtype(Type operationType, DataType inputDtype)
        {
            var rule = _registry.GetRule(operationType);

            if (rule != null)
            {
                return rule.GetForwardDtype(_registry.GetConfig());
            }

            // Default policy: use input dtype
            return inputDtype;
        }

        /// <summary>
        /// Gets the backward dtype for an operation
        /// </summary>
        /// <param name="operationType">The type of operation</param>
        /// <param name="inputDtype">The input tensor dtype</param>
        /// <returns>The target dtype for the operation</returns>
        public DataType GetBackwardDtype(Type operationType, DataType inputDtype)
        {
            var rule = _registry.GetRule(operationType);

            if (rule != null)
            {
                return rule.GetBackwardDtype(_registry.GetConfig());
            }

            // Default policy: use input dtype
            return inputDtype;
        }

        /// <summary>
        /// Enters the AutoCast context
        /// </summary>
        public void Enter()
        {
            _contextStack.Push(Current ?? this);
            Current = this;
        }

        /// <summary>
        /// Exits the AutoCast context
        /// </summary>
        public void Exit()
        {
            if (_contextStack.Count > 0)
            {
                Current = _contextStack.Pop();
            }
        }

        /// <summary>
        /// Disposes the context and restores previous context
        /// </summary>
        public void Dispose()
        {
            Exit();
        }
    }
}
