using System;
using RitterFramework.Core.Tensor;
using MLFramework.Amp;
using MLFramework.Optimizers;

namespace MLFramework.Amp
{
    /// <summary>
    /// Context manager for using GradScaler with automatic cleanup
    /// </summary>
    public class GradScalerContext : IDisposable
    {
        private readonly GradScaler _scaler;
        private readonly Tensor _scaledLoss;
        private bool _stepped;
        private bool _disposed;

        /// <summary>
        /// Gets the scaled loss
        /// </summary>
        public Tensor ScaledLoss => _scaledLoss;

        /// <summary>
        /// Creates a new GradScalerContext
        /// </summary>
        /// <param name="scaler">The GradScaler to use</param>
        /// <param name="loss">The loss tensor to scale</param>
    public GradScalerContext(GradScaler scaler, Tensor loss)
    {
        _scaler = scaler ?? throw new ArgumentNullException(nameof(scaler));
        _scaledLoss = _scaler.ScaleLoss(loss ?? throw new ArgumentNullException(nameof(loss)));
        _stepped = false;
        _disposed = false;
    }

        /// <summary>
        /// Prepares gradients for optimizer step with unscaling and updates scale
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <returns>True if step should be performed, false if skipped</returns>
        public bool PrepareStep(Dictionary<string, Tensor> gradients)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(GradScalerContext));

            if (_stepped)
                throw new InvalidOperationException("Step has already been called");

            _stepped = true;
            return _scaler.PrepareGradients(gradients);
        }

        /// <summary>
        /// Prepares gradients and returns them (or null if overflow)
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <returns>Unscaled gradients if valid, null if overflow detected</returns>
        public Dictionary<string, Tensor> PrepareAndGetGradients(Dictionary<string, Tensor> gradients)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(GradScalerContext));

            if (_stepped)
                throw new InvalidOperationException("Step has already been called");

            _stepped = true;
            return _scaler.PrepareAndGetGradients(gradients);
        }

        /// <summary>
        /// Disposes the context and cleans up resources
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                // Note: We don't dispose the tensor as it's managed by the framework
                _disposed = true;
            }
        }
    }
}
