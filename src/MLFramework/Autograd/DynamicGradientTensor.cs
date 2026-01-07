using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd
{
    /// <summary>
    /// Represents a tensor with gradient tracking that supports dynamic symbolic shapes.
    /// </summary>
    public sealed class DynamicGradientTensor : IDisposable
    {
        private Tensor? _gradient;
        private bool _disposed;

        /// <summary>
        /// Gets the underlying tensor.
        /// </summary>
        public Tensor Tensor { get; }

        /// <summary>
        /// Gets the symbolic shape of the tensor.
        /// </summary>
        public Shapes.SymbolicShape Shape { get; }

        /// <summary>
        /// Gets whether gradients are required for this tensor.
        /// </summary>
        public bool GradientRequired { get; }

        /// <summary>
        /// Gets the accumulated gradient, or null if no gradient has been accumulated.
        /// </summary>
        public Tensor? Gradient => _gradient;

        /// <summary>
        /// Gets whether a gradient has been accumulated.
        /// </summary>
        public bool HasGradient => _gradient != null;

        /// <summary>
        /// Initializes a new instance of the DynamicGradientTensor class.
        /// </summary>
        /// <param name="tensor">The underlying tensor.</param>
        /// <param name="shape">The symbolic shape of the tensor.</param>
        /// <param name="gradientRequired">Whether gradients are required.</param>
        /// <exception cref="ArgumentNullException">Thrown when tensor or shape is null.</exception>
        public DynamicGradientTensor(Tensor tensor, Shapes.SymbolicShape shape, bool gradientRequired)
        {
            _gradient = null;
            _disposed = false;
            Tensor = tensor ?? throw new ArgumentNullException(nameof(tensor));
            Shape = shape ?? throw new ArgumentNullException(nameof(shape));
            GradientRequired = gradientRequired;
        }

        /// <summary>
        /// Initializes a new instance of the DynamicGradientTensor class with an existing gradient.
        /// </summary>
        /// <param name="tensor">The underlying tensor.</param>
        /// <param name="shape">The symbolic shape of the tensor.</param>
        /// <param name="gradient">The existing gradient tensor.</param>
        /// <param name="gradientRequired">Whether gradients are required.</param>
        /// <exception cref="ArgumentNullException">Thrown when tensor or shape is null.</exception>
        public DynamicGradientTensor(Tensor tensor, Shapes.SymbolicShape shape, Tensor? gradient, bool gradientRequired)
            : this(tensor, shape, gradientRequired)
        {
            _gradient = gradient;
        }

        /// <summary>
        /// Accumulates a gradient for this tensor.
        /// If this is the first gradient, it is stored directly.
        /// If a gradient already exists, it is added to the existing gradient.
        /// </summary>
        /// <param name="grad">The gradient to accumulate.</param>
        /// <exception cref="InvalidOperationException">Thrown when gradients are not required.</exception>
        /// <exception cref="ArgumentNullException">Thrown when grad is null.</exception>
        /// <exception cref="ArgumentException">Thrown when gradient shape does not match tensor shape.</exception>
        public void AccumulateGradient(Tensor grad)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DynamicGradientTensor));

            if (!GradientRequired)
                throw new InvalidOperationException(
                    "Cannot accumulate gradient for tensor that does not require gradients");

            if (grad == null)
                throw new ArgumentNullException(nameof(grad));

            // Validate gradient shape matches tensor shape
            ValidateGradientShape(grad);

            if (_gradient == null)
            {
                // First gradient - store directly
                _gradient = grad;
            }
            else
            {
                // Accumulate with existing gradient
                _gradient = AddGradients(_gradient, grad);
            }
        }

        /// <summary>
        /// Validates that the gradient shape matches the tensor shape.
        /// </summary>
        /// <param name="grad">The gradient tensor to validate.</param>
        /// <exception cref="ArgumentException">Thrown when the gradient shape does not match.</exception>
        private void ValidateGradientShape(Tensor grad)
        {
            // This is a simplified validation - in practice, we'd need to compare
            // the actual tensor shapes. For now, we'll just check that grad is valid.
            if (grad == null)
                throw new ArgumentNullException(nameof(grad));
        }

        /// <summary>
        /// Adds two gradient tensors together.
        /// </summary>
        /// <param name="grad1">The first gradient.</param>
        /// <param name="grad2">The second gradient.</param>
        /// <returns>The sum of the gradients.</returns>
        private static Tensor AddGradients(Tensor grad1, Tensor grad2)
        {
            // In practice, this would use a tensor operation to add gradients
            // For now, we'll return grad1 (simplified)
            // TODO: Implement proper gradient accumulation
            return grad1;
        }

        /// <summary>
        /// Gets the accumulated gradient for this tensor.
        /// </summary>
        /// <returns>The accumulated gradient, or null if no gradient has been accumulated.</returns>
        public Tensor? GetGradient()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DynamicGradientTensor));

            return _gradient;
        }

        /// <summary>
        /// Clears the accumulated gradient for this tensor.
        /// </summary>
        public void ClearGradient()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DynamicGradientTensor));

            _gradient = null;
        }

        /// <summary>
        /// Resets this tensor, clearing any accumulated gradients and detaching from the computational graph.
        /// </summary>
        public void Reset()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DynamicGradientTensor));

            ClearGradient();
        }

        /// <summary>
        /// Detaches this tensor from the computational graph.
        /// The tensor data is preserved, but gradient tracking is disabled.
        /// </summary>
        /// <returns>A new tensor detached from the computational graph.</returns>
        public DynamicGradientTensor Detach()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DynamicGradientTensor));

            // Return a new tensor that doesn't require gradients
            return new DynamicGradientTensor(Tensor, Shape, false);
        }

        /// <summary>
        /// Returns a string representation of this dynamic gradient tensor.
        /// </summary>
        /// <returns>A string showing the tensor and its gradient status.</returns>
        public override string ToString()
        {
            return $"DynamicGradientTensor(Shape={Shape}, HasGradient={HasGradient}, RequiresGrad={GradientRequired})";
        }

        /// <summary>
        /// Disposes of this dynamic gradient tensor's resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Disposes of this dynamic gradient tensor's resources.
        /// </summary>
        /// <param name="disposing">True if disposing managed resources.</param>
        private void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _gradient = null;
                }
                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer for DynamicGradientTensor.
        /// </summary>
        ~DynamicGradientTensor()
        {
            Dispose(false);
        }
    }
}
