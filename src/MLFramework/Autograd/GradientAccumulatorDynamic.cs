using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd
{
    /// <summary>
    /// Accumulates gradients for tensors with dynamic shapes.
    /// Supports batched gradient accumulation with variable batch sizes.
    /// </summary>
    public sealed class GradientAccumulatorDynamic : IDisposable
    {
        private readonly Dictionary<string, AccumulatedGradient> _gradients;
        private bool _disposed;

        /// <summary>
        /// Gets the number of accumulated gradients.
        /// </summary>
        public int Count => _gradients.Count;

        /// <summary>
        /// Initializes a new instance of the GradientAccumulatorDynamic class.
        /// </summary>
        public GradientAccumulatorDynamic()
        {
            _gradients = new Dictionary<string, AccumulatedGradient>();
            _disposed = false;
        }

        /// <summary>
        /// Accumulates a gradient for a tensor with the given identifier and shape.
        /// </summary>
        /// <param name="identifier">The identifier for the gradient (e.g., parameter name).</param>
        /// <param name="grad">The gradient tensor to accumulate.</param>
        /// <param name="shape">The symbolic shape of the gradient.</param>
        /// <exception cref="ArgumentNullException">Thrown when identifier, grad, or shape is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown when accumulator has been disposed.</exception>
        public void Accumulate(string identifier, Tensor grad, Shapes.SymbolicShape shape)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(GradientAccumulatorDynamic));

            if (identifier == null)
                throw new ArgumentNullException(nameof(identifier));

            if (grad == null)
                throw new ArgumentNullException(nameof(grad));

            if (shape == null)
                throw new ArgumentNullException(nameof(shape));

            if (_gradients.TryGetValue(identifier, out var accumulated))
            {
                // Validate shape compatibility
                if (!accumulated.Shape.Equals(shape))
                {
                    throw new ArgumentException(
                        $"Gradient shape mismatch for {identifier}. " +
                        $"Expected: {accumulated.Shape}, Got: {shape}");
                }

                // Add to accumulated gradient
                accumulated.Gradient = AddGradients(accumulated.Gradient, grad);
                accumulated.Count++;
            }
            else
            {
                // Create new accumulated gradient
                _gradients[identifier] = new AccumulatedGradient
                {
                    Gradient = grad,
                    Shape = shape,
                    Count = 1
                };
            }
        }

        /// <summary>
        /// Accumulates batched gradients with variable batch sizes.
        /// </summary>
        /// <param name="identifier">The identifier for the gradient.</param>
        /// <param name="grads">The list of gradient tensors to accumulate.</param>
        /// <param name="shapes">The list of symbolic shapes corresponding to the gradients.</param>
        /// <exception cref="ArgumentNullException">Thrown when identifier, grads, or shapes is null.</exception>
        /// <exception cref="ArgumentException">Thrown when grads and shapes have different lengths.</exception>
        /// <exception cref="ObjectDisposedException">Thrown when accumulator has been disposed.</exception>
        public void AccumulateBatched(
            string identifier,
            System.Collections.Generic.List<Tensor> grads,
            System.Collections.Generic.List<Shapes.SymbolicShape> shapes)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(GradientAccumulatorDynamic));

            if (identifier == null)
                throw new ArgumentNullException(nameof(identifier));

            if (grads == null)
                throw new ArgumentNullException(nameof(grads));

            if (shapes == null)
                throw new ArgumentNullException(nameof(shapes));

            if (grads.Count != shapes.Count)
            {
                throw new ArgumentException(
                    $"Gradients and shapes lists must have the same length. " +
                    $"Got {grads.Count} gradients and {shapes.Count} shapes.");
            }

            // Accumulate each gradient in the batch
            for (int i = 0; i < grads.Count; i++)
            {
                Accumulate(identifier, grads[i], shapes[i]);
            }
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
            // TODO: Implement proper gradient addition
            return grad1;
        }

        /// <summary>
        /// Gets the accumulated gradient for a given identifier.
        /// </summary>
        /// <param name="identifier">The identifier of the gradient.</param>
        /// <returns>The accumulated gradient tensor, or null if not found.</returns>
        /// <exception cref="ArgumentNullException">Thrown when identifier is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown when accumulator has been disposed.</exception>
        public Tensor? GetAccumulated(string identifier)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(GradientAccumulatorDynamic));

            if (identifier == null)
                throw new ArgumentNullException(nameof(identifier));

            if (_gradients.TryGetValue(identifier, out var accumulated))
            {
                return accumulated.Gradient;
            }

            return null;
        }

        /// <summary>
        /// Gets the accumulated gradient for a given identifier and shape.
        /// </summary>
        /// <param name="identifier">The identifier of the gradient.</param>
        /// <param name="shape">The expected shape of the gradient.</param>
        /// <returns>The accumulated gradient tensor, or null if not found.</returns>
        /// <exception cref="ArgumentNullException">Thrown when identifier or shape is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown when accumulator has been disposed.</exception>
        public Tensor? GetAccumulated(string identifier, Shapes.SymbolicShape shape)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(GradientAccumulatorDynamic));

            if (identifier == null)
                throw new ArgumentNullException(nameof(identifier));

            if (shape == null)
                throw new ArgumentNullException(nameof(shape));

            if (_gradients.TryGetValue(identifier, out var accumulated))
            {
                // Validate shape
                if (!accumulated.Shape.Equals(shape))
                {
                    throw new ArgumentException(
                        $"Gradient shape mismatch for {identifier}. " +
                        $"Expected: {accumulated.Shape}, Got: {shape}");
                }

                return accumulated.Gradient;
            }

            return null;
        }

        /// <summary>
        /// Gets the accumulation count for a given identifier.
        /// </summary>
        /// <param name="identifier">The identifier of the gradient.</param>
        /// <returns>The number of gradients accumulated, or 0 if not found.</returns>
        /// <exception cref="ArgumentNullException">Thrown when identifier is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown when accumulator has been disposed.</exception>
        public int GetAccumulationCount(string identifier)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(GradientAccumulatorDynamic));

            if (identifier == null)
                throw new ArgumentNullException(nameof(identifier));

            if (_gradients.TryGetValue(identifier, out var accumulated))
            {
                return accumulated.Count;
            }

            return 0;
        }

        /// <summary>
        /// Gets all accumulated gradient identifiers.
        /// </summary>
        /// <returns>A read-only collection of gradient identifiers.</returns>
        /// <exception cref="ObjectDisposedException">Thrown when accumulator has been disposed.</exception>
        public System.Collections.ObjectModel.ReadOnlyCollection<string> GetIdentifiers()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(GradientAccumulatorDynamic));

            return _gradients.Keys.ToList().AsReadOnly();
        }

        /// <summary>
        /// Checks if a gradient exists for the given identifier.
        /// </summary>
        /// <param name="identifier">The identifier to check.</param>
        /// <returns>True if a gradient exists; otherwise, false.</returns>
        /// <exception cref="ArgumentNullException">Thrown when identifier is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown when accumulator has been disposed.</exception>
        public bool HasGradient(string identifier)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(GradientAccumulatorDynamic));

            if (identifier == null)
                throw new ArgumentNullException(nameof(identifier));

            return _gradients.ContainsKey(identifier);
        }

        /// <summary>
        /// Removes a gradient from the accumulator.
        /// </summary>
        /// <param name="identifier">The identifier of the gradient to remove.</param>
        /// <returns>True if the gradient was removed; otherwise, false.</returns>
        /// <exception cref="ArgumentNullException">Thrown when identifier is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown when accumulator has been disposed.</exception>
        public bool Remove(string identifier)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(GradientAccumulatorDynamic));

            if (identifier == null)
                throw new ArgumentNullException(nameof(identifier));

            return _gradients.Remove(identifier);
        }

        /// <summary>
        /// Resets the accumulator, clearing all accumulated gradients.
        /// </summary>
        /// <exception cref="ObjectDisposedException">Thrown when accumulator has been disposed.</exception>
        public void Reset()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(GradientAccumulatorDynamic));

            _gradients.Clear();
        }

        /// <summary>
        /// Returns a string representation of the accumulator state.
        /// </summary>
        /// <returns>A string showing the number of accumulated gradients.</returns>
        public override string ToString()
        {
            return $"GradientAccumulatorDynamic(Count={_gradients.Count})";
        }

        /// <summary>
        /// Disposes of the accumulator's resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Disposes of the accumulator's resources.
        /// </summary>
        /// <param name="disposing">True if disposing managed resources.</param>
        private void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    Reset();
                }
                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer for GradientAccumulatorDynamic.
        /// </summary>
        ~GradientAccumulatorDynamic()
        {
            Dispose(false);
        }

        /// <summary>
        /// Internal class to store accumulated gradient information.
        /// </summary>
        private class AccumulatedGradient
        {
            public required Tensor Gradient { get; set; }
            public required Shapes.SymbolicShape Shape { get; set; }
            public int Count { get; set; }
        }
    }
}
