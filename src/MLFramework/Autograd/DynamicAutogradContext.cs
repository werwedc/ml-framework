using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd
{
    /// <summary>
    /// Context for storing information during the forward pass that is needed for the backward pass.
    /// Supports dynamic shapes for variable batch sizes and unknown dimensions.
    /// </summary>
    public sealed class DynamicAutogradContext : IDisposable
    {
        private readonly List<Tensor> _savedTensors;
        private readonly List<Shapes.SymbolicShape> _inputShapes;
        private readonly List<Shapes.SymbolicShape> _outputShapes;
        private readonly Dictionary<int, Shapes.SymbolicShape> _gradientShapes;
        private bool _disposed;

        /// <summary>
        /// Gets the shapes of the input tensors.
        /// </summary>
        public System.Collections.ObjectModel.ReadOnlyCollection<Shapes.SymbolicShape> InputShapes =>
            _inputShapes.AsReadOnly();

        /// <summary>
        /// Gets the shapes of the output tensors.
        /// </summary>
        public System.Collections.ObjectModel.ReadOnlyCollection<Shapes.SymbolicShape> OutputShapes =>
            _outputShapes.AsReadOnly();

        /// <summary>
        /// Gets the saved tensors for use in the backward pass.
        /// </summary>
        public System.Collections.ObjectModel.ReadOnlyCollection<Tensor> SavedTensors =>
            _savedTensors.AsReadOnly();

        /// <summary>
        /// Gets the number of saved tensors.
        /// </summary>
        public int SavedTensorCount => _savedTensors.Count;

        /// <summary>
        /// Gets the number of input shapes.
        /// </summary>
        public int InputShapeCount => _inputShapes.Count;

        /// <summary>
        /// Gets the number of output shapes.
        /// </summary>
        public int OutputShapeCount => _outputShapes.Count;

        /// <summary>
        /// Initializes a new instance of the DynamicAutogradContext class.
        /// </summary>
        public DynamicAutogradContext()
        {
            _savedTensors = new List<Tensor>();
            _inputShapes = new List<Shapes.SymbolicShape>();
            _outputShapes = new List<Shapes.SymbolicShape>();
            _gradientShapes = new Dictionary<int, Shapes.SymbolicShape>();
            _disposed = false;
        }

        /// <summary>
        /// Saves a tensor for use during the backward pass.
        /// </summary>
        /// <param name="tensor">The tensor to save.</param>
        /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown when context has been disposed.</exception>
        public void SaveForBackward(Tensor tensor)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DynamicAutogradContext));

            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            _savedTensors.Add(tensor);
        }

        /// <summary>
        /// Saves multiple tensors for use during the backward pass.
        /// </summary>
        /// <param name="tensors">The tensors to save.</param>
        /// <exception cref="ArgumentNullException">Thrown when tensors is null or contains null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown when context has been disposed.</exception>
        public void SaveForBackward(params Tensor[] tensors)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DynamicAutogradContext));

            if (tensors == null)
                throw new ArgumentNullException(nameof(tensors));

            foreach (var tensor in tensors)
            {
                if (tensor == null)
                    throw new ArgumentNullException(nameof(tensors), "Tensor array contains null element");

                _savedTensors.Add(tensor);
            }
        }

        /// <summary>
        /// Gets a saved tensor by index.
        /// </summary>
        /// <param name="index">The index of the saved tensor.</param>
        /// <returns>The saved tensor.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when index is out of range.</exception>
        /// <exception cref="ObjectDisposedException">Thrown when context has been disposed.</exception>
        public Tensor GetSavedTensor(int index)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DynamicAutogradContext));

            if (index < 0 || index >= _savedTensors.Count)
            {
                throw new ArgumentOutOfRangeException(nameof(index),
                    $"Index {index} is out of range. Saved tensors count: {_savedTensors.Count}");
            }

            return _savedTensors[index];
        }

        /// <summary>
        /// Saves an input shape.
        /// </summary>
        /// <param name="shape">The input shape to save.</param>
        /// <exception cref="ArgumentNullException">Thrown when shape is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown when context has been disposed.</exception>
        public void SaveInputShape(Shapes.SymbolicShape shape)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DynamicAutogradContext));

            if (shape == null)
                throw new ArgumentNullException(nameof(shape));

            _inputShapes.Add(shape);
        }

        /// <summary>
        /// Saves multiple input shapes.
        /// </summary>
        /// <param name="shapes">The input shapes to save.</param>
        /// <exception cref="ArgumentNullException">Thrown when shapes is null or contains null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown when context has been disposed.</exception>
        public void SaveInputShape(params Shapes.SymbolicShape[] shapes)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DynamicAutogradContext));

            if (shapes == null)
                throw new ArgumentNullException(nameof(shapes));

            foreach (var shape in shapes)
            {
                if (shape == null)
                    throw new ArgumentNullException(nameof(shapes), "Shape array contains null element");

                _inputShapes.Add(shape);
            }
        }

        /// <summary>
        /// Saves an output shape.
        /// </summary>
        /// <param name="shape">The output shape to save.</param>
        /// <exception cref="ArgumentNullException">Thrown when shape is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown when context has been disposed.</exception>
        public void SaveOutputShape(Shapes.SymbolicShape shape)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DynamicAutogradContext));

            if (shape == null)
                throw new ArgumentNullException(nameof(shape));

            _outputShapes.Add(shape);
        }

        /// <summary>
        /// Saves multiple output shapes.
        /// </summary>
        /// <param name="shapes">The output shapes to save.</param>
        /// <exception cref="ArgumentNullException">Thrown when shapes is null or contains null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown when context has been disposed.</exception>
        public void SaveOutputShape(params Shapes.SymbolicShape[] shapes)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DynamicAutogradContext));

            if (shapes == null)
                throw new ArgumentNullException(nameof(shapes));

            foreach (var shape in shapes)
            {
                if (shape == null)
                    throw new ArgumentNullException(nameof(shapes), "Shape array contains null element");

                _outputShapes.Add(shape);
            }
        }

        /// <summary>
        /// Registers the expected gradient shape for a given input index.
        /// </summary>
        /// <param name="index">The input index.</param>
        /// <param name="shape">The expected gradient shape.</param>
        /// <exception cref="ArgumentNullException">Thrown when shape is null.</exception>
        /// <exception cref="ObjectDisposedException">Thrown when context has been disposed.</exception>
        public void RegisterGradientShape(int index, Shapes.SymbolicShape shape)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DynamicAutogradContext));

            if (shape == null)
                throw new ArgumentNullException(nameof(shape));

            _gradientShapes[index] = shape;
        }

        /// <summary>
        /// Validates that a gradient matches the expected shape for the given input index.
        /// </summary>
        /// <param name="index">The input index.</param>
        /// <param name="gradShape">The shape of the gradient to validate.</param>
        /// <exception cref="ArgumentNullException">Thrown when gradShape is null.</exception>
        /// <exception cref="ArgumentException">Thrown when the gradient shape doesn't match or no gradient shape is registered.</exception>
        /// <exception cref="ObjectDisposedException">Thrown when context has been disposed.</exception>
        public void ValidateGradient(int index, Shapes.SymbolicShape gradShape)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DynamicAutogradContext));

            if (gradShape == null)
                throw new ArgumentNullException(nameof(gradShape));

            if (!_gradientShapes.ContainsKey(index))
            {
                throw new ArgumentException(
                    $"No gradient shape registered for input index {index}");
            }

            var expectedShape = _gradientShapes[index];
            if (!expectedShape.Equals(gradShape))
            {
                throw new ArgumentException(
                    $"Gradient shape mismatch at index {index}. Expected: {expectedShape}, Got: {gradShape}");
            }
        }

        /// <summary>
        /// Gets the expected gradient shape for a given input index.
        /// </summary>
        /// <param name="index">The input index.</param>
        /// <returns>The expected gradient shape.</returns>
        /// <exception cref="KeyNotFoundException">Thrown when no gradient shape is registered for the index.</exception>
        /// <exception cref="ObjectDisposedException">Thrown when context has been disposed.</exception>
        public Shapes.SymbolicShape GetGradientShape(int index)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DynamicAutogradContext));

            if (!_gradientShapes.ContainsKey(index))
            {
                throw new KeyNotFoundException(
                    $"No gradient shape registered for input index {index}");
            }

            return _gradientShapes[index];
        }

        /// <summary>
        /// Gets the input shape at the specified index.
        /// </summary>
        /// <param name="index">The index of the input shape.</param>
        /// <returns>The input shape.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when index is out of range.</exception>
        /// <exception cref="ObjectDisposedException">Thrown when context has been disposed.</exception>
        public Shapes.SymbolicShape GetInputShape(int index)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DynamicAutogradContext));

            if (index < 0 || index >= _inputShapes.Count)
            {
                throw new ArgumentOutOfRangeException(nameof(index),
                    $"Index {index} is out of range. Input shapes count: {_inputShapes.Count}");
            }

            return _inputShapes[index];
        }

        /// <summary>
        /// Gets the output shape at the specified index.
        /// </summary>
        /// <param name="index">The index of the output shape.</param>
        /// <returns>The output shape.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when index is out of range.</exception>
        /// <exception cref="ObjectDisposedException">Thrown when context has been disposed.</exception>
        public Shapes.SymbolicShape GetOutputShape(int index)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DynamicAutogradContext));

            if (index < 0 || index >= _outputShapes.Count)
            {
                throw new ArgumentOutOfRangeException(nameof(index),
                    $"Index {index} is out of range. Output shapes count: {_outputShapes.Count}");
            }

            return _outputShapes[index];
        }

        /// <summary>
        /// Clears all saved tensors and shapes.
        /// </summary>
        /// <exception cref="ObjectDisposedException">Thrown when context has been disposed.</exception>
        public void Clear()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DynamicAutogradContext));

            _savedTensors.Clear();
            _inputShapes.Clear();
            _outputShapes.Clear();
            _gradientShapes.Clear();
        }

        /// <summary>
        /// Returns a string representation of this context.
        /// </summary>
        /// <returns>A string showing the context state.</returns>
        public override string ToString()
        {
            return $"DynamicAutogradContext(SavedTensors={_savedTensors.Count}, " +
                   $"InputShapes={_inputShapes.Count}, OutputShapes={_outputShapes.Count})";
        }

        /// <summary>
        /// Disposes of this context's resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Disposes of this context's resources.
        /// </summary>
        /// <param name="disposing">True if disposing managed resources.</param>
        private void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    Clear();
                }
                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer for DynamicAutogradContext.
        /// </summary>
        ~DynamicAutogradContext()
        {
            Dispose(false);
        }
    }
}
