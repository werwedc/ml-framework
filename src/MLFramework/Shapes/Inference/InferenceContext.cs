using MLFramework.Shapes;

namespace MLFramework.Shapes.Inference
{
    /// <summary>
    /// Context for managing tensor shapes and operation results during shape inference.
    /// </summary>
    public class InferenceContext
    {
        private readonly Dictionary<string, SymbolicShape> _tensorShapes;
        private readonly Dictionary<string, List<SymbolicShape>> _operationResults;

        /// <summary>
        /// Gets the tensor shapes managed by this context.
        /// </summary>
        public IReadOnlyDictionary<string, SymbolicShape> TensorShapes => _tensorShapes;

        /// <summary>
        /// Gets the operation results managed by this context.
        /// </summary>
        public IReadOnlyDictionary<string, List<SymbolicShape>> OperationResults => _operationResults;

        /// <summary>
        /// Initializes a new instance of the InferenceContext class.
        /// </summary>
        public InferenceContext()
        {
            _tensorShapes = new Dictionary<string, SymbolicShape>();
            _operationResults = new Dictionary<string, List<SymbolicShape>>();
        }

        /// <summary>
        /// Initializes a new instance of the InferenceContext class with initial shapes.
        /// </summary>
        /// <param name="initialShapes">Initial tensor shapes to populate the context.</param>
        public InferenceContext(IDictionary<string, SymbolicShape> initialShapes)
            : this()
        {
            if (initialShapes == null)
                throw new ArgumentNullException(nameof(initialShapes));

            foreach (var kvp in initialShapes)
            {
                _tensorShapes[kvp.Key] = kvp.Value;
            }
        }

        /// <summary>
        /// Gets the shape of a tensor by name.
        /// </summary>
        /// <param name="tensorName">The name of the tensor.</param>
        /// <returns>The shape of the tensor, or null if not found.</returns>
        public SymbolicShape? GetShape(string tensorName)
        {
            if (string.IsNullOrEmpty(tensorName))
                return null;

            _tensorShapes.TryGetValue(tensorName, out var shape);
            return shape;
        }

        /// <summary>
        /// Sets the shape of a tensor.
        /// </summary>
        /// <param name="tensorName">The name of the tensor.</param>
        /// <param name="shape">The shape to set.</param>
        /// <exception cref="ArgumentNullException">Thrown when tensorName or shape is null.</exception>
        public void SetShape(string tensorName, SymbolicShape shape)
        {
            if (string.IsNullOrEmpty(tensorName))
                throw new ArgumentNullException(nameof(tensorName));

            if (shape == null)
                throw new ArgumentNullException(nameof(shape));

            _tensorShapes[tensorName] = shape;
        }

        /// <summary>
        /// Records the inference result of an operation.
        /// </summary>
        /// <param name="opId">The ID of the operation.</param>
        /// <param name="outputs">The output shapes of the operation.</param>
        /// <exception cref="ArgumentNullException">Thrown when opId or outputs is null.</exception>
        public void RecordInference(string opId, List<SymbolicShape> outputs)
        {
            if (string.IsNullOrEmpty(opId))
                throw new ArgumentNullException(nameof(opId));

            if (outputs == null)
                throw new ArgumentNullException(nameof(outputs));

            _operationResults[opId] = new List<SymbolicShape>(outputs);
        }

        /// <summary>
        /// Gets the inference results of an operation.
        /// </summary>
        /// <param name="opId">The ID of the operation.</param>
        /// <returns>The output shapes of the operation, or null if not found.</returns>
        public List<SymbolicShape>? GetOperationResults(string opId)
        {
            if (string.IsNullOrEmpty(opId))
                return null;

            _operationResults.TryGetValue(opId, out var results);
            return results;
        }

        /// <summary>
        /// Checks if a shape exists for a tensor.
        /// </summary>
        /// <param name="tensorName">The name of the tensor.</param>
        /// <returns>True if the shape exists; otherwise, false.</returns>
        public bool HasShape(string tensorName)
        {
            if (string.IsNullOrEmpty(tensorName))
                return false;

            return _tensorShapes.ContainsKey(tensorName);
        }

        /// <summary>
        /// Checks if inference results exist for an operation.
        /// </summary>
        /// <param name="opId">The ID of the operation.</param>
        /// <returns>True if the results exist; otherwise, false.</returns>
        public bool HasOperationResults(string opId)
        {
            if (string.IsNullOrEmpty(opId))
                return false;

            return _operationResults.ContainsKey(opId);
        }

        /// <summary>
        /// Clears all tensor shapes from the context.
        /// </summary>
        public void ClearTensorShapes()
        {
            _tensorShapes.Clear();
        }

        /// <summary>
        /// Clears all operation results from the context.
        /// </summary>
        public void ClearOperationResults()
        {
            _operationResults.Clear();
        }

        /// <summary>
        /// Clears all data from the context.
        /// </summary>
        public void Clear()
        {
            ClearTensorShapes();
            ClearOperationResults();
        }

        /// <summary>
        /// Creates a clone of this inference context.
        /// </summary>
        /// <returns>A new InferenceContext with cloned data.</returns>
        public InferenceContext Clone()
        {
            var clonedContext = new InferenceContext();

            // Clone tensor shapes
            foreach (var kvp in _tensorShapes)
            {
                clonedContext.SetShape(kvp.Key, kvp.Value.Clone());
            }

            // Clone operation results
            foreach (var kvp in _operationResults)
            {
                var clonedOutputs = kvp.Value.Select(s => s.Clone()).ToList();
                clonedContext.RecordInference(kvp.Key, clonedOutputs);
            }

            return clonedContext;
        }
    }
}
