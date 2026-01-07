using MLFramework.Shapes;

namespace MLFramework.Shapes.Inference
{
    /// <summary>
    /// Engine for inferring output shapes for operations based on registered rules.
    /// </summary>
    public class ShapeInferenceEngine
    {
        private readonly Dictionary<string, IShapeInferenceRule> _rules;
        private readonly InferenceContext _defaultContext;

        /// <summary>
        /// Gets the registered shape inference rules.
        /// </summary>
        public IReadOnlyDictionary<string, IShapeInferenceRule> Rules => _rules;

        /// <summary>
        /// Initializes a new instance of the ShapeInferenceEngine class.
        /// </summary>
        public ShapeInferenceEngine()
            : this(new InferenceContext())
        {
        }

        /// <summary>
        /// Initializes a new instance of the ShapeInferenceEngine class with a default context.
        /// </summary>
        /// <param name="defaultContext">The default inference context.</param>
        public ShapeInferenceEngine(InferenceContext defaultContext)
        {
            _defaultContext = defaultContext ?? throw new ArgumentNullException(nameof(defaultContext));
            _rules = new Dictionary<string, IShapeInferenceRule>(StringComparer.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Registers a shape inference rule for an operation.
        /// </summary>
        /// <param name="opName">The name of the operation.</param>
        /// <param name="rule">The shape inference rule to register.</param>
        /// <exception cref="ArgumentNullException">Thrown when opName or rule is null.</exception>
        public void RegisterRule(string opName, IShapeInferenceRule rule)
        {
            if (string.IsNullOrEmpty(opName))
                throw new ArgumentNullException(nameof(opName));

            if (rule == null)
                throw new ArgumentNullException(nameof(rule));

            _rules[opName] = rule;
        }

        /// <summary>
        /// Unregisters the shape inference rule for an operation.
        /// </summary>
        /// <param name="opName">The name of the operation.</param>
        /// <returns>True if the rule was removed; otherwise, false.</returns>
        public bool UnregisterRule(string opName)
        {
            if (string.IsNullOrEmpty(opName))
                return false;

            return _rules.Remove(opName);
        }

        /// <summary>
        /// Checks if the engine can infer the output shapes for the given operation and input shapes.
        /// </summary>
        /// <param name="opName">The name of the operation.</param>
        /// <param name="inputs">The input shapes.</param>
        /// <returns>True if the engine can infer the output shapes; otherwise, false.</returns>
        public bool CanInfer(string opName, IReadOnlyList<SymbolicShape> inputs)
        {
            if (string.IsNullOrEmpty(opName))
                return false;

            if (!_rules.TryGetValue(opName, out var rule))
                return false;

            return rule.CanInfer(opName, inputs);
        }

        /// <summary>
        /// Infers the output shapes for the given operation and input shapes.
        /// </summary>
        /// <param name="opName">The name of the operation.</param>
        /// <param name="inputs">The input shapes.</param>
        /// <returns>The inferred output shapes.</returns>
        /// <exception cref="ArgumentException">Thrown when the operation is not registered or cannot be inferred.</exception>
        /// <exception cref="InvalidOperationException">Thrown when inference fails.</exception>
        public List<SymbolicShape> Infer(string opName, IReadOnlyList<SymbolicShape> inputs)
        {
            if (string.IsNullOrEmpty(opName))
                throw new ArgumentNullException(nameof(opName));

            if (!_rules.TryGetValue(opName, out var rule))
            {
                throw new ArgumentException(
                    $"No shape inference rule registered for operation '{opName}'");
            }

            if (!rule.CanInfer(opName, inputs))
            {
                throw new InvalidOperationException(
                    $"Cannot infer shape for operation '{opName}' with {inputs.Count} input(s)");
            }

            return rule.Infer(opName, inputs);
        }

        /// <summary>
        /// Validates that the input shapes are valid for the given operation.
        /// </summary>
        /// <param name="opName">The name of the operation.</param>
        /// <param name="inputs">The input shapes to validate.</param>
        /// <returns>True if the shapes are valid; otherwise, false.</returns>
        public bool Validate(string opName, IReadOnlyList<SymbolicShape> inputs)
        {
            if (string.IsNullOrEmpty(opName))
                return false;

            try
            {
                return CanInfer(opName, inputs);
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Gets the registered rule for an operation.
        /// </summary>
        /// <param name="opName">The name of the operation.</param>
        /// <returns>The registered rule, or null if not found.</returns>
        public IShapeInferenceRule? GetRule(string opName)
        {
            if (string.IsNullOrEmpty(opName))
                return null;

            _rules.TryGetValue(opName, out var rule);
            return rule;
        }

        /// <summary>
        /// Checks if a rule is registered for an operation.
        /// </summary>
        /// <param name="opName">The name of the operation.</param>
        /// <returns>True if a rule is registered; otherwise, false.</returns>
        public bool HasRule(string opName)
        {
            if (string.IsNullOrEmpty(opName))
                return false;

            return _rules.ContainsKey(opName);
        }

        /// <summary>
        /// Gets the names of all registered operations.
        /// </summary>
        /// <returns>A collection of registered operation names.</returns>
        public IEnumerable<string> GetRegisteredOperations()
        {
            return _rules.Keys;
        }

        /// <summary>
        /// Clears all registered rules.
        /// </summary>
        public void ClearRules()
        {
            _rules.Clear();
        }

        /// <summary>
        /// Registers multiple rules at once.
        /// </summary>
        /// <param name="rules">A dictionary of operation names to rules.</param>
        /// <exception cref="ArgumentNullException">Thrown when rules is null.</exception>
        public void RegisterRules(IDictionary<string, IShapeInferenceRule> rules)
        {
            if (rules == null)
                throw new ArgumentNullException(nameof(rules));

            foreach (var kvp in rules)
            {
                RegisterRule(kvp.Key, kvp.Value);
            }
        }
    }
}
