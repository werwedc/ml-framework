using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.Shapes
{
    /// <summary>
    /// Validates symbolic dimensions against a set of constraints.
    /// Provides thread-safe constraint checking with caching for performance.
    /// </summary>
    public sealed class ConstraintValidator
    {
        private readonly object _lock = new object();
        private readonly Dictionary<string, (bool isValid, List<string> violations)> _validationCache;

        /// <summary>
        /// Initializes a new instance of the ConstraintValidator class.
        /// </summary>
        public ConstraintValidator()
        {
            _validationCache = new Dictionary<string, (bool, List<string>)>();
        }

        /// <summary>
        /// Validates all dimensions against their associated constraints.
        /// All constraints must be satisfied (AND logic).
        /// </summary>
        /// <param name="dims">The dimensions to validate.</param>
        /// <param name="constraints">A dictionary mapping dimension names to their constraint lists.</param>
        /// <returns>True if all dimensions satisfy all their constraints; otherwise, false.</returns>
        public bool ValidateAll(IEnumerable<SymbolicDimension> dims, Dictionary<string, List<IShapeConstraint>> constraints)
        {
            if (dims == null)
                throw new ArgumentNullException(nameof(dims));

            if (constraints == null)
                throw new ArgumentNullException(nameof(constraints));

            foreach (var dim in dims)
            {
                if (dim == null)
                    return false;

                // Skip dimensions without constraints
                if (!constraints.ContainsKey(dim.Name))
                    continue;

                var dimConstraints = constraints[dim.Name];
                if (dimConstraints == null || dimConstraints.Count == 0)
                    continue;

                foreach (var constraint in dimConstraints)
                {
                    if (constraint == null || !constraint.Validate(dim))
                        return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Gets a list of violation messages for dimensions that fail their constraints.
        /// </summary>
        /// <param name="dims">The dimensions to validate.</param>
        /// <param name="constraints">A dictionary mapping dimension names to their constraint lists.</param>
        /// <returns>A list of descriptive error messages for each constraint violation.</returns>
        public List<string> GetViolations(IEnumerable<SymbolicDimension> dims, Dictionary<string, List<IShapeConstraint>> constraints)
        {
            if (dims == null)
                throw new ArgumentNullException(nameof(dims));

            if (constraints == null)
                throw new ArgumentNullException(nameof(constraints));

            var violations = new List<string>();

            foreach (var dim in dims)
            {
                if (dim == null)
                {
                    violations.Add("Null dimension encountered.");
                    continue;
                }

                // Skip dimensions without constraints
                if (!constraints.ContainsKey(dim.Name))
                    continue;

                var dimConstraints = constraints[dim.Name];
                if (dimConstraints == null || dimConstraints.Count == 0)
                    continue;

                // Check cache first
                var cacheKey = $"{dim.Name}:{dim.Value ?? -1}:{dim.MinValue}:{dim.MaxValue}";
                lock (_lock)
                {
                    if (_validationCache.TryGetValue(cacheKey, out var cachedResult))
                    {
                        if (!cachedResult.isValid)
                        {
                            violations.AddRange(cachedResult.violations);
                        }
                        continue;
                    }
                }

                var dimViolations = new List<string>();

                foreach (var constraint in dimConstraints)
                {
                    if (constraint == null)
                    {
                        dimViolations.Add($"Dimension '{dim.Name}': Null constraint encountered.");
                        continue;
                    }

                    if (!constraint.Validate(dim))
                    {
                        string violationMessage;

                        if (!dim.Value.HasValue)
                        {
                            violationMessage = $"Dimension '{dim.Name}' has unknown value - cannot validate constraint '{constraint}'.";
                        }
                        else
                        {
                            violationMessage = $"Dimension '{dim.Name}' with value {dim.Value} does not satisfy constraint '{constraint}'.";
                        }

                        dimViolations.Add(violationMessage);
                    }
                }

                // Cache the result
                lock (_lock)
                {
                    _validationCache[cacheKey] = (dimViolations.Count == 0, new List<string>(dimViolations));
                }

                violations.AddRange(dimViolations);
            }

            return violations;
        }

        /// <summary>
        /// Validates a single dimension against its constraints.
        /// </summary>
        /// <param name="dim">The dimension to validate.</param>
        /// <param name="constraints">The list of constraints to validate against.</param>
        /// <returns>True if the dimension satisfies all constraints; otherwise, false.</returns>
        public bool ValidateDimension(SymbolicDimension dim, List<IShapeConstraint> constraints)
        {
            if (dim == null)
                return false;

            if (constraints == null || constraints.Count == 0)
                return true;

            foreach (var constraint in constraints)
            {
                if (constraint == null || !constraint.Validate(dim))
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Clears the validation cache, forcing all validations to be recomputed.
        /// </summary>
        public void ClearCache()
        {
            lock (_lock)
            {
                _validationCache.Clear();
            }
        }
    }
}
