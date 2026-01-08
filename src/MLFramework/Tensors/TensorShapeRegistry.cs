using System;
using System.Collections.Generic;

namespace MLFramework.Tensors
{
    /// <summary>
    /// Manages bindings between symbolic dimension names and concrete values.
    /// This registry is used to track the concrete values assigned to symbolic dimensions
    /// during tensor materialization.
    /// </summary>
    public sealed class TensorShapeRegistry
    {
        private readonly Dictionary<string, int> _bindings;

        /// <summary>
        /// Gets the number of bindings in this registry.
        /// </summary>
        public int Count => _bindings.Count;

        /// <summary>
        /// Initializes a new instance of the TensorShapeRegistry class.
        /// </summary>
        public TensorShapeRegistry()
        {
            _bindings = new Dictionary<string, int>(StringComparer.Ordinal);
        }

        /// <summary>
        /// Initializes a new instance of the TensorShapeRegistry class with existing bindings.
        /// </summary>
        /// <param name="bindings">The initial bindings.</param>
        private TensorShapeRegistry(Dictionary<string, int> bindings)
        {
            _bindings = new Dictionary<string, int>(bindings, StringComparer.Ordinal);
        }

        /// <summary>
        /// Registers a binding between a symbolic dimension name and a concrete value.
        /// </summary>
        /// <param name="dimName">The name of the symbolic dimension.</param>
        /// <param name="value">The concrete value to bind.</param>
        /// <exception cref="ArgumentNullException">Thrown when dimName is null or empty.</exception>
        /// <exception cref="ArgumentException">Thrown when value is negative.</exception>
        public void RegisterBinding(string dimName, int value)
        {
            if (string.IsNullOrWhiteSpace(dimName))
            {
                throw new ArgumentNullException(nameof(dimName), "Dimension name cannot be null or empty.");
            }

            if (value < 0)
            {
                throw new ArgumentException("Value must be non-negative.", nameof(value));
            }

            _bindings[dimName] = value;
        }

        /// <summary>
        /// Gets the concrete value bound to a symbolic dimension name.
        /// </summary>
        /// <param name="dimName">The name of the symbolic dimension.</param>
        /// <returns>The concrete value if found; otherwise, null.</returns>
        public int? GetBinding(string dimName)
        {
            if (string.IsNullOrWhiteSpace(dimName))
            {
                throw new ArgumentNullException(nameof(dimName), "Dimension name cannot be null or empty.");
            }

            if (_bindings.TryGetValue(dimName, out var value))
            {
                return value;
            }

            return null;
        }

        /// <summary>
        /// Checks if a binding exists for the given dimension name.
        /// </summary>
        /// <param name="dimName">The name of the symbolic dimension.</param>
        /// <returns>True if a binding exists; otherwise, false.</returns>
        public bool HasBinding(string dimName)
        {
            if (string.IsNullOrWhiteSpace(dimName))
            {
                throw new ArgumentNullException(nameof(dimName), "Dimension name cannot be null or empty.");
            }

            return _bindings.ContainsKey(dimName);
        }

        /// <summary>
        /// Removes a binding for the given dimension name.
        /// </summary>
        /// <param name="dimName">The name of the symbolic dimension.</param>
        /// <returns>True if the binding was removed; false if it didn't exist.</returns>
        public bool RemoveBinding(string dimName)
        {
            if (string.IsNullOrWhiteSpace(dimName))
            {
                throw new ArgumentNullException(nameof(dimName), "Dimension name cannot be null or empty.");
            }

            return _bindings.Remove(dimName);
        }

        /// <summary>
        /// Clears all bindings from the registry.
        /// </summary>
        public void ClearBindings()
        {
            _bindings.Clear();
        }

        /// <summary>
        /// Creates a deep clone of this registry.
        /// </summary>
        /// <returns>A new TensorShapeRegistry with the same bindings.</returns>
        public TensorShapeRegistry Clone()
        {
            return new TensorShapeRegistry(new Dictionary<string, int>(_bindings, StringComparer.Ordinal));
        }

        /// <summary>
        /// Applies bindings from another registry to this one.
        /// Existing bindings are overwritten if there are conflicts.
        /// </summary>
        /// <param name="other">The registry to merge from.</param>
        public void MergeFrom(TensorShapeRegistry other)
        {
            if (other == null)
            {
                throw new ArgumentNullException(nameof(other));
            }

            foreach (var kvp in other._bindings)
            {
                _bindings[kvp.Key] = kvp.Value;
            }
        }

        /// <summary>
        /// Gets all dimension names that have bindings.
        /// </summary>
        /// <returns>A collection of dimension names with bindings.</returns>
        public IEnumerable<string> GetBoundDimensionNames()
        {
            return _bindings.Keys;
        }

        /// <summary>
        /// Returns a string representation of this registry.
        /// </summary>
        /// <returns>A string showing all bindings.</returns>
        public override string ToString()
        {
            if (_bindings.Count == 0)
            {
                return "TensorShapeRegistry: No bindings";
            }

            var bindings = string.Join(", ", _bindings.Select(kvp => $"{kvp.Key}={kvp.Value}"));
            return $"TensorShapeRegistry: {{{bindings}}}";
        }
    }
}
