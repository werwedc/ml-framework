using System;
using System.Collections.Generic;

namespace MLFramework.IR
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.Types;
    using MLFramework.IR.Values;

    /// <summary>
    /// IRContext manages the creation and tracking of values and operations in the IR system.
    /// It acts as a factory for IR constructs and maintains their unique IDs.
    /// </summary>
    public class IRContext
    {
        private int _nextValueId = 0;
        private Dictionary<int, IRValue> _values;
        private Dictionary<int, IROperation> _operations;

        /// <summary>
        /// Initializes a new instance of the IRContext class.
        /// </summary>
        public IRContext()
        {
            _values = new Dictionary<int, IRValue>();
            _operations = new Dictionary<int, IROperation>();
        }

        /// <summary>
        /// Creates a new IRValue with the specified type and optional name.
        /// </summary>
        /// <param name="type">The type of the value.</param>
        /// <param name="name">The optional name for the value.</param>
        /// <returns>A new IRValue instance.</returns>
        public IRValue CreateValue(IIRType type, string name = null)
        {
            if (type == null)
            {
                throw new ArgumentNullException(nameof(type));
            }

            var value = new IRValue(type, name);
            _values[value.Id] = value;
            return value;
        }

        /// <summary>
        /// Registers an operation in this context.
        /// </summary>
        /// <param name="op">The operation to register.</param>
        public void RegisterOperation(IROperation op)
        {
            if (op == null)
            {
                throw new ArgumentNullException(nameof(op));
            }

            // For now, use a hash code as the operation ID
            int opId = op.GetHashCode();
            _operations[opId] = op;

            // Also register all result values
            foreach (var result in op.Results)
            {
                _values[result.Id] = result;
            }
        }

        /// <summary>
        /// Gets a value by its ID.
        /// </summary>
        /// <param name="id">The ID of the value.</param>
        /// <returns>The IRValue with the specified ID, or null if not found.</returns>
        public IRValue GetValue(int id)
        {
            return _values.TryGetValue(id, out var value) ? value : null;
        }

        /// <summary>
        /// Gets an operation by its ID.
        /// </summary>
        /// <param name="id">The ID of the operation.</param>
        /// <returns>The IROperation with the specified ID, or null if not found.</returns>
        public IROperation GetOperation(int id)
        {
            return _operations.TryGetValue(id, out var operation) ? operation : null;
        }

        /// <summary>
        /// Gets all values registered in this context.
        /// </summary>
        /// <returns>A read-only collection of all values.</returns>
        public IReadOnlyCollection<IRValue> GetAllValues()
        {
            return _values.Values;
        }

        /// <summary>
        /// Gets all operations registered in this context.
        /// </summary>
        /// <returns>A read-only collection of all operations.</returns>
        public IReadOnlyCollection<IROperation> GetAllOperations()
        {
            return _operations.Values;
        }

        /// <summary>
        /// Clears all values and operations from this context.
        /// </summary>
        public void Clear()
        {
            _values.Clear();
            _operations.Clear();
            _nextValueId = 0;
        }
    }
}
