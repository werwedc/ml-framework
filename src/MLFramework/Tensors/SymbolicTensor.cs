using System;
using System.Collections.Generic;
using System.Linq;
using MLFramework.Shapes;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tensors
{
    /// <summary>
    /// Specifies how a tensor should be filled when materialized from symbolic form.
    /// </summary>
    public enum TensorFillType
    {
        /// <summary>
        /// No fill semantics - pure placeholder.
        /// </summary>
        None,

        /// <summary>
        /// Fill with zeros.
        /// </summary>
        Zeros,

        /// <summary>
        /// Fill with ones.
        /// </summary>
        Ones,

        /// <summary>
        /// Fill with random values.
        /// </summary>
        Random
    }

    /// <summary>
    /// Represents a symbolic tensor - a placeholder for a tensor that may not have concrete shape information yet.
    /// This class is not usable for actual computation until shapes are bound to concrete values.
    /// </summary>
    public sealed class SymbolicTensor
    {
        /// <summary>
        /// Gets the symbolic shape of this tensor.
        /// </summary>
        public SymbolicShape SymbolicShape { get; }

        /// <summary>
        /// Gets the constraints applied to each dimension.
        /// Dimension name -> list of constraints.
        /// </summary>
        public Dictionary<string, List<IShapeConstraint>> Constraints { get; }

        /// <summary>
        /// Gets the fill type for this tensor.
        /// </summary>
        public TensorFillType FillType { get; }

        /// <summary>
        /// Initializes a new instance of the SymbolicTensor class.
        /// </summary>
        /// <param name="shape">The symbolic shape.</param>
        /// <param name="fillType">The fill type (defaults to None).</param>
        public SymbolicTensor(SymbolicShape shape, TensorFillType fillType = TensorFillType.None)
        {
            SymbolicShape = shape ?? throw new ArgumentNullException(nameof(shape));
            FillType = fillType;
            Constraints = new Dictionary<string, List<IShapeConstraint>>();
        }

        /// <summary>
        /// Initializes a new instance of the SymbolicTensor class with constraints.
        /// </summary>
        /// <param name="shape">The symbolic shape.</param>
        /// <param name="constraints">The constraints for each dimension.</param>
        /// <param name="fillType">The fill type (defaults to None).</param>
        private SymbolicTensor(SymbolicShape shape, Dictionary<string, List<IShapeConstraint>> constraints, TensorFillType fillType = TensorFillType.None)
        {
            SymbolicShape = shape ?? throw new ArgumentNullException(nameof(shape));
            Constraints = constraints ?? new Dictionary<string, List<IShapeConstraint>>();
            FillType = fillType;
        }

        /// <summary>
        /// Validates that all constraints are satisfied by the current shape.
        /// </summary>
        /// <returns>True if all constraints are satisfied; otherwise, false.</returns>
        public bool ValidateShape()
        {
            foreach (var kvp in Constraints)
            {
                var dimName = kvp.Key;
                var constraints = kvp.Value;

                // Find the dimension with this name
                var dim = SymbolicShape.Dimensions.FirstOrDefault(d => d.Name == dimName);
                if (dim == null)
                {
                    return false;
                }

                // Validate each constraint
                foreach (var constraint in constraints)
                {
                    if (!constraint.Validate(dim))
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        /// <summary>
        /// Substitutes symbolic dimensions with concrete values.
        /// </summary>
        /// <param name="values">The concrete values to substitute.</param>
        /// <returns>An array of concrete dimensions.</returns>
        /// <exception cref="InvalidOperationException">
        /// Thrown when the number of values doesn't match the rank,
        /// or when constraints are not satisfied.
        /// </exception>
        public int[] GetConcreteShape(params int[] values)
        {
            if (values == null || values.Length != SymbolicShape.Rank)
            {
                throw new InvalidOperationException(
                    $"Expected {SymbolicShape.Rank} concrete values, got {values?.Length ?? 0}");
            }

            // Check if the shape can be instantiated with these values
            if (!CanInstantiateWith(values))
            {
                throw new InvalidOperationException(
                    $"Cannot instantiate tensor with shape [{string.Join(", ", values)}]. " +
                    "Constraints are not satisfied.");
            }

            return (int[])values.Clone();
        }

        /// <summary>
        /// Checks if the tensor can be instantiated with the given concrete dimensions.
        /// </summary>
        /// <param name="dims">The concrete dimensions to check.</param>
        /// <returns>True if the tensor can be instantiated; otherwise, false.</returns>
        public bool CanInstantiateWith(params int[] dims)
        {
            if (dims == null || dims.Length != SymbolicShape.Rank)
            {
                return false;
            }

            // Check each dimension against constraints
            for (int i = 0; i < dims.Length; i++)
            {
                var dim = SymbolicShape.Dimensions[i];
                var value = dims[i];

                // Check if constraints exist for this dimension
                if (Constraints.TryGetValue(dim.Name, out var constraints))
                {
                    // Create a temporary symbolic dimension with the concrete value
                    var tempDim = new SymbolicDimension(dim.Name, value, dim.MinValue, dim.MaxValue);

                    // Validate each constraint
                    foreach (var constraint in constraints)
                    {
                        if (!constraint.Validate(tempDim))
                        {
                            return false;
                        }
                    }
                }
            }

            return true;
        }

        /// <summary>
        /// Materializes this symbolic tensor into a concrete Tensor with the given dimensions.
        /// </summary>
        /// <param name="concreteDims">The concrete dimensions.</param>
        /// <returns>A concrete Tensor with actual data.</returns>
        /// <exception cref="InvalidOperationException">
        /// Thrown when constraints are not satisfied or fill type is unsupported.
        /// </exception>
        public Tensor ToConcrete(params int[] concreteDims)
        {
            if (!CanInstantiateWith(concreteDims))
            {
                throw new InvalidOperationException(
                    $"Cannot instantiate tensor with shape [{string.Join(", ", concreteDims)}]. " +
                    "Constraints are not satisfied.");
            }

            return FillType switch
            {
                TensorFillType.Zeros => Tensor.Zeros(concreteDims),
                TensorFillType.Ones => Tensor.Ones(concreteDims),
                TensorFillType.Random => CreateRandomTensor(concreteDims),
                TensorFillType.None => throw new InvalidOperationException(
                    "Cannot materialize symbolic tensor with no fill semantics. " +
                    "Use Zeros, Ones, or Random to specify fill behavior."),
                _ => throw new NotSupportedException($"Unsupported fill type: {FillType}")
            };
        }

        /// <summary>
        /// Returns a new SymbolicTensor with a constraint added to a dimension.
        /// </summary>
        /// <param name="dimName">The name of the dimension.</param>
        /// <param name="constraint">The constraint to add.</param>
        ///returns>A new SymbolicTensor with the constraint added.</returns>
        internal SymbolicTensor WithConstraint(string dimName, IShapeConstraint constraint)
        {
            var newConstraints = CloneConstraints();

            if (!newConstraints.TryGetValue(dimName, out var constraints))
            {
                constraints = new List<IShapeConstraint>();
                newConstraints[dimName] = constraints;
            }

            constraints.Add(constraint);

            return new SymbolicTensor(SymbolicShape, newConstraints, FillType);
        }

        /// <summary>
        /// Returns a new SymbolicTensor with a different shape.
        /// </summary>
        /// <param name="newShape">The new symbolic shape.</param>
        /// <returns>A new SymbolicTensor with the new shape.</returns>
        internal SymbolicTensor WithShape(SymbolicShape newShape)
        {
            return new SymbolicTensor(newShape, CloneConstraints(), FillType);
        }

        /// <summary>
        /// Creates a deep copy of the constraints dictionary.
        /// </summary>
        /// <returns>A new dictionary with cloned constraint lists.</returns>
        private Dictionary<string, List<IShapeConstraint>> CloneConstraints()
        {
            var cloned = new Dictionary<string, List<IShapeConstraint>>();

            foreach (var kvp in Constraints)
            {
                cloned[kvp.Key] = new List<IShapeConstraint>(kvp.Value);
            }

            return cloned;
        }

        /// <summary>
        /// Creates a tensor filled with random values.
        /// </summary>
        /// <param name="concreteDims">The concrete dimensions.</param>
        /// <returns>A Tensor with random values.</returns>
        private Tensor CreateRandomTensor(int[] concreteDims)
        {
            var random = new Random();
            long size = 1;
            foreach (var dim in concreteDims)
            {
                size *= dim;
            }

            var data = new float[size];
            for (int i = 0; i < size; i++)
            {
                data[i] = (float)random.NextDouble();
            }

            return new Tensor(data, concreteDims, false);
        }

        /// <summary>
        /// Returns a string representation of this symbolic tensor.
        /// </summary>
        /// <returns>A string showing the shape and constraints.</returns>
        public override string ToString()
        {
            var constraintsStr = Constraints.Count > 0
                ? $", Constraints: {Constraints.Count}"
                : "";

            var fillTypeStr = FillType != TensorFillType.None
                ? $", Fill: {FillType}"
                : "";

            return $"SymbolicTensor(Shape: {SymbolicShape}{constraintsStr}{fillTypeStr})";
        }
    }
}
