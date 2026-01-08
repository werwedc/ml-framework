using System;
using System.Collections.Generic;
using System.Linq;
using MLFramework.Shapes;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tensors
{
    /// <summary>
    /// Extension methods for creating and working with symbolic tensors.
    /// </summary>
    public static class TensorSymbolic
    {
        /// <summary>
        /// Creates a symbolic tensor with the given symbolic dimensions.
        /// </summary>
        /// <param name="dims">The symbolic dimensions.</param>
        /// <returns>A SymbolicTensor with the specified dimensions.</returns>
        public static SymbolicTensor Symbolic(params SymbolicDimension[] dims)
        {
            if (dims == null || dims.Length == 0)
            {
                throw new ArgumentException("At least one dimension is required", nameof(dims));
            }

            var shape = new SymbolicShape(dims);
            return new SymbolicTensor(shape);
        }

        /// <summary>
        /// Creates a symbolic tensor with the given symbolic shape.
        /// </summary>
        /// <param name="shape">The symbolic shape.</param>
        /// <returns>A SymbolicTensor with the specified shape.</returns>
        public static SymbolicTensor Symbolic(SymbolicShape shape)
        {
            if (shape == null)
            {
                throw new ArgumentNullException(nameof(shape));
            }

            return new SymbolicTensor(shape);
        }

        /// <summary>
        /// Creates a symbolic tensor representing a tensor filled with zeros.
        /// Throws if dimensions are not fully known.
        /// </summary>
        /// <param name="dims">The symbolic dimensions.</param>
        /// <returns>A SymbolicTensor with zero-filling semantics.</returns>
        /// <exception cref="InvalidOperationException">Thrown when any dimension is not fully known.</exception>
        public static SymbolicTensor Zeros(params SymbolicDimension[] dims)
        {
            if (dims == null || dims.Length == 0)
            {
                throw new ArgumentException("At least one dimension is required", nameof(dims));
            }

            // Validate all dimensions are known
            foreach (var dim in dims)
            {
                if (!dim.IsKnown())
                {
                    throw new InvalidOperationException(
                        $"Cannot create zeros tensor with unknown dimension: {dim}. " +
                        "All dimensions must have concrete values.");
                }
            }

            var shape = new SymbolicShape(dims);
            return new SymbolicTensor(shape, TensorFillType.Zeros);
        }

        /// <summary>
        /// Creates a symbolic tensor representing a tensor filled with ones.
        /// Throws if dimensions are not fully known.
        /// </summary>
        /// <param name="dims">The symbolic dimensions.</param>
        /// <returns>A SymbolicTensor with one-filling semantics.</returns>
        /// <exception cref="InvalidOperationException">Thrown when any dimension is not fully known.</exception>
        public static SymbolicTensor Ones(params SymbolicDimension[] dims)
        {
            if (dims == null || dims.Length == 0)
            {
                throw new ArgumentException("At least one dimension is required", nameof(dims));
            }

            // Validate all dimensions are known
            foreach (var dim in dims)
            {
                if (!dim.IsKnown())
                {
                    throw new InvalidOperationException(
                        $"Cannot create ones tensor with unknown dimension: {dim}. " +
                        "All dimensions must have concrete values.");
                }
            }

            var shape = new SymbolicShape(dims);
            return new SymbolicTensor(shape, TensorFillType.Ones);
        }

        /// <summary>
        /// Creates a symbolic tensor representing a tensor filled with random values.
        /// Throws if dimensions are not fully known.
        /// </summary>
        /// <param name="dims">The symbolic dimensions.</param>
        /// <returns>A SymbolicTensor with random-filling semantics.</returns>
        /// <exception cref="InvalidOperationException">Thrown when any dimension is not fully known.</exception>
        public static SymbolicTensor Random(params SymbolicDimension[] dims)
        {
            if (dims == null || dims.Length == 0)
            {
                throw new ArgumentException("At least one dimension is required", nameof(dims));
            }

            // Validate all dimensions are known
            foreach (var dim in dims)
            {
                if (!dim.IsKnown())
                {
                    throw new InvalidOperationException(
                        $"Cannot create random tensor with unknown dimension: {dim}. " +
                        "All dimensions must have concrete values.");
                }
            }

            var shape = new SymbolicShape(dims);
            return new SymbolicTensor(shape, TensorFillType.Random);
        }

        /// <summary>
        /// Attaches a constraint to a dimension of the symbolic tensor.
        /// </summary>
        /// <param name="tensor">The symbolic tensor.</param>
        /// <param name="dimName">The name of the dimension to constrain.</param>
        /// <param name="constraint">The constraint to apply.</param>
        /// <returns>A new SymbolicTensor with the constraint added.</returns>
        public static SymbolicTensor ShapeHint(this SymbolicTensor tensor, string dimName, IShapeConstraint constraint)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            if (string.IsNullOrWhiteSpace(dimName))
            {
                throw new ArgumentException("Dimension name cannot be empty", nameof(dimName));
            }

            if (constraint == null)
            {
                throw new ArgumentNullException(nameof(constraint));
            }

            return tensor.WithConstraint(dimName, constraint);
        }

        /// <summary>
        /// Attaches bounds constraints to a dimension of the symbolic tensor.
        /// </summary>
        /// <param name="tensor">The symbolic tensor.</param>
        /// <param name="dimName">The name of the dimension to bound.</param>
        /// <param name="min">The minimum value.</param>
        /// <param name="max">The maximum value, or null for unbounded.</param>
        /// <returns>A new SymbolicTensor with the bounds added.</returns>
        public static SymbolicTensor WithBounds(this SymbolicTensor tensor, string dimName, int min, int? max = null)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            if (string.IsNullOrWhiteSpace(dimName))
            {
                throw new ArgumentException("Dimension name cannot be empty", nameof(dimName));
            }

            // Use int.MaxValue for unbounded (null max)
            var maxVal = max ?? int.MaxValue;
            var constraint = new RangeConstraint(min, maxVal);
            return tensor.WithConstraint(dimName, constraint);
        }

        /// <summary>
        /// Gets the symbolic shape of the tensor.
        /// </summary>
        /// <param name="tensor">The tensor (must be a SymbolicTensor).</param>
        /// <returns>The symbolic shape of the tensor.</returns>
        /// <exception cref="InvalidCastException">Thrown when tensor is not a SymbolicTensor.</exception>
        public static SymbolicShape GetShape(this SymbolicTensor tensor)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            return tensor.SymbolicShape;
        }

        /// <summary>
        /// Resizes the tensor to concrete dimensions at runtime.
        /// </summary>
        /// <param name="tensor">The symbolic tensor.</param>
        /// <param name="concreteDims">The concrete dimensions to resize to.</param>
        /// <returns>A new Tensor with concrete data.</returns>
        public static Tensor ResizeTo(this SymbolicTensor tensor, params int[] concreteDims)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            if (concreteDims == null || concreteDims.Length == 0)
            {
                throw new ArgumentException("At least one dimension is required", nameof(concreteDims));
            }

            return tensor.ToConcrete(concreteDims);
        }

        /// <summary>
        /// Resizes the tensor to a symbolic shape.
        /// </summary>
        /// <param name="tensor">The symbolic tensor.</param>
        /// <param name="symbolicDims">The symbolic dimensions to resize to.</param>
        /// <returns>A new SymbolicTensor with the new shape.</returns>
        public static SymbolicTensor ResizeTo(this SymbolicTensor tensor, params SymbolicDimension[] symbolicDims)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            if (symbolicDims == null || symbolicDims.Length == 0)
            {
                throw new ArgumentException("At least one dimension is required", nameof(symbolicDims));
            }

            var newShape = new SymbolicShape(symbolicDims);
            return tensor.WithShape(newShape);
        }

        /// <summary>
        /// Gets a specific named dimension from the tensor's shape.
        /// </summary>
        /// <param name="tensor">The symbolic tensor.</param>
        /// <param name="name">The name of the dimension to get.</param>
        /// <returns>The symbolic dimension with the specified name.</returns>
        /// <exception cref="InvalidOperationException">Thrown when no dimension with the given name exists.</exception>
        public static SymbolicDimension GetDimension(this SymbolicTensor tensor, string name)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            if (string.IsNullOrWhiteSpace(name))
            {
                throw new ArgumentException("Dimension name cannot be empty", nameof(name));
            }

            return tensor.SymbolicShape.Dimensions.FirstOrDefault(d => d.Name == name)
                ?? throw new InvalidOperationException($"No dimension named '{name}' found in tensor shape {tensor.SymbolicShape}");
        }
    }
}
