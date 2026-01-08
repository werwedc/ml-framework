using System;
using System.Collections.Generic;
using System.Linq;
using RitterFramework.Core.Tensor;

namespace MLFramework.Functional.Distributed
{
    /// <summary>
    /// Extension methods for tensor operations used in pmap transformations.
    /// </summary>
    public static class PMapTensorExtensions
    {
        /// <summary>
        /// Slice tensor along axis with start and end indices.
        /// </summary>
        /// <param name="tensor">The tensor to slice.</param>
        /// <param name="axis">The axis along which to slice.</param>
        /// <param name="start">The start index (inclusive).</param>
        /// <param name="end">The end index (exclusive).</param>
        /// <returns>A sliced tensor along the specified axis.</returns>
        public static Tensor Slice(this Tensor tensor, int axis, int start, int end)
        {
            if (axis < 0 || axis >= tensor.Dimensions)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(axis),
                    $"Axis {axis} is out of bounds for tensor with {tensor.Dimensions} dimensions");
            }

            if (start < 0 || start > tensor.Shape[axis])
            {
                throw new ArgumentOutOfRangeException(
                    nameof(start),
                    $"Start index {start} is out of bounds for axis {axis} with size {tensor.Shape[axis]}");
            }

            if (end < 0 || end > tensor.Shape[axis])
            {
                throw new ArgumentOutOfRangeException(
                    nameof(end),
                    $"End index {end} is out of bounds for axis {axis} with size {tensor.Shape[axis]}");
            }

            if (start >= end)
            {
                throw new ArgumentException($"Start index {start} must be less than end index {end}");
            }

            // Create new shape with the sliced dimension
            var newShape = new int[tensor.Dimensions];
            for (int i = 0; i < tensor.Dimensions; i++)
            {
                if (i == axis)
                {
                    newShape[i] = end - start;
                }
                else
                {
                    newShape[i] = tensor.Shape[i];
                }
            }

            // Calculate strides for navigating the original tensor
            var strides = ComputeStrides(tensor.Shape);

            // Copy data for the slice
            var newData = new float[GetSliceSize(tensor.Shape, axis, start, end)];
            int newDataIdx = 0;
            int totalSize = tensor.Data.Length;

            for (int i = 0; i < totalSize; i++)
            {
                // Determine the position along the axis for this element
                int axisPos = (i / strides[axis]) % tensor.Shape[axis];

                // Check if this element belongs to the slice
                if (axisPos >= start && axisPos < end)
                {
                    newData[newDataIdx++] = tensor.Data[i];
                }
            }

            return new Tensor(newData, newShape, tensor.RequiresGrad, tensor.Dtype);
        }

        /// <summary>
        /// Concatenate tensors along axis.
        /// </summary>
        /// <param name="tensors">The collection of tensors to concatenate.</param>
        /// <param name="axis">The axis along which to concatenate.</param>
        /// <returns>A new tensor with all input tensors concatenated along the specified axis.</returns>
        public static Tensor Concat(Tensor[] tensors, int axis)
        {
            if (tensors == null || tensors.Length == 0)
            {
                throw new ArgumentException("Cannot concatenate empty collection of tensors", nameof(tensors));
            }

            if (tensors.Length == 1)
            {
                return tensors[0].Clone();
            }

            // Validate all tensors have the same number of dimensions
            int dimensions = tensors[0].Dimensions;
            foreach (var tensor in tensors)
            {
                if (tensor.Dimensions != dimensions)
                {
                    throw new ArgumentException(
                        $"All tensors must have the same number of dimensions. " +
                        $"Expected {dimensions}, got {tensor.Dimensions}");
                }

                if (axis < 0 || axis >= dimensions)
                {
                    throw new ArgumentOutOfRangeException(
                        nameof(axis),
                        $"Axis {axis} is out of bounds for tensors with {dimensions} dimensions");
                }
            }

            // Validate all dimensions except the concatenation axis match
            var referenceShape = tensors[0].Shape;
            for (int i = 1; i < tensors.Length; i++)
            {
                for (int dim = 0; dim < dimensions; dim++)
                {
                    if (dim != axis && tensors[i].Shape[dim] != referenceShape[dim])
                    {
                        throw new ArgumentException(
                            $"All tensors must have the same size in dimension {dim}. " +
                            $"Expected {referenceShape[dim]}, got {tensors[i].Shape[dim]}");
                    }
                }
            }

            // Create new shape with the concatenated dimension
            var newShape = new int[dimensions];
            int concatAxisSize = 0;
            for (int dim = 0; dim < dimensions; dim++)
            {
                if (dim == axis)
                {
                    for (int i = 0; i < tensors.Length; i++)
                    {
                        concatAxisSize += tensors[i].Shape[dim];
                    }
                    newShape[dim] = concatAxisSize;
                }
                else
                {
                    newShape[dim] = referenceShape[dim];
                }
            }

            // Create new data array
            var newData = new float[newShape.Aggregate(1, (a, b) => a * b)];
            int dataOffset = 0;

            for (int i = 0; i < tensors.Length; i++)
            {
                Array.Copy(tensors[i].Data, 0, newData, dataOffset, tensors[i].Data.Length);
                dataOffset += tensors[i].Data.Length;
            }

            return new Tensor(newData, newShape, tensors[0].RequiresGrad, tensors[0].Dtype);
        }

        /// <summary>
        /// Computes the strides for a given shape.
        /// </summary>
        private static int[] ComputeStrides(int[] shape)
        {
            var strides = new int[shape.Length];
            var stride = 1;

            for (var i = shape.Length - 1; i >= 0; i--)
            {
                strides[i] = stride;
                stride *= shape[i];
            }

            return strides;
        }

        /// <summary>
        /// Calculates the size of a slice when taking a subset of one dimension.
        /// </summary>
        private static int GetSliceSize(int[] shape, int axis, int start, int end)
        {
            int size = end - start;
            for (int i = 0; i < shape.Length; i++)
            {
                if (i != axis)
                {
                    size *= shape[i];
                }
            }
            return size;
        }
    }
}
