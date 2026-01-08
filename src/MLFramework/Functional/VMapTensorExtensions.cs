using System;
using System.Collections.Generic;
using System.Linq;
using RitterFramework.Core.Tensor;

namespace MLFramework.Functional
{
    /// <summary>
    /// Extension methods for tensor operations used in vmap transformations.
    /// </summary>
    public static class VMapTensorExtensions
    {
        /// <summary>
        /// Takes a slice from tensor along specified axis at the given index.
        /// This removes the dimension at the specified axis.
        /// </summary>
        /// <param name="tensor">The tensor to slice.</param>
        /// <param name="axis">The axis along which to slice.</param>
        /// <param name="index">The index along the axis.</param>
        /// <returns>A new tensor with the specified slice.</returns>
        /// <exception cref="IndexOutOfRangeException">Thrown when index is out of bounds.</exception>
        public static Tensor Take(this Tensor tensor, int axis, int index)
        {
            if (axis < 0 || axis >= tensor.Dimensions)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(axis),
                    $"Axis {axis} is out of bounds for tensor with {tensor.Dimensions} dimensions");
            }

            if (index < 0 || index >= tensor.Shape[axis])
            {
                throw new ArgumentOutOfRangeException(
                    nameof(index),
                    $"Index {index} is out of bounds for axis {axis} with size {tensor.Shape[axis]}");
            }

            // Create new shape without the axis dimension
            var newShape = new int[tensor.Dimensions - 1];
            int newShapeIdx = 0;
            for (int i = 0; i < tensor.Dimensions; i++)
            {
                if (i != axis)
                {
                    newShape[newShapeIdx++] = tensor.Shape[i];
                }
            }

            // Extract data for the slice
            var newData = new float[GetSliceSize(tensor.Shape, axis)];

            // Calculate strides for navigating the original tensor
            var strides = ComputeStrides(tensor.Shape);

            // Starting offset for the slice
            var startOffset = index * strides[axis];

            // Copy data for the slice
            int newDataIdx = 0;
            int totalSize = tensor.Data.Length;

            for (int i = 0; i < totalSize; i++)
            {
                // Check if this element belongs to the slice
                bool inSlice = ((i / strides[axis]) % tensor.Shape[axis]) == index;

                if (inSlice)
                {
                    newData[newDataIdx++] = tensor.Data[i];
                }
            }

            return new Tensor(newData, newShape, tensor.RequiresGrad, tensor.Dtype);
        }

        /// <summary>
        /// Stacks tensors along a new axis.
        /// </summary>
        /// <param name="tensors">The collection of tensors to stack.</param>
        /// <param name="axis">The axis along which to stack.</param>
        /// <returns>A new tensor with all input tensors stacked along the specified axis.</returns>
        /// <exception cref="ArgumentException">Thrown when tensors have incompatible shapes.</exception>
        public static Tensor Stack(this IEnumerable<Tensor> tensors, int axis)
        {
            var tensorArray = tensors.ToArray();

            if (tensorArray.Length == 0)
            {
                throw new ArgumentException("Cannot stack empty collection of tensors", nameof(tensors));
            }

            if (tensorArray.Length == 1)
            {
                return tensorArray[0].Clone();
            }

            // Validate all tensors have the same shape
            var firstShape = tensorArray[0].Shape;
            foreach (var tensor in tensorArray)
            {
                if (!tensor.Shape.SequenceEqual(firstShape))
                {
                    throw new ArgumentException(
                        $"All tensors must have the same shape for stacking. " +
                        $"Expected {string.Join("x", firstShape)}, " +
                        $"got {string.Join("x", tensor.Shape)}");
                }
            }

            // Calculate axis position in the output tensor
            int outputAxis = axis;
            if (outputAxis < 0)
            {
                outputAxis += firstShape.Length + 1;
            }

            if (outputAxis < 0 || outputAxis > firstShape.Length)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(axis),
                    $"Axis {axis} is out of bounds. Valid range for stacking is [{-firstShape.Length}, {firstShape.Length}]");
            }

            // Create new shape with the added axis
            var newShape = new int[firstShape.Length + 1];
            int newShapeIdx = 0;
            for (int i = 0; i <= firstShape.Length; i++)
            {
                if (i == outputAxis)
                {
                    newShape[newShapeIdx++] = tensorArray.Length;
                }
                else
                {
                    newShape[newShapeIdx++] = firstShape[i - (i > outputAxis ? 1 : 0)];
                }
            }

            // Create new data array
            var newData = new float[tensorArray.Length * tensorArray[0].Data.Length];

            // Copy data from each tensor
            int dataOffset = 0;
            var singleTensorSize = tensorArray[0].Data.Length;

            for (int i = 0; i < tensorArray.Length; i++)
            {
                Array.Copy(tensorArray[i].Data, 0, newData, dataOffset, singleTensorSize);
                dataOffset += singleTensorSize;
            }

            return new Tensor(newData, newShape, tensorArray[0].RequiresGrad, tensorArray[0].Dtype);
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
        /// Calculates the size of a slice when removing one dimension.
        /// </summary>
        private static int GetSliceSize(int[] shape, int axis)
        {
            int size = 1;
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
