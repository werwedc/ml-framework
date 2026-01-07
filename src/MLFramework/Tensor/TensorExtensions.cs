using MLFramework.Core;
using RitterFramework.Core.Tensor;
using System;

namespace RitterFramework.Core.Tensor
{
    /// <summary>
    /// Extension methods for Tensor operations.
    /// </summary>
    public static class TensorExtensions
    {
        /// <summary>
        /// Gets the number of elements in the tensor.
        /// </summary>
        public static int NumElements(this Tensor tensor)
        {
            return tensor.Size;
        }

        /// <summary>
        /// Gets the device where the tensor is located.
        /// For now, all tensors are on CPU.
        /// </summary>
        public static Device GetDevice(this Tensor tensor)
        {
            // For now, assume all tensors are on CPU
            return Device.CreateCpu();
        }

        /// <summary>
        /// Copies the data from another tensor into this tensor (in-place).
        /// </summary>
        public static void Copy_(this Tensor tensor, Tensor other)
        {
            if (tensor.Size != other.Size)
            {
                throw new ArgumentException("Tensor sizes must match for copy operation");
            }

            Array.Copy(other.Data, 0, tensor.Data, 0, tensor.Size);
        }

        /// <summary>
        /// Adds another tensor to this tensor in-place.
        /// </summary>
        public static void Add_(this Tensor tensor, Tensor other)
        {
            if (tensor.Size != other.Size)
            {
                throw new ArgumentException("Tensor sizes must match for add operation");
            }

            for (int i = 0; i < tensor.Size; i++)
            {
                tensor.Data[i] += other.Data[i];
            }
        }

        /// <summary>
        /// Divides this tensor by a scalar in-place.
        /// </summary>
        public static void Div_(this Tensor tensor, float scalar)
        {
            if (scalar == 0)
            {
                throw new DivideByZeroException("Cannot divide by zero");
            }

            for (int i = 0; i < tensor.Size; i++)
            {
                tensor.Data[i] /= scalar;
            }
        }

        /// <summary>
        /// Divides this tensor by an integer in-place.
        /// </summary>
        public static void Div_(this Tensor tensor, int scalar)
        {
            tensor.Div_((float)scalar);
        }

        /// <summary>
        /// Computes the element-wise maximum of two tensors.
        /// </summary>
        public static Tensor Maximum(Tensor a, Tensor b)
        {
            if (a.Size != b.Size)
            {
                throw new ArgumentException("Tensor sizes must match for maximum operation");
            }

            var resultData = new float[a.Size];
            for (int i = 0; i < a.Size; i++)
            {
                resultData[i] = Math.Max(a.Data[i], b.Data[i]);
            }

            return new Tensor(resultData, a.Shape, a.RequiresGrad || b.RequiresGrad, a.Dtype);
        }

        /// <summary>
        /// Moves the tensor to the specified device.
        /// For now, this is a no-op since we only support CPU.
        /// </summary>
        public static Tensor To(this Tensor tensor, Device device)
        {
            // For now, we only support CPU, so just return the tensor itself
            if (device.Type == DeviceType.CPU)
            {
                return tensor;
            }

            // In the future, this would implement actual device transfer
            throw new NotSupportedException($"Device {device.Type} is not yet supported");
        }

        /// <summary>
        /// Slices the tensor.
        /// </summary>
        public static Tensor Slice(this Tensor tensor, int dim, long start, long length)
        {
            // Simplified implementation for 1D tensors
            if (tensor.Dimensions != 1)
            {
                throw new NotSupportedException("Slicing only supports 1D tensors for now");
            }

            if (start < 0 || start >= tensor.Size)
            {
                throw new ArgumentOutOfRangeException(nameof(start));
            }

            if (length <= 0 || start + length > tensor.Size)
            {
                throw new ArgumentOutOfRangeException(nameof(length));
            }

            var resultData = new float[length];
            Array.Copy(tensor.Data, (int)start, resultData, 0, (int)length);

            return new Tensor(resultData, new int[] { (int)length }, tensor.RequiresGrad, tensor.Dtype);
        }

        /// <summary>
        /// Creates a view of the tensor with a different shape.
        /// The data is shared between the original tensor and the view.
        /// </summary>
        /// <param name="tensor">The tensor to view.</param>
        /// <param name="shape">The new shape for the view.</param>
        /// <returns>A tensor with the new shape sharing the same data.</returns>
        public static Tensor View(this Tensor tensor, int[] shape)
        {
            // Verify that the new shape is compatible with the tensor's size
            int newSize = 1;
            foreach (var dim in shape)
            {
                newSize *= dim;
            }

            if (newSize != tensor.Size)
            {
                throw new ArgumentException(
                    $"Shape {string.Join("x", shape)} is incompatible with tensor of size {tensor.Size}");
            }

            // Create a new tensor that shares the same underlying data
            var newShape = new int[shape.Length];
            Array.Copy(shape, newShape, shape.Length);

            return new Tensor(tensor.Data, newShape, tensor.RequiresGrad, tensor.Dtype);
        }

        /// <summary>
        /// Computes element-wise exponential.
        /// </summary>
        public static Tensor Exp(this Tensor tensor)
        {
            var resultData = new float[tensor.Size];
            for (int i = 0; i < tensor.Size; i++)
            {
                resultData[i] = MathF.Exp(tensor.Data[i]);
            }

            return new Tensor(resultData, tensor.Shape, tensor.RequiresGrad, tensor.Dtype);
        }

        /// <summary>
        /// Subtracts another tensor element-wise.
        /// </summary>
        public static Tensor Sub(this Tensor a, Tensor b)
        {
            if (a.Size != b.Size)
            {
                throw new ArgumentException("Tensor sizes must match for subtraction");
            }

            var resultData = new float[a.Size];
            for (int i = 0; i < a.Size; i++)
            {
                resultData[i] = a.Data[i] - b.Data[i];
            }

            return new Tensor(resultData, a.Shape, a.RequiresGrad || b.RequiresGrad, a.Dtype);
        }

        /// <summary>
        /// Subtracts a scalar from each element.
        /// </summary>
        public static Tensor Sub(this Tensor tensor, float scalar)
        {
            var resultData = new float[tensor.Size];
            for (int i = 0; i < tensor.Size; i++)
            {
                resultData[i] = tensor.Data[i] - scalar;
            }

            return new Tensor(resultData, tensor.Shape, tensor.RequiresGrad, tensor.Dtype);
        }

        /// <summary>
        /// Divides another tensor element-wise.
        /// </summary>
        public static Tensor Div(this Tensor a, Tensor b)
        {
            if (a.Size != b.Size)
            {
                throw new ArgumentException("Tensor sizes must match for division");
            }

            var resultData = new float[a.Size];
            for (int i = 0; i < a.Size; i++)
            {
                if (b.Data[i] == 0)
                {
                    throw new DivideByZeroException("Division by zero");
                }
                resultData[i] = a.Data[i] / b.Data[i];
            }

            return new Tensor(resultData, a.Shape, a.RequiresGrad || b.RequiresGrad, a.Dtype);
        }

        /// <summary>
        /// Computes the maximum along a dimension.
        /// </summary>
        public static Tensor Max(this Tensor tensor, int dim = -1, bool keepDim = false)
        {
            if (tensor.Dimensions == 0)
            {
                throw new InvalidOperationException("Cannot compute max of scalar tensor");
            }

            // Handle default dimension (last dimension)
            if (dim < 0)
            {
                dim = tensor.Dimensions - 1;
            }

            if (dim >= tensor.Dimensions)
            {
                throw new ArgumentOutOfRangeException(nameof(dim));
            }

            int dimSize = tensor.Shape[dim];
            int resultSize = tensor.Size / dimSize;

            var resultData = new float[resultSize];
            var resultShape = new int[tensor.Dimensions - 1];

            // Compute result shape
            int shapeIdx = 0;
            for (int i = 0; i < tensor.Dimensions; i++)
            {
                if (i != dim)
                {
                    resultShape[shapeIdx++] = tensor.Shape[i];
                }
            }

            // Compute max along dimension
            for (int i = 0; i < resultSize; i++)
            {
                float maxVal = float.NegativeInfinity;

                for (int j = 0; j < dimSize; j++)
                {
                    int srcIdx = ComputeIndexForReduce(tensor.Shape, dim, i, j);
                    maxVal = Math.Max(maxVal, tensor.Data[srcIdx]);
                }

                resultData[i] = maxVal;
            }

            if (keepDim)
            {
                var keepDimShape = new int[tensor.Dimensions];
                for (int i = 0; i < tensor.Dimensions; i++)
                {
                    keepDimShape[i] = (i == dim) ? 1 : tensor.Shape[i];
                }
                return new Tensor(resultData, keepDimShape, tensor.RequiresGrad, tensor.Dtype);
            }

            return new Tensor(resultData, resultShape, tensor.RequiresGrad, tensor.Dtype);
        }

        /// <summary>
        /// Computes the sum of all elements in the tensor.
        /// Returns a scalar tensor (1-element tensor).
        /// </summary>
        public static Tensor Sum(this Tensor tensor)
        {
            float totalSum = tensor.Data.Sum();
            return new Tensor(new[] { totalSum }, new int[] { 1 }, tensor.RequiresGrad, tensor.Dtype);
        }

        /// <summary>
        /// Computes the sum along a dimension.
        /// </summary>
        public static Tensor Sum(this Tensor tensor, int dim, bool keepDim = false)
        {
            if (tensor.Dimensions == 0)
            {
                throw new InvalidOperationException("Cannot compute sum of scalar tensor");
            }

            // Handle default dimension (last dimension)
            if (dim < 0)
            {
                dim = tensor.Dimensions - 1;
            }

            if (dim >= tensor.Dimensions)
            {
                throw new ArgumentOutOfRangeException(nameof(dim));
            }

            int dimSize = tensor.Shape[dim];
            int resultSize = tensor.Size / dimSize;

            var resultData = new float[resultSize];
            var resultShape = new int[tensor.Dimensions - 1];

            // Compute result shape
            int shapeIdx = 0;
            for (int i = 0; i < tensor.Dimensions; i++)
            {
                if (i != dim)
                {
                    resultShape[shapeIdx++] = tensor.Shape[i];
                }
            }

            // Compute sum along dimension
            for (int i = 0; i < resultSize; i++)
            {
                float sumVal = 0.0f;

                for (int j = 0; j < dimSize; j++)
                {
                    int srcIdx = ComputeIndexForReduce(tensor.Shape, dim, i, j);
                    sumVal += tensor.Data[srcIdx];
                }

                resultData[i] = sumVal;
            }

            if (keepDim)
            {
                var keepDimShape = new int[tensor.Dimensions];
                for (int i = 0; i < tensor.Dimensions; i++)
                {
                    keepDimShape[i] = (i == dim) ? 1 : tensor.Shape[i];
                }
                return new Tensor(resultData, keepDimShape, tensor.RequiresGrad, tensor.Dtype);
            }

            return new Tensor(resultData, resultShape, tensor.RequiresGrad, tensor.Dtype);
        }

        /// <summary>
        /// Transposes the tensor by swapping two dimensions.
        /// </summary>
        public static Tensor Transpose(this Tensor tensor, int dim1, int dim2)
        {
            if (dim1 < 0 || dim1 >= tensor.Dimensions)
            {
                throw new ArgumentOutOfRangeException(nameof(dim1));
            }
            if (dim2 < 0 || dim2 >= tensor.Dimensions)
            {
                throw new ArgumentOutOfRangeException(nameof(dim2));
            }

            // Create new shape with swapped dimensions
            var newShape = new int[tensor.Dimensions];
            Array.Copy(tensor.Shape, newShape, tensor.Dimensions);
            newShape[dim1] = tensor.Shape[dim2];
            newShape[dim2] = tensor.Shape[dim1];

            var resultData = new float[tensor.Size];

            // Compute strides for both original and transposed layouts
            var strides = ComputeStrides(tensor.Shape);
            var newStrides = ComputeStrides(newShape);

            // Transpose data
            for (int i = 0; i < tensor.Size; i++)
            {
                // Convert flat index to multi-dimensional indices
                var indices = UnflattenIndex(i, strides);

                // Swap the two dimensions
                (indices[dim1], indices[dim2]) = (indices[dim2], indices[dim1]);

                // Compute new flat index
                int newIdx = FlattenIndex(indices, newStrides);
                resultData[newIdx] = tensor.Data[i];
            }

            return new Tensor(resultData, newShape, tensor.RequiresGrad, tensor.Dtype);
        }

        /// <summary>
        /// Performs matrix multiplication between two tensors.
        /// Both tensors must be 2D or batched 3D.
        /// </summary>
        public static Tensor Matmul(Tensor a, Tensor b)
        {
            if (a.Dimensions != b.Dimensions)
            {
                throw new ArgumentException("Tensors must have the same number of dimensions");
            }

            if (a.Dimensions == 2)
            {
                // Standard matrix multiplication: (M x K) @ (K x N) = (M x N)
                int m = a.Shape[0];
                int k = a.Shape[1];
                int n = b.Shape[1];

                if (k != b.Shape[0])
                {
                    throw new ArgumentException($"Matrix dimensions incompatible: ({a.Shape[0]}, {a.Shape[1]}) @ ({b.Shape[0]}, {b.Shape[1]})");
                }

                var resultData = new float[m * n];

                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        float sum = 0.0f;
                        for (int l = 0; l < k; l++)
                        {
                            sum += a.Data[i * k + l] * b.Data[l * n + j];
                        }
                        resultData[i * n + j] = sum;
                    }
                }

                return new Tensor(resultData, new int[] { m, n }, a.RequiresGrad || b.RequiresGrad, a.Dtype);
            }
            else if (a.Dimensions == 3)
            {
                // Batched matrix multiplication: (B x M x K) @ (B x K x N) = (B x M x N)
                int batchSize = a.Shape[0];
                int m = a.Shape[1];
                int k = a.Shape[2];
                int n = b.Shape[2];

                if (k != b.Shape[1] || batchSize != b.Shape[0])
                {
                    throw new ArgumentException($"Batched matrix dimensions incompatible: ({a.Shape[0]}, {a.Shape[1]}, {a.Shape[2]}) @ ({b.Shape[0]}, {b.Shape[1]}, {b.Shape[2]})");
                }

                var resultData = new float[batchSize * m * n];

                for (int bIdx = 0; bIdx < batchSize; bIdx++)
                {
                    for (int i = 0; i < m; i++)
                    {
                        for (int j = 0; j < n; j++)
                        {
                            float sum = 0.0f;
                            for (int l = 0; l < k; l++)
                            {
                                int aIdx = bIdx * m * k + i * k + l;
                                int bIdx_ = bIdx * k * n + l * n + j;
                                sum += a.Data[aIdx] * b.Data[bIdx_];
                            }
                            resultData[bIdx * m * n + i * n + j] = sum;
                        }
                    }
                }

                return new Tensor(resultData, new int[] { batchSize, m, n }, a.RequiresGrad || b.RequiresGrad, a.Dtype);
            }
            else
            {
                throw new NotSupportedException($"Matrix multiplication not supported for {a.Dimensions}D tensors");
            }
        }

        #region Helper Methods

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

        private static int[] UnflattenIndex(int flatIdx, int[] strides)
        {
            var indices = new int[strides.Length];
            for (int i = 0; i < strides.Length; i++)
            {
                indices[i] = flatIdx / strides[i];
                flatIdx %= strides[i];
            }
            return indices;
        }

        private static int FlattenIndex(int[] indices, int[] strides)
        {
            int idx = 0;
            for (int i = 0; i < indices.Length; i++)
            {
                idx += indices[i] * strides[i];
            }
            return idx;
        }

        private static int ComputeIndexForReduce(int[] shape, int reduceDim, int outerIdx, int innerIdx)
        {
            // Compute strides
            var strides = ComputeStrides(shape);

            // Convert outerIdx to indices in all dimensions except reduceDim
            var indices = new int[shape.Length];
            int temp = outerIdx;

            for (int i = 0; i < shape.Length; i++)
            {
                if (i == reduceDim) continue;

                indices[i] = temp % shape[i];
                temp /= shape[i];
            }

            // Set the reduced dimension
            indices[reduceDim] = innerIdx;

            // Compute flat index
            return FlattenIndex(indices, strides);
        }

        #endregion
    }
}
