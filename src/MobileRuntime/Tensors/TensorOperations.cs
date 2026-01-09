using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace MobileRuntime
{
    /// <summary>
    /// Static class providing lightweight tensor operations optimized for mobile/edge devices
    /// </summary>
    public static class TensorOperations
    {
        #region Arithmetic Operations

        /// <summary>
        /// Element-wise addition of two tensors
        /// </summary>
        public static Tensor Add(Tensor a, Tensor b)
        {
            ValidateBinaryOp(a, b);

            var result = Tensor.Empty(a.Shape, a.DataType);
            ExecuteElementwise(a, b, result, (va, vb) => va + vb);
            return result;
        }

        /// <summary>
        /// Element-wise subtraction of two tensors
        /// </summary>
        public static Tensor Subtract(Tensor a, Tensor b)
        {
            ValidateBinaryOp(a, b);

            var result = Tensor.Empty(a.Shape, a.DataType);
            ExecuteElementwise(a, b, result, (va, vb) => va - vb);
            return result;
        }

        /// <summary>
        /// Element-wise multiplication of two tensors
        /// </summary>
        public static Tensor Multiply(Tensor a, Tensor b)
        {
            ValidateBinaryOp(a, b);

            var result = Tensor.Empty(a.Shape, a.DataType);
            ExecuteElementwise(a, b, result, (va, vb) => va * vb);
            return result;
        }

        /// <summary>
        /// Element-wise division of two tensors
        /// </summary>
        public static Tensor Divide(Tensor a, Tensor b)
        {
            ValidateBinaryOp(a, b);

            var result = Tensor.Empty(a.Shape, a.DataType);
            ExecuteElementwise(a, b, result, (va, vb) => va / vb);
            return result;
        }

        #endregion

        #region Unary Operations

        /// <summary>
        /// Element-wise absolute value
        /// </summary>
        public static Tensor Abs(Tensor input)
        {
            var result = Tensor.Empty(input.Shape, input.DataType);
            ExecuteUnary(input, result, v => (float)Math.Abs(v));
            return result;
        }

        /// <summary>
        /// Element-wise square root
        /// </summary>
        public static Tensor Sqrt(Tensor input)
        {
            var result = Tensor.Empty(input.Shape, input.DataType);
            ExecuteUnary(input, result, v => (float)Math.Sqrt(v));
            return result;
        }

        /// <summary>
        /// Element-wise square
        /// </summary>
        public static Tensor Square(Tensor input)
        {
            var result = Tensor.Empty(input.Shape, input.DataType);
            ExecuteUnary(input, result, v => v * v);
            return result;
        }

        /// <summary>
        /// Element-wise natural logarithm
        /// </summary>
        public static Tensor Log(Tensor input)
        {
            var result = Tensor.Empty(input.Shape, input.DataType);
            ExecuteUnary(input, result, v => (float)Math.Log(v));
            return result;
        }

        /// <summary>
        /// Element-wise exponential
        /// </summary>
        public static Tensor Exp(Tensor input)
        {
            var result = Tensor.Empty(input.Shape, input.DataType);
            ExecuteUnary(input, result, v => (float)Math.Exp(v));
            return result;
        }

        /// <summary>
        /// ReLU activation: max(0, x)
        /// </summary>
        public static Tensor Relu(Tensor input)
        {
            var result = Tensor.Empty(input.Shape, input.DataType);
            ExecuteUnary(input, result, v => v > 0 ? v : 0);
            return result;
        }

        #endregion

        #region Reduction Operations

        /// <summary>
        /// Sum along an axis
        /// </summary>
        public static Tensor Sum(Tensor input, int axis = -1, bool keepDim = false)
        {
            long size = input.Size;
            int[] shape = input.Shape;

            if (axis < 0)
            {
                // Sum all elements
                float total = 0;
                for (long i = 0; i < size; i++)
                {
                    total += input.GetData<float>(GetIndices(i, shape));
                }
                return Tensor.FromArray(new[] { total }, new[] { 1 });
            }

            // Sum along specific axis
            var result = ReduceAlongAxis(input, axis, keepDim, (acc, val) => acc + val, 0.0f);
            return result;
        }

        /// <summary>
        /// Mean along an axis
        /// </summary>
        public static Tensor Mean(Tensor input, int axis = -1, bool keepDim = false)
        {
            long size = input.Size;
            int[] shape = input.Shape;

            if (axis < 0)
            {
                // Mean of all elements
                float total = 0;
                for (long i = 0; i < size; i++)
                {
                    total += input.GetData<float>(GetIndices(i, shape));
                }
                return Tensor.FromArray(new[] { total / size }, new[] { 1 });
            }

            // Mean along specific axis
            int axisSize = shape[axis];
            var sumResult = ReduceAlongAxis(input, axis, keepDim, (acc, val) => acc + val, 0.0f);

            // Divide by axis size
            var result = Tensor.Empty(sumResult.Shape, sumResult.DataType);
            long resultSize = sumResult.Size;
            for (long i = 0; i < resultSize; i++)
            {
                float val = sumResult.GetData<float>(GetIndices(i, sumResult.Shape));
                result.GetArray()[i] = val / axisSize;
            }
            return result;
        }

        /// <summary>
        /// Maximum along an axis
        /// </summary>
        public static Tensor Max(Tensor input, int axis = -1, bool keepDim = false)
        {
            int[] shape = input.Shape;

            if (axis < 0)
            {
                // Max of all elements
                float max = float.MinValue;
                long size = input.Size;
                for (long i = 0; i < size; i++)
                {
                    float val = input.GetData<float>(GetIndices(i, shape));
                    if (val > max) max = val;
                }
                return Tensor.FromArray(new[] { max }, new[] { 1 });
            }

            var result = ReduceAlongAxis(input, axis, keepDim,
                (acc, val) => val > acc ? val : acc, float.MinValue);
            return result;
        }

        /// <summary>
        /// Minimum along an axis
        /// </summary>
        public static Tensor Min(Tensor input, int axis = -1, bool keepDim = false)
        {
            int[] shape = input.Shape;

            if (axis < 0)
            {
                // Min of all elements
                float min = float.MaxValue;
                long size = input.Size;
                for (long i = 0; i < size; i++)
                {
                    float val = input.GetData<float>(GetIndices(i, shape));
                    if (val < min) min = val;
                }
                return Tensor.FromArray(new[] { min }, new[] { 1 });
            }

            var result = ReduceAlongAxis(input, axis, keepDim,
                (acc, val) => val < acc ? val : acc, float.MaxValue);
            return result;
        }

        #endregion

        #region Matrix Operations

        /// <summary>
        /// Matrix multiplication
        /// </summary>
        public static Tensor MatMul(Tensor a, Tensor b)
        {
            // For simplicity, assume 2D matrices
            if (a.Shape.Length != 2 || b.Shape.Length != 2)
                throw new ArgumentException("Only 2D matrices are supported");

            int m = a.Shape[0];
            int k = a.Shape[1];
            int n = b.Shape[1];

            if (k != b.Shape[0])
                throw new ArgumentException($"Matrix dimensions incompatible: ({m}x{k}) * ({b.Shape[0]}x{n})");

            var result = Tensor.Zeros(new[] { m, n }, a.DataType);
            float[] aData = a.GetArray();
            float[] bData = b.GetArray();
            float[] cData = result.GetArray();

            // Simple O(n^3) matrix multiplication
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    float sum = 0;
                    for (int p = 0; p < k; p++)
                    {
                        sum += aData[i * k + p] * bData[p * n + j];
                    }
                    cData[i * n + j] = sum;
                }
            }

            return result;
        }

        /// <summary>
        /// Transpose tensor (2D only for simplicity)
        /// </summary>
        public static Tensor Transpose(Tensor input)
        {
            if (input.Shape.Length != 2)
                throw new ArgumentException("Only 2D tensors are supported");

            int rows = input.Shape[0];
            int cols = input.Shape[1];
            var result = Tensor.Empty(new[] { cols, rows }, input.DataType);

            float[] inputData = input.GetArray();
            float[] outputData = result.GetArray();

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    outputData[j * rows + i] = inputData[i * cols + j];
                }
            }

            return result;
        }

        #endregion

        #region Shape Operations

        /// <summary>
        /// Reshape tensor to new shape
        /// </summary>
        public static Tensor Reshape(Tensor input, int[] newShape)
        {
            long newSize = newShape.Aggregate(1L, (a, b) => a * b);
            if (newSize != input.Size)
                throw new ArgumentException($"Cannot reshape tensor of size {input.Size} to shape {string.Join("x", newShape)}");

            // Create new tensor sharing the same data
            return new Tensor(newShape, input.DataType, input.DataPointer, false);
        }

        /// <summary>
        /// Flatten tensor to 1D
        /// </summary>
        public static Tensor Flatten(Tensor input, int startDim = 0, int endDim = -1)
        {
            if (endDim < 0)
                endDim = input.Shape.Length - 1;

            int flattenSize = 1;
            for (int i = startDim; i <= endDim; i++)
            {
                flattenSize *= input.Shape[i];
            }

            var newShape = new List<int>();
            for (int i = 0; i < startDim; i++)
            {
                newShape.Add(input.Shape[i]);
            }
            newShape.Add(flattenSize);
            for (int i = endDim + 1; i < input.Shape.Length; i++)
            {
                newShape.Add(input.Shape[i]);
            }

            return Reshape(input, newShape.ToArray());
        }

        #endregion

        #region Concatenation

        /// <summary>
        /// Concatenate tensors along an axis
        /// </summary>
        public static Tensor Concat(Tensor[] tensors, int axis)
        {
            if (tensors == null || tensors.Length == 0)
                throw new ArgumentException("No tensors to concatenate");

            int ndim = tensors[0].Shape.Length;
            foreach (var tensor in tensors)
            {
                if (tensor.Shape.Length != ndim)
                    throw new ArgumentException("All tensors must have the same number of dimensions");
            }

            var newShape = tensors[0].Shape.ToList();
            newShape[axis] = tensors.Sum(t => t.Shape[axis]);

            var result = Tensor.Empty(newShape.ToArray(), tensors[0].DataType);
            long totalSize = result.Size;

            // Simplified concatenation (only works for contiguous memory)
            float[] outputData = result.GetArray();
            int offset = 0;
            int stride = 1;
            for (int i = axis + 1; i < ndim; i++)
            {
                stride *= newShape[i];
            }

            foreach (var tensor in tensors)
            {
                float[] inputData = tensor.GetArray();
                int copySize = tensor.Shape[axis] * stride;
                Array.Copy(inputData, 0, outputData, offset, copySize);
                offset += copySize;
            }

            return result;
        }

        #endregion

        #region Memory Operations

        /// <summary>
        /// Copy tensor
        /// </summary>
        public static Tensor Copy(Tensor input)
        {
            var result = Tensor.Empty(input.Shape, input.DataType);
            float[] inputData = input.GetArray();
            float[] outputData = result.GetArray();
            Array.Copy(inputData, outputData, inputData.Length);
            return result;
        }

        /// <summary>
        /// Memory copy between tensors
        /// </summary>
        public static void MemCpy(Tensor dst, Tensor src)
        {
            if (dst.ByteCount != src.ByteCount)
                throw new ArgumentException("Destination and source tensors must have the same size");

            float[] srcData = src.GetArray();
            float[] dstData = dst.GetArray();
            Array.Copy(srcData, dstData, srcData.Length);
        }

        #endregion

        #region Type Conversions

        /// <summary>
        /// Cast tensor to different data type
        /// </summary>
        public static Tensor Cast(Tensor input, DataType targetDataType)
        {
            if (input.DataType == targetDataType)
                return Copy(input);

            var result = Tensor.Empty(input.Shape, targetDataType);
            float[] inputData = input.GetArray();
            float[] outputData = result.GetArray();

            // Simple float-to-float copy
            // Additional conversion logic needed for other types
            Array.Copy(inputData, outputData, inputData.Length);

            return result;
        }

        #endregion

        #region Private Helpers

        private static void ValidateBinaryOp(Tensor a, Tensor b)
        {
            if (a.Shape.Length != b.Shape.Length)
                throw new ArgumentException("Tensors must have the same number of dimensions");

            for (int i = 0; i < a.Shape.Length; i++)
            {
                if (a.Shape[i] != b.Shape[i])
                    throw new ArgumentException($"Tensors must have the same shape (dimension {i}: {a.Shape[i]} != {b.Shape[i]})");
            }

            if (a.DataType != b.DataType)
                throw new ArgumentException("Tensors must have the same data type");
        }

        private static void ExecuteElementwise(Tensor a, Tensor b, Tensor result, Func<float, float, float> op)
        {
            long size = a.Size;
            for (long i = 0; i < size; i++)
            {
                int[] indices = GetIndices(i, a.Shape);
                float va = a.GetData<float>(indices);
                float vb = b.GetData<float>(indices);
                result.GetArray()[i] = op(va, vb);
            }
        }

        private static void ExecuteUnary(Tensor input, Tensor result, Func<float, float> op)
        {
            long size = input.Size;
            for (long i = 0; i < size; i++)
            {
                int[] indices = GetIndices(i, input.Shape);
                float val = input.GetData<float>(indices);
                result.GetArray()[i] = op(val);
            }
        }

        private static Tensor ReduceAlongAxis(Tensor input, int axis, bool keepDim,
            Func<float, float, float> reduceFunc, float initialValue)
        {
            int[] inputShape = input.Shape;
            int[] outputShape = keepDim
                ? inputShape.Select((s, i) => i == axis ? 1 : s).ToArray()
                : inputShape.Where((s, i) => i != axis).ToArray();

            var result = Tensor.Empty(outputShape, input.DataType);
            int axisSize = inputShape[axis];
            long inputSize = input.Size;

            // Simplified reduction (works for 2D and simple cases)
            float[] inputData = input.GetArray();
            float[] outputData = result.GetArray();

            if (inputShape.Length == 2 && axis == 1)
            {
                // Sum along columns
                int rows = inputShape[0];
                int cols = inputShape[1];
                for (int i = 0; i < rows; i++)
                {
                    float acc = initialValue;
                    for (int j = 0; j < cols; j++)
                    {
                        acc = reduceFunc(acc, inputData[i * cols + j]);
                    }
                    outputData[i] = acc;
                }
            }
            else if (inputShape.Length == 2 && axis == 0)
            {
                // Sum along rows
                int rows = inputShape[0];
                int cols = inputShape[1];
                for (int j = 0; j < cols; j++)
                {
                    float acc = initialValue;
                    for (int i = 0; i < rows; i++)
                    {
                        acc = reduceFunc(acc, inputData[i * cols + j]);
                    }
                    outputData[j] = acc;
                }
            }

            return result;
        }

        private static int[] GetIndices(long linearIndex, int[] shape)
        {
            int[] indices = new int[shape.Length];
            for (int i = 0; i < shape.Length; i++)
            {
                indices[i] = (int)(linearIndex % shape[i]);
                linearIndex /= shape[i];
            }
            return indices;
        }

        #endregion

        #region Extension Methods

        // Helper to access tensor data as array (simplified)
        private static float[] GetArray(this Tensor tensor)
        {
            long size = tensor.Size;
            float[] array = new float[size];
            for (long i = 0; i < size; i++)
            {
                array[i] = Marshal.PtrToStructure<float>(IntPtr.Add(tensor.DataPointer, (int)i * 4));
            }
            return array;
        }

        #endregion
    }
}
