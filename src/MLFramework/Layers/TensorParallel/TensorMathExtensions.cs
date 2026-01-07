using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using System;
using System.Linq;

namespace RitterFramework.Core.Tensor
{
    /// <summary>
    /// Additional tensor operations for tensor parallel layers.
    /// </summary>
    public static class TensorMathExtensions
    {
        /// <summary>
        /// Matrix multiplication: computes A @ B
        /// </summary>
        /// <param name="a">Left matrix [m, n]</param>
        /// <param name="b">Right matrix [n, p]</param>
        /// <param name="transposeA">Whether to transpose A before multiplication</param>
        /// <param name="transposeB">Whether to transpose B before multiplication</param>
        /// <returns>Result tensor [m, p]</returns>
        public static Tensor MatMul(Tensor a, Tensor b, bool transposeA = false, bool transposeB = false)
        {
            if (a.Dimensions != 2 || b.Dimensions != 2)
            {
                throw new ArgumentException("MatMul only supports 2D tensors");
            }

            // Get actual dimensions considering transposes
            int m = transposeA ? a.Shape[1] : a.Shape[0];
            int n1 = transposeA ? a.Shape[0] : a.Shape[1];
            int n2 = transposeB ? b.Shape[1] : b.Shape[0];
            int p = transposeB ? b.Shape[0] : b.Shape[1];

            if (n1 != n2)
            {
                throw new ArgumentException($"Inner dimensions must match: {n1} != {n2}");
            }

            int n = n1;
            var resultData = new float[m * p];

            if (!transposeA && !transposeB)
            {
                // Standard matmul: a [m, n] @ b [n, p]
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < p; j++)
                    {
                        float sum = 0;
                        for (int k = 0; k < n; k++)
                        {
                            sum += a[new[] { i, k }] * b[new[] { k, j }];
                        }
                        resultData[i * p + j] = sum;
                    }
                }
            }
            else if (transposeA && !transposeB)
            {
                // a^T [m, n] @ b [n, p] -> a is [n, m], use as [m, n]
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < p; j++)
                    {
                        float sum = 0;
                        for (int k = 0; k < n; k++)
                        {
                            sum += a[new[] { k, i }] * b[new[] { k, j }];
                        }
                        resultData[i * p + j] = sum;
                    }
                }
            }
            else if (!transposeA && transposeB)
            {
                // a [m, n] @ b^T [n, p] -> b is [p, n], use as [n, p]
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < p; j++)
                    {
                        float sum = 0;
                        for (int k = 0; k < n; k++)
                        {
                            sum += a[new[] { i, k }] * b[new[] { j, k }];
                        }
                        resultData[i * p + j] = sum;
                    }
                }
            }
            else // transposeA && transposeB
            {
                // a^T [m, n] @ b^T [n, p] -> a is [n, m], b is [p, n]
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < p; j++)
                    {
                        float sum = 0;
                        for (int k = 0; k < n; k++)
                        {
                            sum += a[new[] { k, i }] * b[new[] { j, k }];
                        }
                        resultData[i * p + j] = sum;
                    }
                }
            }

            return new Tensor(resultData, new[] { m, p }, a.RequiresGrad || b.RequiresGrad, a.Dtype);
        }

        /// <summary>
        /// Sum the tensor along specified dimensions.
        /// </summary>
        /// <param name="tensor">Input tensor</param>
        /// <param name="dimensions">Dimensions to sum over. If null, sums all elements.</param>
        /// <returns>Tensor with summed values</returns>
        public static Tensor Sum(this Tensor tensor, int[]? dimensions = null)
        {
            if (dimensions == null || dimensions.Length == 0)
            {
                // Sum all elements
                float totalSum = tensor.Data.Sum();
                return new Tensor(new[] { totalSum }, new int[] { 1 }, tensor.RequiresGrad, tensor.Dtype);
            }

            // Remove dimensions that will be summed over
            var resultShape = tensor.Shape
                .Select((dim, idx) => dimensions.Contains(idx) ? 1 : dim)
                .Where(dim => dim > 0)
                .ToArray();

            if (resultShape.Length == 0)
            {
                // All dimensions were summed over
                float totalSum = tensor.Data.Sum();
                return new Tensor(new[] { totalSum }, new int[] { 1 }, tensor.RequiresGrad, tensor.Dtype);
            }

            // For now, implement a simple sum over all dimensions
            // A full implementation would need to handle arbitrary dimension reduction
            float sum = tensor.Data.Sum();
            return new Tensor(new[] { sum }, resultShape, tensor.RequiresGrad, tensor.Dtype);
        }

        /// <summary>
        /// Computes the L2 norm (Frobenius norm for matrices) of the tensor.
        /// </summary>
        public static Tensor Norm(this Tensor tensor)
        {
            float sumSquares = tensor.Data.Sum(x => x * x);
            float norm = (float)Math.Sqrt(sumSquares);
            return new Tensor(new[] { norm }, new int[] { 1 }, false, tensor.Dtype);
        }

        /// <summary>
        /// Creates a tensor filled with random normal (Gaussian) values.
        /// </summary>
        /// <param name="shape">Shape of the tensor</param>
        /// <param name="mean">Mean of the distribution</param>
        /// <param name="std">Standard deviation of the distribution</param>
        /// <param name="seed">Optional random seed</param>
        /// <returns>Random tensor</returns>
        public static Tensor RandomNormal(int[] shape, double mean = 0.0, double std = 1.0, int? seed = null)
        {
            var random = seed.HasValue ? new Random(seed.Value) : new Random();
            int totalSize = 1;
            foreach (var dim in shape)
            {
                totalSize *= dim;
            }

            var data = new float[totalSize];

            // Box-Muller transform for generating normal distribution
            for (int i = 0; i < totalSize; i += 2)
            {
                double u1 = random.NextDouble();
                double u2 = random.NextDouble();

                // Avoid log(0)
                u1 = u1 == 0 ? 1e-10 : u1;

                double z0 = std * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2) + mean;
                data[i] = (float)z0;

                if (i + 1 < totalSize)
                {
                    double z1 = std * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2) + mean;
                    data[i + 1] = (float)z1;
                }
            }

            return new Tensor(data, shape, requiresGrad: false, DataType.Float32);
        }

        /// <summary>
        /// Creates a tensor filled with random normal (Gaussian) values with explicit dimensions.
        /// </summary>
        public static Tensor RandomNormal(int dim1, int dim2, double mean = 0.0, double std = 1.0, int? seed = null)
        {
            return RandomNormal(new[] { dim1, dim2 }, mean, std, seed);
        }

        /// <summary>
        /// Converts a single-element tensor to a scalar value.
        /// </summary>
        public static float ToScalar(this Tensor tensor)
        {
            if (tensor.Size != 1)
            {
                throw new InvalidOperationException("ToScalar can only be called on tensors with a single element");
            }
            return tensor.Data[0];
        }

        /// <summary>
        /// Slices the tensor along a specific dimension.
        /// Supports negative dimension indexing.
        /// </summary>
        /// <param name="tensor">Input tensor</param>
        /// <param name="dim">Dimension to slice (supports negative indexing)</param>
        /// <param name="start">Start index</param>
        /// <param name="end">End index (exclusive)</param>
        /// <returns>Sliced tensor</returns>
        public static Tensor Slice(this Tensor tensor, int dim, int start, int end)
        {
            // Handle negative dimension indexing
            if (dim < 0)
            {
                dim = tensor.Dimensions + dim;
            }

            if (dim < 0 || dim >= tensor.Dimensions)
            {
                throw new ArgumentOutOfRangeException(nameof(dim),
                    $"Dimension {dim} is out of bounds for tensor with {tensor.Dimensions} dimensions");
            }

            int dimSize = tensor.Shape[dim];

            // Handle negative indices
            if (start < 0) start = dimSize + start;
            if (end < 0) end = dimSize + end;

            if (start < 0 || start >= dimSize)
            {
                throw new ArgumentOutOfRangeException(nameof(start), $"Start index {start} is out of bounds for dimension {dim}");
            }

            if (end <= start || end > dimSize)
            {
                throw new ArgumentOutOfRangeException(nameof(end), $"End index {end} is invalid for dimension {dim}");
            }

            int sliceLength = end - start;
            int strideBefore = 1;
            for (int i = 0; i < dim; i++)
            {
                strideBefore *= tensor.Shape[i];
            }

            int strideAfter = 1;
            for (int i = dim + 1; i < tensor.Dimensions; i++)
            {
                strideAfter *= tensor.Shape[i];
            }

            var newShape = (int[])tensor.Shape.Clone();
            newShape[dim] = sliceLength;

            int resultSize = tensor.Size / dimSize * sliceLength;
            var resultData = new float[resultSize];

            int resultIdx = 0;
            for (int before = 0; before < strideBefore; before++)
            {
                int baseIdx = before * dimSize * strideAfter;
                for (int slice = 0; slice < sliceLength; slice++)
                {
                    int srcIdx = baseIdx + (start + slice) * strideAfter;
                    for (int after = 0; after < strideAfter; after++)
                    {
                        resultData[resultIdx++] = tensor.Data[srcIdx + after];
                    }
                }
            }

            return new Tensor(resultData, newShape, tensor.RequiresGrad, tensor.Dtype);
        }

        /// <summary>
        /// Fills the tensor with a scalar value in-place.
        /// </summary>
        public static void Fill(this Tensor tensor, float value)
        {
            for (int i = 0; i < tensor.Size; i++)
            {
                tensor.Data[i] = value;
            }
        }

        /// <summary>
        /// Gets the gradient tensor for a parameter.
        /// Returns null if no gradient exists.
        /// </summary>
        public static Tensor? GetGrad(this Tensor tensor)
        {
            return tensor.Gradient;
        }

        /// <summary>
        /// Sets the gradient tensor for a parameter.
        /// </summary>
        public static void SetGrad(this Tensor tensor, Tensor grad)
        {
            tensor.Gradient = grad;
        }
    }
}
