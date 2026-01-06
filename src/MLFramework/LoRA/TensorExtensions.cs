using RitterFramework.Core.Tensor;

namespace MLFramework.LoRA
{
    /// <summary>
    /// Extension methods for Tensor operations needed by LoRA
    /// </summary>
    public static class TensorExtensions
    {
        /// <summary>
        /// Transposes a 2D tensor
        /// </summary>
        public static Tensor Transpose(this Tensor tensor)
        {
            if (tensor.Shape.Length != 2)
                throw new ArgumentException("Only 2D tensors can be transposed");

            int rows = tensor.Shape[0];
            int cols = tensor.Shape[1];

            var data = new float[tensor.Size];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    data[j * rows + i] = tensor[new[] { i, j }];
                }
            }

            return new Tensor(data, new[] { cols, rows }, tensor.RequiresGrad);
        }

        /// <summary>
        /// Clones a tensor (creates a copy)
        /// </summary>
        public static Tensor Clone(this Tensor tensor)
        {
            var data = new float[tensor.Size];
            for (int i = 0; i < data.Length; i++)
            {
                // Access tensor data - simplified implementation
                // In a real implementation, we'd access the internal array directly
                int[] indices = GetIndices(tensor.Shape, i);
                data[i] = tensor[indices];
            }
            return new Tensor(data, tensor.Shape, tensor.RequiresGrad);
        }

        /// <summary>
        /// Copies data from another tensor
        /// </summary>
        public static void CopyFrom(this Tensor tensor, Tensor source)
        {
            if (!tensor.Shape.SequenceEqual(source.Shape))
                throw new ArgumentException("Tensor shapes must match for copy");

            for (int i = 0; i < tensor.Size; i++)
            {
                int[] indices = GetIndices(tensor.Shape, i);
                tensor[indices] = source[indices];
            }
        }

        /// <summary>
        /// Checks if all elements are greater than a threshold
        /// </summary>
        public static Tensor GreaterThan(this Tensor tensor, float threshold)
        {
            var data = new float[tensor.Size];
            for (int i = 0; i < data.Length; i++)
            {
                int[] indices = GetIndices(tensor.Shape, i);
                data[i] = tensor[indices] > threshold ? 1.0f : 0.0f;
            }
            return new Tensor(data, tensor.Shape);
        }

        /// <summary>
        /// Reshapes a tensor
        /// </summary>
        public static Tensor Reshape(this Tensor tensor, int[] newShape)
        {
            int totalElements = tensor.Size;
            int newTotalElements = newShape.Aggregate(1, (x, y) => x * y);

            if (totalElements != newTotalElements)
                throw new ArgumentException("New shape must have same number of elements");

            var data = new float[tensor.Size];
            for (int i = 0; i < data.Length; i++)
            {
                int[] indices = GetIndices(tensor.Shape, i);
                data[i] = tensor[indices];
            }

            return new Tensor(data, newShape, tensor.RequiresGrad);
        }

        /// <summary>
        /// Element-wise multiplication of two tensors
        /// </summary>
        public static Tensor Mul(this Tensor a, Tensor b)
        {
            if (!a.Shape.SequenceEqual(b.Shape))
                throw new ArgumentException("Tensor shapes must match for element-wise multiplication");

            var data = new float[a.Size];
            for (int i = 0; i < data.Length; i++)
            {
                int[] indices = GetIndices(a.Shape, i);
                data[i] = a[indices] * b[indices];
            }

            return new Tensor(data, a.Shape, a.RequiresGrad || b.RequiresGrad);
        }

        /// <summary>
        /// Converts flat index to multi-dimensional indices
        /// </summary>
        private static int[] GetIndices(int[] shape, int flatIndex)
        {
            var indices = new int[shape.Length];
            var strides = new int[shape.Length];
            int stride = 1;

            for (int i = shape.Length - 1; i >= 0; i--)
            {
                strides[i] = stride;
                stride *= shape[i];
            }

            int remaining = flatIndex;
            for (int i = 0; i < shape.Length; i++)
            {
                indices[i] = remaining / strides[i];
                remaining %= strides[i];
            }

            return indices;
        }
    }
}
