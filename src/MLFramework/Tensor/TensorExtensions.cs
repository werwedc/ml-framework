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
    }
}
