using System;
using System.Linq;
using RitterFramework.Core.Tensor;

namespace MLFramework.Functional.Tracing
{
    /// <summary>
    /// A tensor wrapper that records operations for tracing.
    /// </summary>
    public class TracedTensor
    {
        private readonly Tensor _underlying;  // Actual tensor (for eager execution)
        private readonly TraceNode _node;     // Corresponding trace node

        /// <summary>
        /// The underlying actual tensor.
        /// </summary>
        public Tensor Underlying => _underlying;

        /// <summary>
        /// The trace node representing this tensor's computation.
        /// </summary>
        public TraceNode Node => _node;

        /// <summary>
        /// Shape of the tensor.
        /// </summary>
        public TensorShape Shape => new TensorShape(_underlying.Shape);

        /// <summary>
        /// Type of the tensor.
        /// </summary>
        public TensorType Type => ConvertToTensorType(_underlying.Dtype);

        private TracedTensor(Tensor underlying, TraceNode node)
        {
            _underlying = underlying ?? throw new ArgumentNullException(nameof(underlying));
            _node = node ?? throw new ArgumentNullException(nameof(node));
        }

        /// <summary>
        /// Create a traced tensor from a regular tensor (input tensor).
        /// </summary>
        public static TracedTensor Create(Tensor tensor, string name = "input")
        {
            var node = new TraceNode(name, Array.Empty<TraceNode>(), new TensorShape(tensor.Shape), ConvertToTensorType(tensor.Dtype));
            return new TracedTensor(tensor, node);
        }

        /// <summary>
        /// Create a traced tensor from an operation.
        /// </summary>
        public static TracedTensor Create(Tensor result, string operation, TracedTensor[] inputs, Dictionary<string, object> attributes = null)
        {
            var inputNodes = inputs.Select(t => t.Node).ToArray();
            var node = new TraceNode(operation, inputNodes, new TensorShape(result.Shape), ConvertToTensorType(result.Dtype), attributes);
            return new TracedTensor(result, node);
        }

        // Implicit conversion for convenience
        public static implicit operator Tensor(TracedTensor traced) => traced._underlying;

        // Wrapper operations that record to trace
        public TracedTensor Add(TracedTensor other)
        {
            var result = _underlying + other._underlying;
            return Create(result, "add", new[] { this, other });
        }

        public TracedTensor Multiply(TracedTensor other)
        {
            // Tensor class doesn't have element-wise multiplication operator
            // We'll implement it manually
            var result = ElementwiseMultiply(_underlying, other._underlying);
            return Create(result, "multiply", new[] { this, other });
        }

        public TracedTensor MatMul(TracedTensor other)
        {
            var result = MatrixMultiply(_underlying, other._underlying);
            return Create(result, "matmul", new[] { this, other });
        }

        public TracedTensor ReLU()
        {
            var result = ApplyReLU(_underlying);
            return Create(result, "relu", new[] { this });
        }

        private static TensorType ConvertToTensorType(RitterFramework.Core.DataType dtype)
        {
            // Convert RitterFramework.Core.DataType to MLFramework.Functional.Tracing.TensorType
            return dtype switch
            {
                RitterFramework.Core.DataType.Float32 => TensorType.Float32,
                RitterFramework.Core.DataType.Float64 => TensorType.Float64,
                RitterFramework.Core.DataType.Int32 => TensorType.Int32,
                RitterFramework.Core.DataType.Int64 => TensorType.Int64,
                _ => TensorType.Float32
            };
        }

        private static Tensor MatrixMultiply(Tensor a, Tensor b)
        {
            // Simple matrix multiplication for 2D tensors
            if (a.Dimensions != 2 || b.Dimensions != 2)
                throw new ArgumentException("Matrix multiplication is only supported for 2D tensors");

            if (a.Shape[1] != b.Shape[0])
                throw new ArgumentException("Incompatible shapes for matrix multiplication");

            int rowsA = a.Shape[0];
            int colsA = a.Shape[1];
            int colsB = b.Shape[1];
            var resultData = new float[rowsA * colsB];

            for (int i = 0; i < rowsA; i++)
            {
                for (int j = 0; j < colsB; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < colsA; k++)
                    {
                        sum += a.Data[i * colsA + k] * b.Data[k * colsB + j];
                    }
                    resultData[i * colsB + j] = sum;
                }
            }

            return new Tensor(resultData, new[] { rowsA, colsB });
        }

        private static Tensor ApplyReLU(Tensor tensor)
        {
            var resultData = new float[tensor.Size];
            for (int i = 0; i < tensor.Size; i++)
            {
                resultData[i] = Math.Max(0, tensor.Data[i]);
            }
            return new Tensor(resultData, tensor.Shape);
        }

        private static Tensor ElementwiseMultiply(Tensor a, Tensor b)
        {
            if (!a.Shape.SequenceEqual(b.Shape))
                throw new ArgumentException("Shapes must match for element-wise multiplication");

            var resultData = new float[a.Size];
            for (int i = 0; i < a.Size; i++)
            {
                resultData[i] = a.Data[i] * b.Data[i];
            }
            return new Tensor(resultData, a.Shape);
        }
    }
}
