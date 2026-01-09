using System.Collections.Generic;

namespace MobileRuntime.Serialization.Models
{
    /// <summary>
    /// Descriptor for an operator in the computation graph
    /// </summary>
    public class OperatorDescriptor
    {
        public OperatorType Type { get; set; }
        public uint[] InputTensorIds { get; set; }
        public uint[] OutputTensorIds { get; set; }
        public Dictionary<string, object> Parameters { get; set; }

        public OperatorDescriptor()
        {
            InputTensorIds = new uint[0];
            OutputTensorIds = new uint[0];
            Parameters = new Dictionary<string, object>();
        }

        public override string ToString()
        {
            return $"{Type}({InputTensorIds.Length} inputs, {OutputTensorIds.Length} outputs)";
        }
    }

    /// <summary>
    /// Supported operator types in the mobile runtime
    /// </summary>
    public enum OperatorType : ushort
    {
        Conv2D,
        DepthwiseConv2D,
        FullyConnected,
        MaxPool2D,
        AvgPool2D,
        BatchNorm,
        Relu,
        Sigmoid,
        Tanh,
        Softmax,
        Add,
        Subtract,
        Multiply,
        Divide,
        Concat,
        Reshape,
        Flatten,
        Transpose,
        MatMul,
        GlobalAveragePool,
        LeakyRelu,
        Pad,
        StridedSlice,
        Slice,
        ExpandDims,
        Squeeze,
        Pow,
        Sqrt,
        Log,
        Exp,
        Abs,
        Mean,
        Sum,
        ReduceMax,
        ReduceMin,
        Cast,
        Identity,
        // Add more as needed
    }
}
