using System;
using System.Collections.Generic;

namespace MLFramework.Functional.Tracing
{
    /// <summary>
    /// Represents a single operation in a computational trace.
    /// </summary>
    public class TraceNode
    {
        public Guid Id { get; } = Guid.NewGuid();
        public string OperationName { get; }
        public TraceNode[] Inputs { get; }
        public TensorShape OutputShape { get; }
        public TensorType OutputType { get; }
        public Dictionary<string, object> Attributes { get; }

        public TraceNode(
            string operationName,
            TraceNode[] inputs,
            TensorShape outputShape,
            TensorType outputType,
            Dictionary<string, object> attributes = null)
        {
            OperationName = operationName;
            Inputs = inputs ?? Array.Empty<TraceNode>();
            OutputShape = outputShape;
            OutputType = outputType;
            Attributes = attributes ?? new Dictionary<string, object>();
        }

        public override string ToString()
        {
            return $"{OperationName}({OutputShape})";
        }
    }
}
