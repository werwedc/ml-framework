using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;

namespace MLFramework.Functional.Tracing
{
    /// <summary>
    /// Manages the current trace context.
    /// </summary>
    public class TraceContext : IDisposable
    {
        private static readonly ThreadLocal<TraceContext> _current = new ThreadLocal<TraceContext>();

        public static TraceContext Current => _current.Value;

        public bool IsActive => _current.Value == this;
        public List<TraceNode> Nodes { get; } = new List<TraceNode>();
        public Dictionary<string, TraceNode> NamedOutputs { get; } = new Dictionary<string, TraceNode>();

        public TraceContext()
        {
            _current.Value = this;
        }

        public void RecordNode(TraceNode node)
        {
            if (!IsActive)
                throw new InvalidOperationException("Trace context is not active");

            Nodes.Add(node);
        }

        public void RegisterOutput(string name, TraceNode node)
        {
            if (!IsActive)
                throw new InvalidOperationException("Trace context is not active");

            NamedOutputs[name] = node;
        }

        public void Dispose()
        {
            if (IsActive)
                _current.Value = null;
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine("Trace:");
            foreach (var node in Nodes)
            {
                sb.AppendLine($"  {node}");
            }
            return sb.ToString();
        }
    }
}
