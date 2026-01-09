using System;
using System.Collections.Generic;

namespace MobileRuntime.Backends.Metal
{
    /// <summary>
    /// Descriptor for an operator
    /// </summary>
    public class OperatorDescriptor
    {
        /// <summary>
        /// Gets or sets the operator type
        /// </summary>
        public OperatorType Type { get; set; }

        /// <summary>
        /// Gets or sets the input tensor IDs
        /// </summary>
        public uint[] InputIds { get; set; }

        /// <summary>
        /// Gets or sets the output tensor IDs
        /// </summary>
        public uint[] OutputIds { get; set; }

        /// <summary>
        /// Gets or sets the operator parameters
        /// </summary>
        public Dictionary<string, object> Parameters { get; set; }

        /// <summary>
        /// Gets or sets the operator name (for debugging)
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Creates a new operator descriptor
        /// </summary>
        public OperatorDescriptor()
        {
            InputIds = Array.Empty<uint>();
            OutputIds = Array.Empty<uint>();
            Parameters = new Dictionary<string, object>();
        }

        /// <summary>
        /// Creates a new operator descriptor with the specified type
        /// </summary>
        public OperatorDescriptor(OperatorType type, string name = "") : this()
        {
            Type = type;
            Name = string.IsNullOrEmpty(name) ? type.ToString() : name;
        }
    }
}
