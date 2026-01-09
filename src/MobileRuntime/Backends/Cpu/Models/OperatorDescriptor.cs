using MLFramework.MobileRuntime.Backends.Cpu.Models;

namespace MLFramework.MobileRuntime.Backends.Cpu.Interfaces
{
    using System.Collections.Generic;

    /// <summary>
    /// Describes an operator to be executed.
    /// </summary>
    public class OperatorDescriptor
    {
        /// <summary>
        /// Type of the operator.
        /// </summary>
        public OperatorType Type { get; set; }

        /// <summary>
        /// Unique identifier for this operator instance.
        /// </summary>
        public uint Id { get; set; }

        /// <summary>
        /// Optional parameters specific to the operator type.
        /// </summary>
        public Dictionary<string, object> Parameters { get; set; } = new Dictionary<string, object>();
    }
}
