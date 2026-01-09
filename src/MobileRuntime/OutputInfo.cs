using System;

namespace MobileRuntime
{
    /// <summary>
    /// Information about a model output.
    /// </summary>
    public class OutputInfo
    {
        /// <summary>
        /// Name of the output tensor.
        /// </summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>
        /// Shape of the output tensor.
        /// </summary>
        public int[] Shape { get; set; } = Array.Empty<int>();

        /// <summary>
        /// Data type of the output tensor.
        /// </summary>
        public DataType DataType { get; set; }
    }
}
