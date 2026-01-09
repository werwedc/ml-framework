namespace MLFramework.MobileRuntime
{
    /// <summary>
    /// Information about a model input.
    /// </summary>
    public class InputInfo
    {
        /// <summary>
        /// Name of the input tensor.
        /// </summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>
        /// Shape of the input tensor.
        /// </summary>
        public int[] Shape { get; set; } = Array.Empty<int>();

        /// <summary>
        /// Data type of the input tensor.
        /// </summary>
        public DataType DataType { get; set; }
    }
}
