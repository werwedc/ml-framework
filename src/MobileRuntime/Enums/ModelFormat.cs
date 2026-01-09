namespace MobileRuntime
{
    /// <summary>
    /// Model format types supported by the mobile runtime.
    /// </summary>
    public enum ModelFormat
    {
        /// <summary>
        /// Custom mobile binary format.
        /// </summary>
        MobileBinary,

        /// <summary>
        /// Protocol Buffers format.
        /// </summary>
        Protobuf,

        /// <summary>
        /// ONNX format (future support).
        /// </summary>
        ONNX
    }
}
