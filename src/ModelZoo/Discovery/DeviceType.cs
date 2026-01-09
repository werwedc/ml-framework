namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Types of devices for model deployment and inference.
    /// </summary>
    public enum DeviceType
    {
        /// <summary>
        /// Central Processing Unit.
        /// </summary>
        CPU,

        /// <summary>
        /// Graphics Processing Unit (CUDA).
        /// </summary>
        GPU,

        /// <summary>
        /// Tensor Processing Unit or other specialized AI accelerators.
        /// </summary>
        TPU,

        /// <summary>
        /// Neural Processing Unit.
        /// </summary>
        NPU,

        /// <summary>
        /// Edge device with limited resources.
        /// </summary>
        Edge
    }
}
