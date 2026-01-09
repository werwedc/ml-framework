namespace MobileRuntime
{
    /// <summary>
    /// Interface for mobile runtime
    /// </summary>
    public interface IMobileRuntime
    {
        /// <summary>
        /// Loads a model from a file path
        /// </summary>
        IModel LoadModel(string modelPath);

        /// <summary>
        /// Loads a model from a byte array
        /// </summary>
        IModel LoadModel(byte[] modelBytes);

        /// <summary>
        /// Gets runtime information
        /// </summary>
        RuntimeInfo GetRuntimeInfo();
    }
}
