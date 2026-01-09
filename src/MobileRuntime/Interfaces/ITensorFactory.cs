namespace MobileRuntime
{
    /// <summary>
    /// Interface for tensor factory
    /// </summary>
    public interface ITensorFactory
    {
        /// <summary>
        /// Creates a tensor with the specified shape and data type
        /// </summary>
        ITensor CreateTensor(int[] shape, DataType dataType);

        /// <summary>
        /// Creates a tensor with the specified shape, data type, and data
        /// </summary>
        ITensor CreateTensor(int[] shape, DataType dataType, float[] data);
    }
}
