using MLFramework.MobileRuntime;

namespace MLFramework.MobileRuntime.Backends.Cpu.Interfaces
{
    /// <summary>
    /// Interface for creating tensors.
    /// </summary>
    public interface ITensorFactory
    {
        /// <summary>
        /// Creates a new tensor with the specified shape and data type.
        /// </summary>
        /// <param name="shape">Shape of the tensor.</param>
        /// <param name="dataType">Data type of the tensor.</param>
        /// <returns>A new tensor instance.</returns>
        ITensor Create(int[] shape, DataType dataType);

        /// <summary>
        /// Creates a tensor from existing data.
        /// </summary>
        /// <typeparam name="T">Type of the data.</typeparam>
        /// <param name="data">Data array.</param>
        /// <param name="shape">Shape of the tensor.</param>
        /// <returns>A new tensor instance.</returns>
        ITensor FromArray<T>(T[] data, int[] shape);
    }
}
