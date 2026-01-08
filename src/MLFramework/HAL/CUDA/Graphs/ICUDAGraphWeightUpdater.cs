namespace MLFramework.HAL.CUDA;

/// <summary>
/// Interface for weight updates within graphs
/// </summary>
public interface ICUDAGraphWeightUpdater
{
    /// <summary>
    /// Updates weights in the graph without re-capturing
    /// </summary>
    /// <param name="weightBuffer">Buffer containing updated weights</param>
    /// <param name="offset">Offset in the weight buffer</param>
    /// <param name="size">Size of the weight update</param>
    void UpdateWeights(IntPtr weightBuffer, long offset, long size);

    /// <summary>
    /// Gets the number of weight parameters in the graph
    /// </summary>
    int WeightParameterCount { get; }
}
