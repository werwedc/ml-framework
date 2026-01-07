namespace MLFramework.Distributed.Mesh;

/// <summary>
/// Defines the different dimensions of parallelism in a device mesh.
/// </summary>
public enum ParallelismDimension
{
    /// <summary>
    /// Data Parallelism: split across batches
    /// </summary>
    Data = 0,

    /// <summary>
    /// Tensor Parallelism: split within layers
    /// </summary>
    Tensor = 1,

    /// <summary>
    /// Pipeline Parallelism: split across layers (future)
    /// </summary>
    Pipeline = 2,

    /// <summary>
    /// Mixture of Experts: split across experts (future)
    /// </summary>
    Expert = 3
}
