namespace MLFramework.Communication.Backends;

using MLFramework.Distributed.Communication;

/// <summary>
/// MPI-specific configuration
/// </summary>
public class MPIConfig
{
    /// <summary>
    /// Use CUDA-aware MPI for GPU communication
    /// </summary>
    public bool UseCudaAwareMPI { get; set; } = true;

    /// <summary>
    /// Number of threads for MPI initialization
    /// </summary>
    public int ThreadLevel { get; set; } = 1; // MPI_THREAD_SINGLE

    /// <summary>
    /// Enable MPI profiling
    /// </summary>
    public bool EnableProfiling { get; set; } = false;

    /// <summary>
    /// MPI buffer size for non-blocking operations
    /// </summary>
    public int BufferSize { get; set; } = 65536; // 64KB

    /// <summary>
    /// Enable collective algorithm tuning
    /// </summary>
    public bool EnableTuning { get; set; } = false;

    /// <summary>
    /// Apply MPI configuration
    /// </summary>
    public void Apply()
    {
        // Set MPI environment variables
        if (EnableTuning)
        {
            Environment.SetEnvironmentVariable("I_MPI_ADJUST_ALLREDUCE", "2");
        }

        if (EnableProfiling)
        {
            Environment.SetEnvironmentVariable("I_MPI_STATS", "1");
        }
    }
}
