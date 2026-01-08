namespace MLFramework.Communication.Backends;

using MLFramework.Communication;
using MLFramework.Distributed.Communication;

/// <summary>
/// Factory for NCCL backend
/// </summary>
public class NCCLBackendFactory : ICommunicationBackendFactory
{
    private static readonly Lazy<bool> _isAvailable = new Lazy<bool>(CheckNCCLAvailability);

    public int Priority => 100; // Highest priority for NVIDIA GPUs

    public bool IsAvailable()
    {
        return _isAvailable.Value;
    }

    public ICommunicationBackend Create(MLFramework.Distributed.Communication.CommunicationConfig config)
    {
        if (!IsAvailable())
        {
            throw new CommunicationException("NCCL is not available on this system");
        }

        // Get rank and world size from environment or MPI
        int rank = GetRank();
        int worldSize = GetWorldSize();

        return new NCCLBackend(rank, worldSize, config);
    }

    /// <summary>
    /// Check if NCCL is available
    /// </summary>
    private static bool CheckNCCLAvailability()
    {
        try
        {
            // Try to load NCCL library
            // This is a simplified check - in reality you'd need to verify:
            // 1. NCCL library exists
            // 2. CUDA is available
            // 3. At least one GPU is present

            // For now, we'll assume NCCL is not available unless explicitly detected
            // This prevents crashes on systems without CUDA

            // Try to check CUDA availability
            try
            {
                Native.NCCLNative.CreateCudaStream();
                return true;
            }
            catch
            {
                return false;
            }
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Get rank from environment or MPI
    /// </summary>
    private int GetRank()
    {
        // Check environment variables (e.g., RANK, OMPI_COMM_WORLD_RANK)
        string? rankStr = Environment.GetEnvironmentVariable("RANK") ??
                        Environment.GetEnvironmentVariable("OMPI_COMM_WORLD_RANK") ??
                        Environment.GetEnvironmentVariable("WORLD_RANK");

        if (!string.IsNullOrEmpty(rankStr) && int.TryParse(rankStr, out int rank))
        {
            return rank;
        }

        // Default to rank 0
        return 0;
    }

    /// <summary>
    /// Get world size from environment or MPI
    /// </summary>
    private int GetWorldSize()
    {
        // Check environment variables
        string? sizeStr = Environment.GetEnvironmentVariable("WORLD_SIZE") ??
                        Environment.GetEnvironmentVariable("OMPI_COMM_WORLD_SIZE");

        if (!string.IsNullOrEmpty(sizeStr) && int.TryParse(sizeStr, out int size))
        {
            return size;
        }

        // Default to single process
        return 1;
    }
}
