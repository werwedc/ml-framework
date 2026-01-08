namespace MLFramework.Communication.Backends;

using MLFramework.Communication;
using MLFramework.Distributed.Communication;

/// <summary>
/// Factory for RCCL backend
/// </summary>
public class RCCLBackendFactory : ICommunicationBackendFactory
{
    private static readonly Lazy<bool> _isAvailable = new Lazy<bool>(CheckRCCLAvailability);

    public int Priority => 90; // High priority for AMD GPUs

    public bool IsAvailable()
    {
        return _isAvailable.Value;
    }

    public ICommunicationBackend Create(MLFramework.Distributed.Communication.CommunicationConfig config)
    {
        if (!IsAvailable())
        {
            throw new CommunicationException("RCCL is not available on this system");
        }

        // Get rank and world size from environment or MPI
        int rank = GetRank();
        int worldSize = GetWorldSize();

        return new RCCLBackend(rank, worldSize, config);
    }

    /// <summary>
    /// Check if RCCL is available
    /// </summary>
    private static bool CheckRCCLAvailability()
    {
        try
        {
            // Try to load RCCL library
            // Check for ROCm availability
            // In a real implementation, this would attempt to load the RCCL library
            // and verify it's functional
            return false; // Placeholder - RCCL not available by default
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
        string rankStr = Environment.GetEnvironmentVariable("RANK") ??
                       Environment.GetEnvironmentVariable("OMPI_COMM_WORLD_RANK") ??
                       Environment.GetEnvironmentVariable("WORLD_RANK");

        if (int.TryParse(rankStr, out int rank))
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
        string sizeStr = Environment.GetEnvironmentVariable("WORLD_SIZE") ??
                        Environment.GetEnvironmentVariable("OMPI_COMM_WORLD_SIZE");

        if (int.TryParse(sizeStr, out int size))
        {
            return size;
        }

        // Default to single process
        return 1;
    }
}
