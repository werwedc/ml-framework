namespace MLFramework.Communication.Backends;

using MLFramework.Distributed.Communication;

/// <summary>
/// Factory for MPI backend
/// </summary>
public class MPIBackendFactory : ICommunicationBackendFactory
{
    private static readonly Lazy<bool> _isAvailable = new Lazy<bool>(CheckMPIAvailability);

    public int Priority => 50; // Medium priority (works everywhere)

    public bool IsAvailable()
    {
        return _isAvailable.Value;
    }

    public ICommunicationBackend Create(MLFramework.Distributed.Communication.CommunicationConfig config)
    {
        if (!IsAvailable())
        {
            throw new CommunicationException("MPI is not available on this system");
        }

        return new MPIBackend(config);
    }

    /// <summary>
    /// Check if MPI is available
    /// </summary>
    private static bool CheckMPIAvailability()
    {
        try
        {
            // Try to load MPI library
            // Check for MPI environment variables
            var rank = Environment.GetEnvironmentVariable("OMPI_COMM_WORLD_RANK");
            return rank != null || Native.MPINative.IsMPIAvailable();
        }
        catch
        {
            return false;
        }
    }
}
