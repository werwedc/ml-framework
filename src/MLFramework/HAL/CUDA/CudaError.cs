namespace MLFramework.HAL.CUDA;

/// <summary>
/// CUDA error codes
/// </summary>
public enum CudaError
{
    Success = 0,
    InvalidValue = 1,
    OutOfMemory = 2,
    NotInitialized = 3,
    Deinitialized = 4,
    ProfilerDisabled = 5,
    ProfilerNotInitialized = 6,
    ProfilerAlreadyStarted = 7,
    ProfilerAlreadyStopped = 8,
    NoDevice = 38,
    InvalidDevice = 101,
    InvalidDeviceFunction = 98,
    InvalidConfiguration = 9,
    InvalidMemcpyDirection = 11,
    InvalidTexture = 21,
    InvalidTextureBinding = 22,
    InvalidChannelDescriptor = 23,
    InvalidFilterSetting = 24,
    NotReady = 600,
    LaunchFailure = 700,
    LaunchOutOfResources = 701,
    LaunchTimeout = 702,
    LaunchIncompatibleTexturing = 703,
    PeerAccessAlreadyEnabled = 704,
    PeerAccessNotEnabled = 705,
    InvalidPtx = 706,
    InvalidGraphicsContext = 707,
    Unknown = 999
}

/// <summary>
/// CUDA exception
/// </summary>
public class CudaException : Exception
{
    public CudaError Error { get; }

    public CudaException(CudaError error)
        : base($"CUDA error: {error}")
    {
        Error = error;
    }

    public CudaException(CudaError error, string message)
        : base($"CUDA error: {error} - {message}")
    {
        Error = error;
    }

    /// <summary>
    /// Check CUDA error and throw exception if not success
    /// </summary>
    public static void CheckError(CudaError error)
    {
        if (error != CudaError.Success)
        {
            throw new CudaException(error);
        }
    }

    /// <summary>
    /// Check CUDA error with custom message
    /// </summary>
    public static void CheckError(CudaError error, string message)
    {
        if (error != CudaError.Success)
        {
            throw new CudaException(error, message);
        }
    }
}
