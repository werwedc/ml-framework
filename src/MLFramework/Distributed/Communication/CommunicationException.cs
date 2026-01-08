namespace MLFramework.Distributed.Communication;

/// <summary>
/// Exception thrown when a communication operation fails.
/// </summary>
public class CommunicationException : Exception
{
    /// <summary>
    /// Gets the rank of the process where the exception occurred, if available.
    /// </summary>
    public int? Rank { get; }

    /// <summary>
    /// Gets the name of the backend where the exception occurred, if available.
    /// </summary>
    public string? BackendName { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="CommunicationException"/> class.
    /// </summary>
    /// <param name="message">The error message.</param>
    public CommunicationException(string message)
        : base(message)
    {
        Rank = null;
        BackendName = null;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CommunicationException"/> class
    /// with rank and backend information.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="rank">The rank of the process where the exception occurred.</param>
    /// <param name="backendName">The name of the backend.</param>
    public CommunicationException(string message, int rank, string backendName)
        : base(message)
    {
        Rank = rank;
        BackendName = backendName;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CommunicationException"/> class
    /// with a reference to the inner exception.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="innerException">The inner exception.</param>
    public CommunicationException(string message, Exception innerException)
        : base(message, innerException)
    {
        Rank = null;
        BackendName = null;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CommunicationException"/> class
    /// with a reference to the inner exception and additional information.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="innerException">The inner exception.</param>
    /// <param name="rank">The rank of the process where the exception occurred.</param>
    /// <param name="backendName">The name of the backend.</param>
    public CommunicationException(string message, Exception innerException, int rank, string backendName)
        : base(message, innerException)
    {
        Rank = rank;
        BackendName = backendName;
    }
}
