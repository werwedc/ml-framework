namespace MLFramework.Distributed.Communication;

/// <summary>
/// Exception thrown when a communication operation times out.
/// </summary>
public class CommunicationTimeoutException : CommunicationException
{
    /// <summary>
    /// Gets the timeout duration.
    /// </summary>
    public TimeSpan TimeoutDuration { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="CommunicationTimeoutException"/> class.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="timeoutDuration">The timeout duration.</param>
    public CommunicationTimeoutException(string message, TimeSpan timeoutDuration)
        : base(message)
    {
        TimeoutDuration = timeoutDuration;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CommunicationTimeoutException"/> class
    /// with additional information.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="timeoutDuration">The timeout duration.</param>
    /// <param name="rank">The rank of the process where the exception occurred.</param>
    /// <param name="backendName">The name of the backend.</param>
    public CommunicationTimeoutException(string message, TimeSpan timeoutDuration, int rank, string backendName)
        : base(message, rank, backendName)
    {
        TimeoutDuration = timeoutDuration;
    }
}
