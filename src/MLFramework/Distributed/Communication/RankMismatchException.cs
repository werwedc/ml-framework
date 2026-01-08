namespace MLFramework.Distributed.Communication;

/// <summary>
/// Exception thrown when there is a mismatch between expected and actual ranks.
/// </summary>
public class RankMismatchException : CommunicationException
{
    /// <summary>
    /// Gets the expected rank.
    /// </summary>
    public int ExpectedRank { get; }

    /// <summary>
    /// Gets the actual rank.
    /// </summary>
    public int ActualRank { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="RankMismatchException"/> class.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="expectedRank">The expected rank.</param>
    /// <param name="actualRank">The actual rank.</param>
    public RankMismatchException(string message, int expectedRank, int actualRank)
        : base(message)
    {
        ExpectedRank = expectedRank;
        ActualRank = actualRank;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="RankMismatchException"/> class
    /// with additional information.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="expectedRank">The expected rank.</param>
    /// <param name="actualRank">The actual rank.</param>
    /// <param name="rank">The rank of the process where the exception occurred.</param>
    /// <param name="backendName">The name of the backend.</param>
    public RankMismatchException(string message, int expectedRank, int actualRank, int rank, string backendName)
        : base(message, rank, backendName)
    {
        ExpectedRank = expectedRank;
        ActualRank = actualRank;
    }
}
