namespace MachineLearning.Checkpointing.Tests;

using Moq;
using Xunit;

/// <summary>
/// Tests for FaultToleranceHandler
/// </summary>
public class FaultToleranceHandlerTests
{
    private Mock<ICheckpointStorage> _mockStorage = null!;
    private FaultToleranceHandler _handler = null!;

    public FaultToleranceHandlerTests()
    {
        _mockStorage = new Mock<ICheckpointStorage>();
        _handler = new FaultToleranceHandler(_mockStorage.Object);
    }

    [Fact]
    public async Task ExecuteWithRetryAsync_WithSuccess_ReturnsResult()
    {
        // Arrange
        var expected = "success";
        var operation = Task.FromResult(expected);

        // Act
        var result = await _handler.ExecuteWithRetryAsync(() => operation);

        // Assert
        Assert.Equal(expected, result);
    }

    [Fact]
    public async Task ExecuteWithRetryAsync_WithRetryableException_Retries()
    {
        // Arrange
        var callCount = 0;
        Task<string> operation()
        {
            callCount++;
            if (callCount < 3)
            {
                throw new IOException("Simulated failure");
            }
            return Task.FromResult("success");
        }

        // Act
        var result = await _handler.ExecuteWithRetryAsync(operation);

        // Assert
        Assert.Equal("success", result);
        Assert.Equal(3, callCount);
    }

    [Fact]
    public async Task ExecuteWithTimeoutAsync_WithTimeout_ThrowsTimeoutException()
    {
        // Arrange
        var operation = Task.Delay(TimeSpan.FromMinutes(10));

        // Act & Assert
        await Assert.ThrowsAsync<TimeoutException>(
            () => _handler.ExecuteWithTimeoutAsync(() => operation, TimeSpan.FromMilliseconds(100)));
    }
}
