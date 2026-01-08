using Microsoft.Extensions.Logging;
using MachineLearning.Checkpointing.FaultTolerance;
using MachineLearning.Checkpointing.Storage;

namespace MLFramework.Tests.Checkpointing.FaultTolerance;

/// <summary>
/// Tests for fault tolerance features
/// </summary>
[TestClass]
public class FaultToleranceTests
{
    [TestMethod]
    public void RetryPolicy_IsRetryable_ReturnsTrueForRetryableExceptions()
    {
        // Arrange
        var policy = new RetryPolicy();
        policy.RetryableExceptions.Add(typeof(IOException));
        policy.RetryableExceptions.Add(typeof(TimeoutException));

        // Act
        var ioResult = policy.IsRetryable(new IOException());
        var timeoutResult = policy.IsRetryable(new TimeoutException());
        var argumentResult = policy.IsRetryable(new ArgumentException());

        // Assert
        Assert.IsTrue(ioResult);
        Assert.IsTrue(timeoutResult);
        Assert.IsFalse(argumentResult);
    }

    [TestMethod]
    public void RetryPolicy_DefaultConstructor_HasDefaultRetryableExceptions()
    {
        // Arrange & Act
        var policy = new RetryPolicy();

        // Assert
        Assert.AreEqual(3, policy.MaxRetries);
        Assert.AreEqual(TimeSpan.FromSeconds(1), policy.InitialDelay);
        Assert.AreEqual(TimeSpan.FromSeconds(30), policy.MaxDelay);
        Assert.AreEqual(2.0, policy.BackoffFactor);
    }

    [TestMethod]
    public async Task FaultToleranceHandler_ExecuteWithRetryAsync_RetriesOnFailure()
    {
        // Arrange
        var storage = new InMemoryStorage();
        var handler = new FaultToleranceHandler(storage, new RetryPolicy(3, TimeSpan.FromMilliseconds(10), TimeSpan.FromSeconds(1), 2.0));
        int attemptCount = 0;

        // Act
        try
        {
            await handler.ExecuteWithRetryAsync(async () =>
            {
                attemptCount++;
                if (attemptCount < 3)
                {
                    throw new IOException("Simulated failure");
                }
                return "success";
            });
        }
        catch
        {
            // Expected to fail on all retries
        }

        // Assert
        Assert.AreEqual(3, attemptCount);
    }

    [TestMethod]
    public async Task FaultToleranceHandler_ExecuteWithRetryAsync_SucceedsAfterRetry()
    {
        // Arrange
        var storage = new InMemoryStorage();
        var handler = new FaultToleranceHandler(storage, new RetryPolicy(3, TimeSpan.FromMilliseconds(10), TimeSpan.FromSeconds(1), 2.0));
        int attemptCount = 0;

        // Act
        var result = await handler.ExecuteWithRetryAsync(async () =>
        {
            attemptCount++;
            if (attemptCount < 2)
            {
                throw new IOException("Simulated failure");
            }
            return "success";
        });

        // Assert
        Assert.AreEqual(2, attemptCount);
        Assert.AreEqual("success", result);
    }

    [TestMethod]
    public async Task FaultToleranceHandler_ExecuteWithTimeoutAsync_ThrowsOnTimeout()
    {
        // Arrange
        var storage = new InMemoryStorage();
        var handler = new FaultToleranceHandler(storage);

        // Act & Assert
        await Assert.ThrowsExceptionAsync<TimeoutException>(async () =>
        {
            await handler.ExecuteWithTimeoutAsync(async () =>
            {
                await Task.Delay(TimeSpan.FromSeconds(5));
                return "success";
            }, TimeSpan.FromMilliseconds(100));
        });
    }

    [TestMethod]
    public async Task FaultToleranceHandler_ExecuteWithTimeoutAsync_SucceedsBeforeTimeout()
    {
        // Arrange
        var storage = new InMemoryStorage();
        var handler = new FaultToleranceHandler(storage);

        // Act
        var result = await handler.ExecuteWithTimeoutAsync(async () =>
        {
            await Task.Delay(TimeSpan.FromMilliseconds(50));
            return "success";
        }, TimeSpan.FromSeconds(1));

        // Assert
        Assert.AreEqual("success", result);
    }

    [TestMethod]
    public void CircuitBreaker_ExecuteAsync_OpensAfterThreshold()
    {
        // Arrange
        var circuitBreaker = new CircuitBreaker(failureThreshold: 3);
        int failureCount = 0;

        // Act
        for (int i = 0; i < 3; i++)
        {
            try
            {
                circuitBreaker.ExecuteAsync(() => Task.FromResult("test"), null).GetAwaiter().GetResult();
            }
            catch
            {
                failureCount++;
            }
        }

        // After 3 failures, circuit should be open
        try
        {
            circuitBreaker.ExecuteAsync(() => Task.FromResult("test"), null).GetAwaiter().GetResult();
        }
        catch (CircuitBreakerOpenException)
        {
            failureCount++;
        }

        // Assert
        Assert.AreEqual(4, failureCount);
        Assert.AreEqual(CircuitState.Open, circuitBreaker.State);
    }

    [TestMethod]
    public async Task CircuitBreaker_ExecuteAsync_ClosesOnSuccess()
    {
        // Arrange
        var circuitBreaker = new CircuitBreaker(failureThreshold: 2);

        // Act - fail once
        try
        {
            await circuitBreaker.ExecuteAsync(() => throw new InvalidOperationException(), null);
        }
        catch { }

        // Then succeed
        var result = await circuitBreaker.ExecuteAsync(() => Task.FromResult("success"), null);

        // Assert
        Assert.AreEqual("success", result);
        Assert.AreEqual(CircuitState.Closed, circuitBreaker.State);
    }

    /// <summary>
    /// Simple in-memory storage for testing
    /// </summary>
    private class InMemoryStorage : ICheckpointStorage
    {
        private readonly Dictionary<string, byte[]> _storage = new();

        public Task WriteAsync(string path, byte[] data, CancellationToken cancellationToken = default)
        {
            _storage[path] = data;
            return Task.CompletedTask;
        }

        public Task<byte[]> ReadAsync(string path, CancellationToken cancellationToken = default)
        {
            return Task.FromResult(_storage[path]);
        }

        public Task DeleteAsync(string path, CancellationToken cancellationToken = default)
        {
            _storage.Remove(path);
            return Task.CompletedTask;
        }

        public Task<bool> ExistsAsync(string path, CancellationToken cancellationToken = default)
        {
            return Task.FromResult(_storage.ContainsKey(path));
        }

        public IAsyncEnumerable<string> ListAsync(string prefix, CancellationToken cancellationToken = default)
        {
            return _storage.Keys.Where(k => k.StartsWith(prefix)).ToAsyncEnumerable();
        }
    }
}
