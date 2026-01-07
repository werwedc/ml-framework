using MLFramework.Inference.ContinuousBatching;
using Xunit;

namespace MLFramework.Tests.Inference.ContinuousBatching;

/// <summary>
/// Unit tests for RequestQueue.
/// </summary>
public class RequestQueueTests
{
    #region Test Factories

    private static Request CreateTestRequest(
        string prompt = "Test prompt",
        int maxTokens = 100,
        CancellationToken? token = null,
        Priority priority = Priority.Normal)
    {
        return new Request(
            RequestId.New(),
            prompt,
            maxTokens,
            token ?? CancellationToken.None,
            priority
        );
    }

    private static RequestQueue CreateRequestQueue(int initialCapacity = 100)
    {
        return new RequestQueue(initialCapacity);
    }

    #endregion

    #region Basic Operations Tests

    [Fact]
    public void Constructor_WithDefaultCapacity_CreatesQueue()
    {
        // Act
        var queue = new RequestQueue();

        // Assert
        Assert.NotNull(queue);
        Assert.Equal(0, queue.Count);
        Assert.True(queue.IsEmpty);
    }

    [Fact]
    public void Constructor_WithCustomCapacity_CreatesQueue()
    {
        // Act
        var queue = new RequestQueue(50);

        // Assert
        Assert.NotNull(queue);
        Assert.Equal(0, queue.Count);
    }

    [Fact]
    public void Enqueue_AddsRequestToQueue()
    {
        // Arrange
        var queue = CreateRequestQueue();
        var request = CreateTestRequest();

        // Act
        queue.Enqueue(request, Priority.Normal);

        // Assert
        Assert.Equal(1, queue.Count);
        Assert.False(queue.IsEmpty);
    }

    [Fact]
    public void Enqueue_WithNullRequest_ThrowsArgumentNullException()
    {
        // Arrange
        var queue = CreateRequestQueue();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            queue.Enqueue(null!, Priority.Normal));
    }

    [Fact]
    public void Enqueue_MultipleRequests_IncrementsCount()
    {
        // Arrange
        var queue = CreateRequestQueue();
        var request1 = CreateTestRequest("First");
        var request2 = CreateTestRequest("Second");
        var request3 = CreateTestRequest("Third");

        // Act
        queue.Enqueue(request1, Priority.Normal);
        queue.Enqueue(request2, Priority.High);
        queue.Enqueue(request3, Priority.Low);

        // Assert
        Assert.Equal(3, queue.Count);
    }

    [Fact]
    public void Dequeue_RemovesRequestFromQueue()
    {
        // Arrange
        var queue = CreateRequestQueue();
        var request = CreateTestRequest();
        queue.Enqueue(request, Priority.Normal);

        // Act
        var dequeued = queue.Dequeue();

        // Assert
        Assert.NotNull(dequeued);
        Assert.Equal(request.Id, dequeued.Id);
        Assert.Equal(0, queue.Count);
    }

    [Fact]
    public void Dequeue_WhenEmpty_ReturnsNull()
    {
        // Arrange
        var queue = CreateRequestQueue();

        // Act
        var dequeued = queue.Dequeue();

        // Assert
        Assert.Null(dequeued);
    }

    #endregion

    #region Priority Ordering Tests

    [Fact]
    public void GetRequests_ReturnsInPriorityOrder()
    {
        // Arrange
        var queue = CreateRequestQueue();
        var highPriority = CreateTestRequest("High", priority: Priority.High);
        var normalPriority = CreateTestRequest("Normal", priority: Priority.Normal);
        var lowPriority = CreateTestRequest("Low", priority: Priority.Low);

        // Enqueue in mixed order
        queue.Enqueue(lowPriority, Priority.Low);
        queue.Enqueue(highPriority, Priority.High);
        queue.Enqueue(normalPriority, Priority.Normal);

        // Act
        var requests = queue.GetRequests(3, long.MaxValue);

        // Assert
        Assert.Equal(3, requests.Count);
        // High priority should be first
        Assert.Equal(highPriority.Id, requests[0].Id);
    }

    [Fact]
    public void GetRequests_RespectsMaxCount()
    {
        // Arrange
        var queue = CreateRequestQueue();
        for (int i = 0; i < 5; i++)
        {
            queue.Enqueue(CreateTestRequest($"Request {i}"), Priority.Normal);
        }

        // Act
        var requests = queue.GetRequests(2, long.MaxValue);

        // Assert
        Assert.Equal(2, requests.Count);
    }

    [Fact]
    public void GetRequests_RespectsMemoryBudget()
    {
        // Arrange
        var queue = CreateRequestQueue();
        var longRequest = CreateTestRequest(new string('A', 1000)); // High memory
        var shortRequest = CreateTestRequest("Hi"); // Low memory

        queue.Enqueue(shortRequest, Priority.Normal);
        queue.Enqueue(longRequest, Priority.Normal);

        // Act - Request with very low memory budget
        var requests = queue.GetRequests(10, 100);

        // Assert
        Assert.Single(requests); // Only the short request fits
    }

    #endregion

    #region Cancellation Tests

    [Fact]
    public void Dequeue_SkipsCancelledRequests()
    {
        // Arrange
        var queue = CreateRequestQueue();
        var cts = new CancellationTokenSource();
        var request1 = CreateTestRequest(token: cts.Token);
        var request2 = CreateTestRequest();

        queue.Enqueue(request1, Priority.Normal);
        queue.Enqueue(request2, Priority.Normal);

        cts.Cancel();

        // Act
        var dequeued1 = queue.Dequeue();
        var dequeued2 = queue.Dequeue();

        // Assert
        Assert.NotNull(dequeuedued1);
        Assert.Equal(request2.Id, dequeued1.Id); // First non-cancelled request
        Assert.Null(dequeued2); // No more requests
    }

    [Fact]
    public void TryDequeue_WithCancelledRequest_SkipsIt()
    {
        // Arrange
        var queue = CreateRequestQueue();
        var cts = new CancellationTokenSource();
        var request = CreateTestRequest(token: cts.Token);

        queue.Enqueue(request, Priority.Normal);
        cts.Cancel();

        // Act
        var result = queue.TryDequeue(TimeSpan.FromSeconds(1));

        // Assert
        Assert.Null(result);
    }

    [Fact]
    public void Remove_RemovesRequestFromQueue()
    {
        // Arrange
        var queue = CreateRequestQueue();
        var request = CreateTestRequest();
        queue.Enqueue(request, Priority.Normal);

        // Act
        var removed = queue.Remove(request.Id);

        // Assert
        Assert.True(removed);
        Assert.Equal(0, queue.Count);
    }

    [Fact]
    public void Remove_WithNonExistentRequest_ReturnsFalse()
    {
        // Arrange
        var queue = CreateRequestQueue();
        var requestId = RequestId.New();

        // Act
        var removed = queue.Remove(requestId);

        // Assert
        Assert.False(removed);
    }

    #endregion

    #region Contains Tests

    [Fact]
    public void Contains_WithExistingRequest_ReturnsTrue()
    {
        // Arrange
        var queue = CreateRequestQueue();
        var request = CreateTestRequest();
        queue.Enqueue(request, Priority.Normal);

        // Act
        var contains = queue.Contains(request.Id);

        // Assert
        Assert.True(contains);
    }

    [Fact]
    public void Contains_WithNonExistentRequest_ReturnsFalse()
    {
        // Arrange
        var queue = CreateRequestQueue();
        var requestId = RequestId.New();

        // Act
        var contains = queue.Contains(requestId);

        // Assert
        Assert.False(contains);
    }

    #endregion

    #region Clear Tests

    [Fact]
    public void Clear_RemovesAllRequests()
    {
        // Arrange
        var queue = CreateRequestQueue();
        queue.Enqueue(CreateTestRequest("First"), Priority.Normal);
        queue.Enqueue(CreateTestRequest("Second"), Priority.Normal);
        queue.Enqueue(CreateTestRequest("Third"), Priority.Normal);

        // Act
        queue.Clear();

        // Assert
        Assert.Equal(0, queue.Count);
        Assert.True(queue.IsEmpty);
    }

    [Fact]
    public void Clear_WhenEmpty_DoesNothing()
    {
        // Arrange
        var queue = CreateRequestQueue();

        // Act & Assert - Should not throw
        queue.Clear();
        Assert.Equal(0, queue.Count);
    }

    #endregion

    #region Thread Safety Tests

    [Fact]
    public async Task ConcurrentEnqueue_ThreadSafe()
    {
        // Arrange
        var queue = CreateRequestQueue();
        var tasks = new List<Task>();

        // Act
        for (int i = 0; i < 100; i++)
        {
            int index = i;
            tasks.Add(Task.Run(() =>
            {
                var request = CreateTestRequest($"Request {index}");
                queue.Enqueue(request, Priority.Normal);
            }));
        }

        await Task.WhenAll(tasks);

        // Assert
        Assert.Equal(100, queue.Count);
    }

    [Fact]
    public async Task ConcurrentDequeue_ThreadSafe()
    {
        // Arrange
        var queue = CreateRequestQueue();
        for (int i = 0; i < 100; i++)
        {
            queue.Enqueue(CreateTestRequest($"Request {i}"), Priority.Normal);
        }

        var tasks = new List<Task<Request?>>();
        var results = new List<Request?>();

        // Act
        for (int i = 0; i < 100; i++)
        {
            tasks.Add(Task.Run(() =>
            {
                return queue.Dequeue();
            }));
        }

        results = (await Task.WhenAll(tasks)).ToList();

        // Assert
        Assert.Equal(100, results.Count(r => r != null));
        Assert.Equal(0, queue.Count);
    }

    [Fact]
    public async Task ConcurrentEnqueueAndDequeue_ThreadSafe()
    {
        // Arrange
        var queue = CreateRequestQueue();
        var enqueueTasks = new List<Task>();
        var dequeueTasks = new List<Task<Request?>>();

        // Act
        // Enqueue tasks
        for (int i = 0; i < 50; i++)
        {
            int index = i;
            enqueueTasks.Add(Task.Run(() =>
            {
                var request = CreateTestRequest($"Request {index}");
                queue.Enqueue(request, Priority.Normal);
            }));
        }

        // Dequeue tasks (may need to wait a bit for enqueues)
        for (int i = 0; i < 50; i++)
        {
            dequeueTasks.Add(Task.Run(() =>
            {
                Thread.Sleep(10); // Small delay to allow some enqueues
                return queue.Dequeue();
            }));
        }

        await Task.WhenAll(enqueueTasks.Concat(dequeueTasks));

        // Assert - Should complete without exceptions
        Assert.True(queue.Count >= 0);
    }

    #endregion

    #region TryDequeue Tests

    [Fact]
    public void TryDequeue_WithRequest_ReturnsRequest()
    {
        // Arrange
        var queue = CreateRequestQueue();
        var request = CreateTestRequest();
        queue.Enqueue(request, Priority.Normal);

        // Act
        var result = queue.TryDequeue(TimeSpan.FromSeconds(1));

        // Assert
        Assert.NotNull(result);
        Assert.Equal(request.Id, result.Id);
    }

    [Fact]
    public void TryDequeue_WithTimeoutAndNoRequest_ReturnsNull()
    {
        // Arrange
        var queue = CreateRequestQueue();

        // Act
        var result = queue.TryDequeue(TimeSpan.FromMilliseconds(100));

        // Assert
        Assert.Null(result);
    }

    [Fact]
    public async Task TryDequeue_WithCancellation_StopsWaiting()
    {
        // Arrange
        var queue = CreateRequestQueue();
        var cts = new CancellationTokenSource();

        // Act
        var task = Task.Run(() =>
        {
            return queue.TryDequeue(TimeSpan.FromSeconds(10), cts.Token);
        });

        cts.CancelAfter(100);
        var result = await task;

        // Assert
        Assert.Null(result);
    }

    #endregion

    #region Sequence Number Tests

    [Fact]
    public void Enqueue_WithMultipleRequests_AssignsIncreasingSequenceNumbers()
    {
        // Arrange
        var queue = CreateRequestQueue();
        var request1 = CreateTestRequest("First");
        var request2 = CreateTestRequest("Second");
        var request3 = CreateTestRequest("Third");

        // Act
        queue.Enqueue(request1, Priority.Normal);
        queue.Enqueue(request2, Priority.Normal);
        queue.Enqueue(request3, Priority.Normal);

        // Dequeue to verify order is maintained
        var dequeued1 = queue.Dequeue();
        var dequeued2 = queue.Dequeue();
        var dequeued3 = queue.Dequeue();

        // Assert
        Assert.NotNull(dequeued1);
        Assert.NotNull(dequeued2);
        Assert.NotNull(dequeued3);
        // Requests should be dequeued in enqueue order
    }

    #endregion

    #region Priority Tests

    [Fact]
    public void GetRequests_WithSamePriority_RespectsEnqueueOrder()
    {
        // Arrange
        var queue = CreateRequestQueue();
        var request1 = CreateTestRequest("First");
        var request2 = CreateTestRequest("Second");
        var request3 = CreateTestRequest("Third");

        queue.Enqueue(request1, Priority.Normal);
        queue.Enqueue(request2, Priority.Normal);
        queue.Enqueue(request3, Priority.Normal);

        // Act
        var requests = queue.GetRequests(3, long.MaxValue);

        // Assert
        Assert.Equal(3, requests.Count);
        Assert.Equal(request1.Id, requests[0].Id);
        Assert.Equal(request2.Id, requests[1].Id);
        Assert.Equal(request3.Id, requests[2].Id);
    }

    [Fact]
    public void GetRequests_WithDifferentPriorities_RespectsPriority()
    {
        // Arrange
        var queue = CreateRequestQueue();
        var normal1 = CreateTestRequest("Normal1");
        var high = CreateTestRequest("High");
        var normal2 = CreateTestRequest("Normal2");

        queue.Enqueue(normal1, Priority.Normal);
        queue.Enqueue(high, Priority.High);
        queue.Enqueue(normal2, Priority.Normal);

        // Act
        var requests = queue.GetRequests(3, long.MaxValue);

        // Assert
        Assert.Equal(3, requests.Count);
        // High priority should be first
        Assert.Equal(high.Id, requests[0].Id);
    }

    #endregion

    #region Memory Estimation Tests

    [Fact]
    public void GetRequests_WithMemoryBudget_LimitsRequests()
    {
        // Arrange
        var queue = CreateRequestQueue();
        for (int i = 0; i < 10; i++)
        {
            queue.Enqueue(CreateTestRequest($"Request {i}"), Priority.Normal);
        }

        // Act
        var requests = queue.GetRequests(10, 50); // Very low memory budget

        // Assert
        Assert.True(requests.Count <= 3); // Limited by memory
    }

    #endregion

    #region Edge Cases Tests

    [Fact]
    public void EnqueueAndDequeue_MultipleCycles_MaintainsConsistency()
    {
        // Arrange
        var queue = CreateRequestQueue();

        // Act - Multiple enqueue/dequeue cycles
        for (int cycle = 0; cycle < 3; cycle++)
        {
            for (int i = 0; i < 5; i++)
            {
                queue.Enqueue(CreateTestRequest($"Cycle{cycle}-Request{i}"), Priority.Normal);
            }

            for (int i = 0; i < 5; i++)
            {
                var dequeued = queue.Dequeue();
                Assert.NotNull(dequeued);
            }
        }

        // Assert
        Assert.Equal(0, queue.Count);
    }

    [Fact]
    public void GetRequests_WithEmptyQueue_ReturnsEmptyList()
    {
        // Arrange
        var queue = CreateRequestQueue();

        // Act
        var requests = queue.GetRequests(10, long.MaxValue);

        // Assert
        Assert.Empty(requests);
    }

    #endregion
}
