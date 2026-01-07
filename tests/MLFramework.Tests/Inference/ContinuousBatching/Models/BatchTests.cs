using NUnit.Framework;

namespace MLFramework.Tests.Inference.ContinuousBatching;

[TestFixture]
public class BatchTests
{
    [Test]
    public void Constructor_InitializesWithCorrectProperties()
    {
        // Arrange
        var batchId = 42;

        // Act
        var batch = new Batch(batchId);

        // Assert
        Assert.That(batch.BatchId, Is.EqualTo(batchId));
        Assert.That(batch.Size, Is.EqualTo(0));
        Assert.That(batch.Requests, Is.Not.Null);
        Assert.That(batch.Requests.Count, Is.EqualTo(0));
        Assert.That(batch.EstimatedMemoryBytes, Is.EqualTo(0));
    }

    [Test]
    public void AddRequest_AddsRequestToBatch()
    {
        // Arrange
        var batch = new Batch(1);
        var request = CreateTestRequest();

        // Act
        batch.AddRequest(request);

        // Assert
        Assert.That(batch.Size, Is.EqualTo(1));
        Assert.That(batch.Contains(request.Id), Is.True);
    }

    [Test]
    public void AddRequest_WithNullRequest_ThrowsArgumentNullException()
    {
        // Arrange
        var batch = new Batch(1);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => batch.AddRequest(null!));
    }

    [Test]
    public void RemoveRequest_RemovesRequestFromBatch()
    {
        // Arrange
        var batch = new Batch(1);
        var request = CreateTestRequest();
        batch.AddRequest(request);

        // Act
        batch.RemoveRequest(request.Id);

        // Assert
        Assert.That(batch.Size, Is.EqualTo(0));
        Assert.That(batch.Contains(request.Id), Is.False);
    }

    [Test]
    public void Contains_ReturnsTrueForExistingRequest()
    {
        // Arrange
        var batch = new Batch(1);
        var request = CreateTestRequest();
        batch.AddRequest(request);

        // Act & Assert
        Assert.That(batch.Contains(request.Id), Is.True);
    }

    [Test]
    public void Contains_ReturnsFalseForNonExistingRequest()
    {
        // Arrange
        var batch = new Batch(1);
        var requestId = RequestId.New();

        // Act & Assert
        Assert.That(batch.Contains(requestId), Is.False);
    }

    [Test]
    public void GetRequest_ReturnsRequestForExistingId()
    {
        // Arrange
        var batch = new Batch(1);
        var request = CreateTestRequest();
        batch.AddRequest(request);

        // Act
        var retrieved = batch.GetRequest(request.Id);

        // Assert
        Assert.That(retrieved, Is.Not.Null);
        Assert.That(retrieved.Id, Is.EqualTo(request.Id));
    }

    [Test]
    public void GetRequest_ReturnsNullForNonExistingId()
    {
        // Arrange
        var batch = new Batch(1);
        var requestId = RequestId.New();

        // Act
        var retrieved = batch.GetRequest(requestId);

        // Assert
        Assert.That(retrieved, Is.Null);
    }

    [Test]
    public void Size_ReturnsCorrectCount()
    {
        // Arrange
        var batch = new Batch(1);
        var request1 = CreateTestRequest();
        var request2 = CreateTestRequest();
        var request3 = CreateTestRequest();

        batch.AddRequest(request1);
        batch.AddRequest(request2);
        batch.AddRequest(request3);

        // Act & Assert
        Assert.That(batch.Size, Is.EqualTo(3));
    }

    [Test]
    public void EstimatedMemoryBytes_CanBeSet()
    {
        // Arrange
        var batch = new Batch(1);

        // Act
        batch.EstimatedMemoryBytes = 1024 * 1024; // 1MB

        // Assert
        Assert.That(batch.EstimatedMemoryBytes, Is.EqualTo(1024 * 1024));
    }

    [Test]
    public void Requests_ReturnsReadOnlyList()
    {
        // Arrange
        var batch = new Batch(1);
        var request = CreateTestRequest();
        batch.AddRequest(request);

        // Act
        var requests = batch.Requests;

        // Assert
        Assert.That(requests, Is.Not.Null);
        Assert.That(requests.Count, Is.EqualTo(1));
        Assert.That(requests[0].Id, Is.EqualTo(request.Id));
    }

    [Test]
    public void CreatedTime_IsSetToUtcNow()
    {
        // Arrange
        var before = DateTime.UtcNow;
        var batch = new Batch(1);
        var after = DateTime.UtcNow;

        // Act & Assert
        Assert.That(batch.CreatedTime, Is.GreaterThanOrEqualTo(before));
        Assert.That(batch.CreatedTime, Is.LessThanOrEqualTo(after));
    }

    private Request CreateTestRequest()
    {
        return new Request(
            RequestId.New(),
            "Test prompt",
            100,
            CancellationToken.None
        );
    }
}
