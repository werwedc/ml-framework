using NUnit.Framework;
using MLFramework.Inference.ContinuousBatching;

namespace MLFramework.Tests.Inference.ContinuousBatching;

[TestFixture]
public class RequestTests
{
    [Test]
    public void Constructor_CreatesRequestWithCorrectProperties()
    {
        // Arrange
        var id = RequestId.New();
        var prompt = "Test prompt";
        var maxTokens = 100;
        var token = CancellationToken.None;
        var priority = Priority.High;

        // Act
        var request = new Request(id, prompt, maxTokens, token, priority);

        // Assert
        Assert.That(request.Id, Is.EqualTo(id));
        Assert.That(request.Prompt, Is.EqualTo(prompt));
        Assert.That(request.MaxTokens, Is.EqualTo(maxTokens));
        Assert.That(request.CancellationToken, Is.EqualTo(token));
        Assert.That(request.Priority, Is.EqualTo(priority));
    }

    [Test]
    public void Constructor_InitializesWithDefaultValues()
    {
        // Arrange
        var id = RequestId.New();
        var prompt = "Test prompt";
        var maxTokens = 100;
        var token = CancellationToken.None;

        // Act
        var request = new Request(id, prompt, maxTokens, token);

        // Assert
        Assert.That(request.GeneratedTokens, Is.EqualTo(0));
        Assert.That(request.IsCompleted, Is.False);
        Assert.That(request.GeneratedTokenIds, Is.Not.Null);
        Assert.That(request.GeneratedTokenIds.Count, Is.EqualTo(0));
        Assert.That(request.Priority, Is.EqualTo(Priority.Normal));
    }

    [Test]
    public void GeneratedTokens_CanBeSet()
    {
        // Arrange
        var request = CreateTestRequest();

        // Act
        request.GeneratedTokens = 50;

        // Assert
        Assert.That(request.GeneratedTokens, Is.EqualTo(50));
    }

    [Test]
    public void IsCompleted_CanBeSet()
    {
        // Arrange
        var request = CreateTestRequest();

        // Act
        request.IsCompleted = true;

        // Assert
        Assert.That(request.IsCompleted, Is.True);
    }

    [Test]
    public void GeneratedTokenIds_CanAccumulateTokens()
    {
        // Arrange
        var request = CreateTestRequest();

        // Act
        request.GeneratedTokenIds.Add(1);
        request.GeneratedTokenIds.Add(2);
        request.GeneratedTokenIds.Add(3);

        // Assert
        Assert.That(request.GeneratedTokenIds.Count, Is.EqualTo(3));
        Assert.That(request.GeneratedTokenIds[0], Is.EqualTo(1));
        Assert.That(request.GeneratedTokenIds[1], Is.EqualTo(2));
        Assert.That(request.GeneratedTokenIds[2], Is.EqualTo(3));
    }

    [Test]
    public void CompletionSource_CanBeSet()
    {
        // Arrange
        var request = CreateTestRequest();
        var expectedResult = "Generated text";

        // Act
        request.CompletionSource.SetResult(expectedResult);

        // Assert
        var result = await request.CompletionSource.Task;
        Assert.That(result, Is.EqualTo(expectedResult));
    }

    [Test]
    public void EnqueuedTime_IsSetToUtcNow()
    {
        // Arrange
        var before = DateTime.UtcNow;
        var request = CreateTestRequest();
        var after = DateTime.UtcNow;

        // Act & Assert
        Assert.That(request.EnqueuedTime, Is.GreaterThanOrEqualTo(before));
        Assert.That(request.EnqueuedTime, Is.LessThanOrEqualTo(after));
    }

    [Test]
    public void CancellationToken_CanBeChecked()
    {
        // Arrange
        var cts = new CancellationTokenSource();
        var request = new Request(
            RequestId.New(),
            "Test",
            100,
            cts.Token
        );

        // Act
        var isCancelled = request.CancellationToken.IsCancellationRequested;
        cts.Cancel();

        // Assert
        Assert.That(isCancelled, Is.False);
        Assert.That(request.CancellationToken.IsCancellationRequested, Is.True);
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
