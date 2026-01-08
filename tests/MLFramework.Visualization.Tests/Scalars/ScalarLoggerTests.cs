using MachineLearning.Visualization.Events;
using MachineLearning.Visualization.Scalars;
using MachineLearning.Visualization.Storage;
using Moq;

namespace MLFramework.Visualization.Tests.Scalars;

public class ScalarLoggerTests
{
    private Mock<IStorageBackend> _mockStorage;
    private IEventPublisher _eventPublisher;

    [SetUp]
    public void Setup()
    {
        _mockStorage = new Mock<IStorageBackend>();
        _eventPublisher = new EventSystem();
    }

    [Test]
    public void LogScalar_WithValidNameAndValue_AddsEntryToSeries()
    {
        // Arrange
        var logger = new ScalarLogger(_mockStorage.Object);

        // Act
        logger.LogScalar("test_metric", 1.5f);

        // Assert
        var series = logger.GetSeries("test_metric");
        Assert.That(series, Is.Not.Null);
        Assert.That(series.Count, Is.EqualTo(1));
        Assert.That(series.Entries[0].Value, Is.EqualTo(1.5f));
    }

    [Test]
    public void LogScalar_WithAutoIncrement_UsesIncreasingSteps()
    {
        // Arrange
        var logger = new ScalarLogger(_mockStorage.Object);

        // Act
        logger.LogScalar("test_metric", 1.0f);
        logger.LogScalar("test_metric", 2.0f);
        logger.LogScalar("test_metric", 3.0f);

        // Assert
        var series = logger.GetSeries("test_metric");
        Assert.That(series.Count, Is.EqualTo(3));
        Assert.That(series.Entries[0].Step, Is.EqualTo(0));
        Assert.That(series.Entries[1].Step, Is.EqualTo(1));
        Assert.That(series.Entries[2].Step, Is.EqualTo(2));
    }

    [Test]
    public void LogScalar_WithExplicitStep_UsesProvidedStep()
    {
        // Arrange
        var logger = new ScalarLogger(_mockStorage.Object);

        // Act
        logger.LogScalar("test_metric", 1.0f, step: 100);
        logger.LogScalar("test_metric", 2.0f, step: 200);

        // Assert
        var series = logger.GetSeries("test_metric");
        Assert.That(series.Count, Is.EqualTo(2));
        Assert.That(series.Entries[0].Step, Is.EqualTo(100));
        Assert.That(series.Entries[1].Step, Is.EqualTo(200));
    }

    [Test]
    public void LogScalar_WithDoubleValue_ConvertsToFloat()
    {
        // Arrange
        var logger = new ScalarLogger(_mockStorage.Object);
        double value = 1.23456789;

        // Act
        logger.LogScalar("test_metric", value);

        // Assert
        var series = logger.GetSeries("test_metric");
        Assert.That(series.Entries[0].Value, Is.EqualTo((float)value).Within(0.0001f));
    }

    [Test]
    public void LogScalar_WithNullName_ThrowsArgumentNullException()
    {
        // Arrange
        var logger = new ScalarLogger(_mockStorage.Object);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => logger.LogScalar(null, 1.0f));
    }

    [Test]
    public void GetSeries_WithNonExistentName_ReturnsNull()
    {
        // Arrange
        var logger = new ScalarLogger(_mockStorage.Object);

        // Act
        var series = logger.GetSeries("non_existent");

        // Assert
        Assert.That(series, Is.Null);
    }

    [Test]
    public void GetAllSeries_ReturnsAllLoggedSeries()
    {
        // Arrange
        var logger = new ScalarLogger(_mockStorage.Object);
        logger.LogScalar("metric1", 1.0f);
        logger.LogScalar("metric2", 2.0f);
        logger.LogScalar("metric3", 3.0f);

        // Act
        var allSeries = logger.GetAllSeries();

        // Assert
        Assert.That(allSeries.Count(), Is.EqualTo(3));
        Assert.That(allSeries.Any(s => s.Name == "metric1"), Is.True);
        Assert.That(allSeries.Any(s => s.Name == "metric2"), Is.True);
        Assert.That(allSeries.Any(s => s.Name == "metric3"), Is.True);
    }

    [Test]
    public void GetSmoothedSeries_ReturnsSmoothedValues()
    {
        // Arrange
        var logger = new ScalarLogger(_mockStorage.Object);
        logger.LogScalar("test_metric", 1.0f);
        logger.LogScalar("test_metric", 2.0f);
        logger.LogScalar("test_metric", 3.0f);
        logger.LogScalar("test_metric", 4.0f);
        logger.LogScalar("test_metric", 5.0f);

        // Act
        var smoothed = logger.GetSmoothedSeries("test_metric", windowSize: 3);

        // Assert
        Assert.That(smoothed, Is.Not.Null);
        Assert.That(smoothed.Count, Is.EqualTo(5));
        // Check that smoothed values are different from original
        var original = logger.GetSeries("test_metric");
        Assert.That(smoothed.Entries[2].Value, Is.Not.EqualTo(original.Entries[2].Value));
    }

    [Test]
    public void GetLatestValues_ReturnsMostRecentValueForEachMetric()
    {
        // Arrange
        var logger = new ScalarLogger(_mockStorage.Object);
        logger.LogScalar("metric1", 1.0f);
        logger.LogScalar("metric1", 2.0f);
        logger.LogScalar("metric2", 3.0f);

        // Act
        var latest = logger.GetLatestValues();

        // Assert
        Assert.That(latest.Count, Is.EqualTo(2));
        Assert.That(latest["metric1"], Is.EqualTo(2.0f));
        Assert.That(latest["metric2"], Is.EqualTo(3.0f));
    }

    [Test]
    public void LogScalar_PublishesEvent()
    {
        // Arrange
        var mockPublisher = new Mock<IEventPublisher>();
        var logger = new ScalarLogger(mockPublisher.Object);

        // Act
        logger.LogScalar("test_metric", 1.5f);

        // Assert
        mockPublisher.Verify(p => p.Publish(It.IsAny<ScalarMetricEvent>()), Times.Once);
    }

    [Test]
    public void LogScalar_StoresEvent()
    {
        // Arrange
        var logger = new ScalarLogger(_mockStorage.Object);

        // Act
        logger.LogScalar("test_metric", 1.5f);

        // Assert
        _mockStorage.Verify(s => s.StoreEvent(It.IsAny<ScalarMetricEvent>()), Times.Once);
    }

    [Test]
    public void MaxEntriesPerSeries_LimitsSeriesSize()
    {
        // Arrange
        var logger = new ScalarLogger(_mockStorage.Object);
        logger.MaxEntriesPerSeries = 3;

        // Act
        for (int i = 0; i < 10; i++)
        {
            logger.LogScalar("test_metric", (float)i);
        }

        // Assert
        var series = logger.GetSeries("test_metric");
        Assert.That(series.Count, Is.LessThanOrEqualTo(logger.MaxEntriesPerSeries + 1)); // +1 for buffer
    }

    [Test]
    public void Dispose_FlushesStorage()
    {
        // Arrange
        var logger = new ScalarLogger(_mockStorage.Object);
        logger.LogScalar("test_metric", 1.0f);

        // Act
        logger.Dispose();

        // Assert
        _mockStorage.Verify(s => s.Flush(), Times.Once);
    }

    [Test]
    public void LogScalarAsync_CompletesSuccessfully()
    {
        // Arrange
        var logger = new ScalarLogger(_mockStorage.Object);

        // Act
        var task = logger.LogScalarAsync("test_metric", 1.5f);

        // Assert
        Assert.That(task, Is.Not.Null);
        Assert.That(task.IsCompleted, Is.True);
        var series = logger.GetSeries("test_metric");
        Assert.That(series, Is.Not.Null);
        Assert.That(series.Count, Is.EqualTo(1));
    }

    [Test]
    public void GetSeriesAsync_CompletesSuccessfully()
    {
        // Arrange
        var logger = new ScalarLogger(_mockStorage.Object);
        logger.LogScalar("test_metric", 1.5f);

        // Act
        var task = logger.GetSeriesAsync("test_metric");

        // Assert
        Assert.That(task, Is.Not.Null);
        Assert.That(task.IsCompleted, Is.True);
        Assert.That(task.Result, Is.Not.Null);
        Assert.That(task.Result.Count, Is.EqualTo(1));
    }
}
