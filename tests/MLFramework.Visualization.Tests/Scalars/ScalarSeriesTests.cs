using MachineLearning.Visualization.Scalars;

namespace MLFramework.Visualization.Tests.Scalars;

public class ScalarSeriesTests
{
    [Test]
    public void Constructor_WithName_CreatesEmptySeries()
    {
        // Arrange & Act
        var series = new ScalarSeries("test_metric");

        // Assert
        Assert.That(series.Name, Is.EqualTo("test_metric"));
        Assert.That(series.Count, Is.EqualTo(0));
        Assert.That(series.Min, Is.Null);
        Assert.That(series.Max, Is.Null);
        Assert.That(series.Average, Is.EqualTo(0));
    }

    [Test]
    public void Constructor_WithNullName_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new ScalarSeries(null));
    }

    [Test]
    public void Constructor_WithEntries_CreatesSeriesWithEntries()
    {
        // Arrange
        var entries = new[]
        {
            new ScalarEntry(0, 1.0f),
            new ScalarEntry(1, 2.0f),
            new ScalarEntry(2, 3.0f)
        };

        // Act
        var series = new ScalarSeries("test_metric", entries);

        // Assert
        Assert.That(series.Count, Is.EqualTo(3));
        Assert.That(series.Min, Is.EqualTo(1.0f));
        Assert.That(series.Max, Is.EqualTo(3.0f));
        Assert.That(series.Average, Is.EqualTo(2.0f));
    }

    [Test]
    public void Add_AddsEntryToSeries()
    {
        // Arrange
        var series = new ScalarSeries("test_metric");
        var entry = new ScalarEntry(0, 1.5f);

        // Act
        series.Add(entry);

        // Assert
        Assert.That(series.Count, Is.EqualTo(1));
        Assert.That(series.Entries[0].Value, Is.EqualTo(1.5f));
        Assert.That(series.Min, Is.EqualTo(1.5f));
        Assert.That(series.Max, Is.EqualTo(1.5f));
        Assert.That(series.Average, Is.EqualTo(1.5f));
    }

    [Test]
    public void Add_WithNullEntry_ThrowsArgumentNullException()
    {
        // Arrange
        var series = new ScalarSeries("test_metric");

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => series.Add(null!));
    }

    [Test]
    public void AddRange_AddsMultipleEntries()
    {
        // Arrange
        var series = new ScalarSeries("test_metric");
        var entries = new[]
        {
            new ScalarEntry(0, 1.0f),
            new ScalarEntry(1, 2.0f),
            new ScalarEntry(2, 3.0f)
        };

        // Act
        series.AddRange(entries);

        // Assert
        Assert.That(series.Count, Is.EqualTo(3));
        Assert.That(series.Min, Is.EqualTo(1.0f));
        Assert.That(series.Max, Is.EqualTo(3.0f));
        Assert.That(series.Average, Is.EqualTo(2.0f));
    }

    [Test]
    public void AddRange_WithNullEntries_ThrowsArgumentNullException()
    {
        // Arrange
        var series = new ScalarSeries("test_metric");

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => series.AddRange(null!));
    }

    [Test]
    public void GetRange_WithValidRange_ReturnsEntriesInRange()
    {
        // Arrange
        var series = new ScalarSeries("test_metric");
        series.AddRange(new[]
        {
            new ScalarEntry(0, 1.0f),
            new ScalarEntry(1, 2.0f),
            new ScalarEntry(2, 3.0f),
            new ScalarEntry(3, 4.0f),
            new ScalarEntry(4, 5.0f)
        });

        // Act
        var range = series.GetRange(1, 3);

        // Assert
        Assert.That(range.Count(), Is.EqualTo(3));
        Assert.That(range.ElementAt(0).Step, Is.EqualTo(1));
        Assert.That(range.ElementAt(2).Step, Is.EqualTo(3));
    }

    [Test]
    public void GetRange_WithInvalidRange_ReturnsEmpty()
    {
        // Arrange
        var series = new ScalarSeries("test_metric");
        series.AddRange(new[]
        {
            new ScalarEntry(0, 1.0f),
            new ScalarEntry(1, 2.0f)
        });

        // Act
        var range = series.GetRange(10, 20);

        // Assert
        Assert.That(range.Count(), Is.EqualTo(0));
    }

    [Test]
    public void Smoothed_WithWindow_ReturnsSmoothedSeries()
    {
        // Arrange
        var series = new ScalarSeries("test_metric");
        series.AddRange(new[]
        {
            new ScalarEntry(0, 1.0f),
            new ScalarEntry(1, 2.0f),
            new ScalarEntry(2, 3.0f),
            new ScalarEntry(3, 4.0f),
            new ScalarEntry(4, 5.0f)
        });

        // Act
        var smoothed = series.Smoothed(3);

        // Assert
        Assert.That(smoothed, Is.Not.Null);
        Assert.That(smoothed.Count, Is.EqualTo(5));
        Assert.That(smoothed.Name, Is.EqualTo("test_metric_smoothed"));
        // Middle values should be smoothed
        Assert.That(smoothed.Entries[2].Value, Is.Not.EqualTo(3.0f));
    }

    [Test]
    public void Smoothed_WithInvalidWindow_ThrowsArgumentException()
    {
        // Arrange
        var series = new ScalarSeries("test_metric");

        // Act & Assert
        Assert.Throws<ArgumentException>(() => series.Smoothed(0));
        Assert.Throws<ArgumentException>(() => series.Smoothed(-1));
    }

    [Test]
    public void Resampled_WithTargetCount_ReturnsResampledSeries()
    {
        // Arrange
        var series = new ScalarSeries("test_metric");
        for (int i = 0; i < 100; i++)
        {
            series.Add(new ScalarEntry(i, (float)i));
        }

        // Act
        var resampled = series.Resampled(10);

        // Assert
        Assert.That(resampled, Is.Not.Null);
        Assert.That(resampled.Count, Is.EqualTo(10));
        Assert.That(resampled.Name, Is.EqualTo("test_metric_resampled"));
    }

    [Test]
    public void Resampled_WithInvalidTargetCount_ThrowsArgumentException()
    {
        // Arrange
        var series = new ScalarSeries("test_metric");

        // Act & Assert
        Assert.Throws<ArgumentException>(() => series.Resampled(0));
        Assert.Throws<ArgumentException>(() => series.Resampled(-1));
    }

    [Test]
    public void Resampled_WithTargetCountGreaterThanCount_ReturnsOriginal()
    {
        // Arrange
        var series = new ScalarSeries("test_metric");
        series.AddRange(new[]
        {
            new ScalarEntry(0, 1.0f),
            new ScalarEntry(1, 2.0f),
            new ScalarEntry(2, 3.0f)
        });

        // Act
        var resampled = series.Resampled(10);

        // Assert
        Assert.That(resampled.Count, Is.EqualTo(3));
        Assert.That(resampled.Entries[0].Value, Is.EqualTo(1.0f));
        Assert.That(resampled.Entries[1].Value, Is.EqualTo(2.0f));
        Assert.That(resampled.Entries[2].Value, Is.EqualTo(3.0f));
    }

    [Test]
    public void Statistics_AreCalculatedCorrectly()
    {
        // Arrange
        var series = new ScalarSeries("test_metric");
        series.AddRange(new[]
        {
            new ScalarEntry(0, 1.0f),
            new ScalarEntry(1, 2.0f),
            new ScalarEntry(2, 3.0f),
            new ScalarEntry(3, 4.0f),
            new ScalarEntry(4, 5.0f)
        });

        // Assert
        Assert.That(series.Min, Is.EqualTo(1.0f));
        Assert.That(series.Max, Is.EqualTo(5.0f));
        Assert.That(series.Average, Is.EqualTo(3.0f));
    }

    [Test]
    public void Statistics_WithNegativeValues_AreCalculatedCorrectly()
    {
        // Arrange
        var series = new ScalarSeries("test_metric");
        series.AddRange(new[]
        {
            new ScalarEntry(0, -5.0f),
            new ScalarEntry(1, -2.0f),
            new ScalarEntry(2, 0.0f),
            new ScalarEntry(3, 3.0f),
            new ScalarEntry(4, 7.0f)
        });

        // Assert
        Assert.That(series.Min, Is.EqualTo(-5.0f));
        Assert.That(series.Max, Is.EqualTo(7.0f));
        Assert.That(series.Average, Is.EqualTo(0.6f).Within(0.01f));
    }

    [Test]
    public void Statistics_WithSingleValue_AreCalculatedCorrectly()
    {
        // Arrange
        var series = new ScalarSeries("test_metric");
        series.Add(new ScalarEntry(0, 42.0f));

        // Assert
        Assert.That(series.Min, Is.EqualTo(42.0f));
        Assert.That(series.Max, Is.EqualTo(42.0f));
        Assert.That(series.Average, Is.EqualTo(42.0f));
    }

    [Test]
    public void Entries_ReturnsReadOnlyList()
    {
        // Arrange
        var series = new ScalarSeries("test_metric");
        series.Add(new ScalarEntry(0, 1.0f));

        // Act
        var entries = series.Entries;

        // Assert
        Assert.That(entries, Is.InstanceOf<IReadOnlyList<ScalarEntry>>());
    }

    [Test]
    public void Add_IsThreadSafe()
    {
        // Arrange
        var series = new ScalarSeries("test_metric");
        var tasks = new List<Task>();
        int numThreads = 10;
        int entriesPerThread = 100;

        // Act
        for (int t = 0; t < numThreads; t++)
        {
            int threadId = t;
            tasks.Add(Task.Run(() =>
            {
                for (int i = 0; i < entriesPerThread; i++)
                {
                    series.Add(new ScalarEntry(threadId * entriesPerThread + i, (float)(threadId * entriesPerThread + i)));
                }
            }));
        }
        Task.WaitAll(tasks.ToArray());

        // Assert
        Assert.That(series.Count, Is.EqualTo(numThreads * entriesPerThread));
    }
}
