using Xunit;
using Microsoft.Extensions.Logging.Abstractions;

namespace MLFramework.Tests.Memory;

public class MemoryPressureHandlerTests
{
    [Fact]
    public void TrackModelLoad_AddsModelToTracking()
    {
        var handler = new MemoryPressureHandler(logger: NullLogger<MemoryPressureHandler>.Instance);

        handler.TrackModelLoad("model1", "v1", 1000);

        var info = handler.GetLoadedModelsInfo().FirstOrDefault(m => m.ModelName == "model1" && m.Version == "v1");

        Assert.NotNull(info);
        Assert.Equal(1000, info.MemoryBytes);
        Assert.Equal(0, info.AccessCount);
        Assert.False(info.IsPinned);
    }

    [Fact]
    public void TrackModelAccess_UpdatesLastAccessTimeAndAccessCount()
    {
        var handler = new MemoryPressureHandler(logger: NullLogger<MemoryPressureHandler>.Instance);

        handler.TrackModelLoad("model1", "v1", 1000);

        System.Threading.Thread.Sleep(10); // Small delay to ensure time difference

        handler.TrackModelAccess("model1", "v1");

        var info = handler.GetLoadedModelsInfo().First(m => m.ModelName == "model1" && m.Version == "v1");

        Assert.Equal(1, info.AccessCount);
        Assert.True((DateTime.UtcNow - info.LastAccessTime) < TimeSpan.FromSeconds(1));
    }

    [Fact]
    public void PinModel_PreventsEviction()
    {
        var handler = new MemoryPressureHandler(logger: NullLogger<MemoryPressureHandler>.Instance);

        handler.TrackModelLoad("model1", "v1", 1000);
        handler.PinModel("model1", "v1");
        handler.SetMemoryThreshold(100); // Set low threshold to trigger eviction

        var result = handler.EvictIfNeededAsync().GetAwaiter().GetResult();

        Assert.Equal(0, result.EvictedVersions.Count);
        var info = handler.GetLoadedModelsInfo().FirstOrDefault(m => m.ModelName == "model1" && m.Version == "v1");
        Assert.NotNull(info);
    }

    [Fact]
    public void EvictUnpinnedModel_RemovesFromTracking()
    {
        var canEvictCallback = (string _, string __) => Task.FromResult(true);
        var unloadCallback = (string _, string __) => Task.CompletedTask;
        var handler = new MemoryPressureHandler(canEvictCallback, unloadCallback, NullLogger<MemoryPressureHandler>.Instance);

        handler.TrackModelLoad("model1", "v1", 1000);
        handler.SetMemoryThreshold(100); // Set low threshold to trigger eviction

        var result = handler.EvictIfNeededAsync().GetAwaiter().GetResult();

        Assert.Single(result.EvictedVersions);
        Assert.Contains("model1:v1", result.EvictedVersions);
        Assert.Equal(1000, result.BytesFreed);
        var info = handler.GetLoadedModelsInfo().FirstOrDefault(m => m.ModelName == "model1" && m.Version == "v1");
        Assert.Null(info);
    }

    [Fact]
    public void GetTotalMemoryUsage_ReturnsCorrectSum()
    {
        var handler = new MemoryPressureHandler(logger: NullLogger<MemoryPressureHandler>.Instance);

        handler.TrackModelLoad("model1", "v1", 1000);
        handler.TrackModelLoad("model2", "v1", 2000);
        handler.TrackModelLoad("model3", "v1", 3000);

        var total = handler.GetTotalMemoryUsage();

        Assert.Equal(6000, total);
    }

    [Fact]
    public void SetMemoryThreshold_UpdatesThreshold()
    {
        var handler = new MemoryPressureHandler(logger: NullLogger<MemoryPressureHandler>.Instance);

        handler.SetMemoryThreshold(5000);

        handler.TrackModelLoad("model1", "v1", 1000);
        handler.TrackModelLoad("model2", "v1", 2000);

        var result = handler.EvictIfNeededAsync().GetAwaiter().GetResult();

        Assert.Equal(0, result.EvictedVersions.Count); // 3000 < 5000 threshold
    }

    [Fact]
    public void EvictIfNeeded_FreesRequiredMemory()
    {
        var canEvictCallback = (string _, string __) => Task.FromResult(true);
        var unloadCallback = (string _, string __) => Task.CompletedTask;
        var handler = new MemoryPressureHandler(canEvictCallback, unloadCallback, NullLogger<MemoryPressureHandler>.Instance);

        handler.TrackModelLoad("model1", "v1", 1000);
        handler.TrackModelLoad("model2", "v1", 2000);
        handler.TrackModelLoad("model3", "v1", 3000);
        handler.SetMemoryThreshold(4000); // Total is 6000, need to free 2000

        var result = handler.EvictIfNeededAsync().GetAwaiter().GetResult();

        Assert.True(result.BytesFreed >= 2000);
        Assert.True(handler.GetTotalMemoryUsage() <= 4000);
    }

    [Fact]
    public void EvictIfNeeded_RespectsActiveReferences()
    {
        var canEvictCallback = (string model, string version) =>
        {
            // model2:v1 has active references
            return Task.FromResult(model != "model2" || version != "v1");
        };
        var unloadCallback = (string _, string __) => Task.CompletedTask;
        var handler = new MemoryPressureHandler(canEvictCallback, unloadCallback, NullLogger<MemoryPressureHandler>.Instance);

        handler.TrackModelLoad("model1", "v1", 1000);
        handler.TrackModelLoad("model2", "v1", 2000); // Has active references
        handler.TrackModelLoad("model3", "v1", 3000);
        handler.SetMemoryThreshold(1000);

        var result = handler.EvictIfNeededAsync().GetAwaiter().GetResult();

        Assert.DoesNotContain("model2:v1", result.EvictedVersions);
        var model2Info = handler.GetLoadedModelsInfo().FirstOrDefault(m => m.ModelName == "model2" && m.Version == "v1");
        Assert.NotNull(model2Info);
    }

    [Fact]
    public void LRUEvictionOrder_IsCorrect()
    {
        var canEvictCallback = (string _, string __) => Task.FromResult(true);
        var unloadCallback = (string _, string __) => Task.CompletedTask;
        var handler = new MemoryPressureHandler(canEvictCallback, unloadCallback, NullLogger<MemoryPressureHandler>.Instance);

        handler.TrackModelLoad("model1", "v1", 1000);
        System.Threading.Thread.Sleep(10);

        handler.TrackModelLoad("model2", "v1", 1000);
        System.Threading.Thread.Sleep(10);

        handler.TrackModelLoad("model3", "v1", 1000);

        // Access model3 to make it more recent
        handler.TrackModelAccess("model3", "v1");

        handler.SetMemoryThreshold(2000);

        var result = handler.EvictIfNeededAsync().GetAwaiter().GetResult();

        // model1 should be evicted first (least recently accessed)
        Assert.Contains("model1:v1", result.EvictedVersions);
    }

    [Fact]
    public void ConcurrentAccessTracking_ThreadSafe()
    {
        var handler = new MemoryPressureHandler(logger: NullLogger<MemoryPressureHandler>.Instance);

        // Load 10 models
        for (int i = 0; i < 10; i++)
        {
            handler.TrackModelLoad($"model{i}", "v1", 1000);
        }

        // Spawn 100 threads to access models concurrently
        var tasks = new List<Task>();
        for (int i = 0; i < 100; i++)
        {
            var modelIndex = i % 10;
            var task = Task.Run(() =>
            {
                handler.TrackModelAccess($"model{modelIndex}", "v1");
            });
            tasks.Add(task);
        }

        Task.WhenAll(tasks).GetAwaiter().GetResult();

        // Verify all models have the correct access counts
        var models = handler.GetLoadedModelsInfo();
        foreach (var model in models)
        {
            Assert.Equal(10, model.AccessCount);
        }
    }

    [Fact]
    public void EvictionDecision_PerformanceTest()
    {
        var canEvictCallback = (string _, string __) => Task.FromResult(true);
        var unloadCallback = (string _, string __) => Task.CompletedTask;
        var handler = new MemoryPressureHandler(canEvictCallback, unloadCallback, NullLogger<MemoryPressureHandler>.Instance);

        // Load 100 models
        for (int i = 0; i < 100; i++)
        {
            handler.TrackModelLoad($"model{i}", "v1", 10_000_000); // 10MB each
        }

        handler.SetMemoryThreshold(500_000_000); // 500MB threshold

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var result = handler.EvictIfNeededAsync().GetAwaiter().GetResult();
        stopwatch.Stop();

        Assert.True(stopwatch.Elapsed.TotalMilliseconds < 10,
            $"Eviction decision took {stopwatch.Elapsed.TotalMilliseconds:F2}ms, expected < 10ms");
    }

    [Fact]
    public void UntrackModel_RemovesFromTracking()
    {
        var handler = new MemoryPressureHandler(logger: NullLogger<MemoryPressureHandler>.Instance);

        handler.TrackModelLoad("model1", "v1", 1000);
        handler.UntrackModel("model1", "v1");

        var info = handler.GetLoadedModelsInfo().FirstOrDefault(m => m.ModelName == "model1" && m.Version == "v1");
        Assert.Null(info);
    }

    [Fact]
    public void PinAndUnpinModel_TogglesPinnedStatus()
    {
        var handler = new MemoryPressureHandler(logger: NullLogger<MemoryPressureHandler>.Instance);

        handler.TrackModelLoad("model1", "v1", 1000);

        handler.PinModel("model1", "v1");
        var info = handler.GetLoadedModelsInfo().First(m => m.ModelName == "model1" && m.Version == "v1");
        Assert.True(info.IsPinned);

        handler.UnpinModel("model1", "v1");
        info = handler.GetLoadedModelsInfo().First(m => m.ModelName == "model1" && m.Version == "v1");
        Assert.False(info.IsPinned);
    }

    [Fact]
    public void TrackModelLoad_WithNegativeMemory_ThrowsException()
    {
        var handler = new MemoryPressureHandler(logger: NullLogger<MemoryPressureHandler>.Instance);

        Assert.Throws<ArgumentException>(() => handler.TrackModelLoad("model1", "v1", -1000));
    }

    [Fact]
    public void SetMemoryThreshold_WithNegativeValue_ThrowsException()
    {
        var handler = new MemoryPressureHandler(logger: NullLogger<MemoryPressureHandler>.Instance);

        Assert.Throws<ArgumentException>(() => handler.SetMemoryThreshold(-100));
    }

    [Fact]
    public void GetLoadedModelsInfo_ReturnsAllModels()
    {
        var handler = new MemoryPressureHandler(logger: NullLogger<MemoryPressureHandler>.Instance);

        handler.TrackModelLoad("model1", "v1", 1000);
        handler.TrackModelLoad("model1", "v2", 2000);
        handler.TrackModelLoad("model2", "v1", 3000);

        var models = handler.GetLoadedModelsInfo();

        Assert.Equal(3, models.Count());
    }

    [Fact]
    public void ModelWeight_CombinesLRUAndLFU()
    {
        var handler = new MemoryPressureHandler(logger: NullLogger<MemoryPressureHandler>.Instance);

        handler.TrackModelLoad("model1", "v1", 1000);
        System.Threading.Thread.Sleep(10);

        handler.TrackModelLoad("model2", "v1", 1000);

        // Access model1 multiple times to increase its LFU score
        for (int i = 0; i < 5; i++)
        {
            handler.TrackModelAccess("model1", "v1");
        }

        var models = handler.GetLoadedModelsInfo().OrderBy(m => m.Weight).ToList();

        // model2 should have lower weight (less frequent access) even though model1 is older
        Assert.Equal("model2", models[0].ModelName);
    }
}
