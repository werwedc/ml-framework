using MLFramework.ModelZoo.Progressive;
using MLFramework.Core;
using MLFramework.ModelVersioning;
using Xunit;
using System.IO;
using System.Linq;

namespace ModelZooTests.Progressive;

/// <summary>
/// Unit tests for progressive model loading functionality.
/// </summary>
public class ProgressiveLoadingTests : IDisposable
{
    private readonly string _testCacheDir;
    private readonly List<string> _tempFiles;

    public ProgressiveLoadingTests()
    {
        _testCacheDir = Path.Combine(Path.GetTempPath(), "ProgressiveLoadingTests");
        Directory.CreateDirectory(_testCacheDir);
        _tempFiles = new List<string>();
    }

    public void Dispose()
    {
        // Clean up temporary files
        foreach (var file in _tempFiles)
        {
            try
            {
                if (File.Exists(file))
                {
                    File.Delete(file);
                }
            }
            catch
            {
                // Ignore cleanup errors
            }
        }

        // Clean up test directory
        try
        {
            if (Directory.Exists(_testCacheDir))
            {
                Directory.Delete(_testCacheDir, true);
            }
        }
        catch
        {
            // Ignore cleanup errors
        }
    }

    private string CreateTempWeightFile(float[] data, int[] shape)
    {
        string fileName = $"weights_{Guid.NewGuid()}.bin";
        string filePath = Path.Combine(_testCacheDir, fileName);
        _tempFiles.Add(filePath);

        using var writer = new BinaryWriter(File.Create(filePath));
        foreach (var value in data)
        {
            writer.Write(value);
        }

        return filePath;
    }

    private ProgressiveModelLoader CreateTestLoader(string modelName = "test_model", ProgressiveLoadOptions? options = null)
    {
        var metadata = new ModelMetadata
        {
            ModelName = modelName,
            Architecture = "CNN",
            InputShape = new[] { 1, 3, 224, 224 },
            OutputShape = new[] { 1, 1000 }
        };

        string cachePath = Path.Combine(_testCacheDir, $"{modelName}.bin");
        var device = Device.CreateCpu("TestCPU");
        var loadOrder = LayerLoadOrder.ForCNN();
        options ??= new ProgressiveLoadOptions();

        var context = new ProgressiveLoadContext(metadata, device, cachePath, options, loadOrder);
        return new ProgressiveModelLoader(context);
    }

    [Fact]
    public void Constructor_WithValidContext_CreatesLoader()
    {
        // Arrange
        var options = new ProgressiveLoadOptions();
        var metadata = new ModelMetadata { ModelName = "test", Architecture = "CNN" };
        var device = Device.CreateCpu();
        var loadOrder = LayerLoadOrder.ForCNN();
        var context = new ProgressiveLoadContext(metadata, device, "test.bin", options, loadOrder);

        // Act
        var loader = new ProgressiveModelLoader(context);

        // Assert
        Assert.NotNull(loader);
        Assert.NotNull(loader.Context);
        Assert.Equal(0.0, loader.GetLoadingProgress());
        Assert.False(loader.IsFullyLoaded());
    }

    [Fact]
    public void LoadProgressive_WithModelName_ReturnsLoader()
    {
        // Act
        var loader = ProgressiveModelLoader.LoadProgressive("test_model");

        // Assert
        Assert.NotNull(loader);
        Assert.NotNull(loader.Context);
    }

    [Fact]
    public void LoadProgressive_WithMetadata_ReturnsLoader()
    {
        // Arrange
        var metadata = new ModelMetadata
        {
            ModelName = "test_model",
            Architecture = "Transformer"
        };

        // Act
        var loader = ProgressiveModelLoader.LoadProgressive(metadata);

        // Assert
        Assert.NotNull(loader);
        Assert.Equal("Transformer", loader.Context.Metadata.Architecture);
    }

    [Fact]
    public void GetLoadingProgress_Initially_ReturnsZero()
    {
        // Arrange
        var loader = CreateTestLoader();

        // Act
        double progress = loader.GetLoadingProgress();

        // Assert
        Assert.Equal(0.0, progress);
    }

    [Fact]
    public void IsFullyLoaded_Initially_ReturnsFalse()
    {
        // Arrange
        var loader = CreateTestLoader();

        // Act
        bool isFullyLoaded = loader.IsFullyLoaded();

        // Assert
        Assert.False(isFullyLoaded);
    }

    [Fact]
    public void RegisterParameter_WithValidData_CreatesLazyParameter()
    {
        // Arrange
        var loader = CreateTestLoader();
        var data = new float[] { 1f, 2f, 3f, 4f };
        var shape = new[] { 2, 2 };
        var weightPath = CreateTempWeightFile(data, shape);

        // Act
        var param = loader.RegisterParameter("layer1", weightPath, shape);

        // Assert
        Assert.NotNull(param);
        Assert.Equal("layer1", param.LayerName);
        Assert.Equal(weightPath, param.WeightPath);
        Assert.False(param.IsLoaded);
    }

    [Fact]
    public void GetParameter_WithRegisteredLayer_ReturnsParameter()
    {
        // Arrange
        var loader = CreateTestLoader();
        var data = new float[] { 1f, 2f, 3f, 4f };
        var shape = new[] { 2, 2 };
        var weightPath = CreateTempWeightFile(data, shape);
        loader.RegisterParameter("layer1", weightPath, shape);

        // Act
        var param = loader.GetParameter("layer1");

        // Assert
        Assert.NotNull(param);
        Assert.Equal("layer1", param.LayerName);
    }

    [Fact]
    public void GetParameter_WithUnregisteredLayer_ReturnsNull()
    {
        // Arrange
        var loader = CreateTestLoader();

        // Act
        var param = loader.GetParameter("nonexistent");

        // Assert
        Assert.Null(param);
    }

    [Fact]
    public void LazyParameter_DataAccess_EnsuresLoaded()
    {
        // Arrange
        var loader = CreateTestLoader();
        var data = new float[] { 1f, 2f, 3f, 4f };
        var shape = new[] { 2, 2 };
        var weightPath = CreateTempWeightFile(data, shape);
        var param = loader.RegisterParameter("layer1", weightPath, shape);

        // Act
        var loadedData = param.Data;

        // Assert
        Assert.NotNull(loadedData);
        Assert.Equal(4, loadedData.Length);
        Assert.True(param.IsLoaded);
    }

    [Fact]
    public void PrefetchLayer_WithValidLayer_LoadsLayer()
    {
        // Arrange
        var loader = CreateTestLoader();
        var data = new float[] { 1f, 2f, 3f, 4f };
        var shape = new[] { 2, 2 };
        var weightPath = CreateTempWeightFile(data, shape);
        loader.RegisterParameter("layer1", weightPath, shape);

        // Act
        loader.PrefetchLayer("layer1");

        // Assert
        var param = loader.GetParameter("layer1");
        Assert.NotNull(param);
        Assert.True(param.IsLoaded);
    }

    [Fact]
    public void PrefetchLayers_WithMultipleLayers_LoadsAllLayers()
    {
        // Arrange
        var loader = CreateTestLoader();
        var data = new float[] { 1f, 2f, 3f, 4f };
        var shape = new[] { 2, 2 };

        var path1 = CreateTempWeightFile(data, shape);
        var path2 = CreateTempWeightFile(data, shape);
        var path3 = CreateTempWeightFile(data, shape);

        loader.RegisterParameter("layer1", path1, shape);
        loader.RegisterParameter("layer2", path2, shape);
        loader.RegisterParameter("layer3", path3, shape);

        // Act
        loader.PrefetchLayers(new[] { "layer1", "layer2", "layer3" });

        // Assert
        Assert.True(loader.GetParameter("layer1")?.IsLoaded);
        Assert.True(loader.GetParameter("layer2")?.IsLoaded);
        Assert.True(loader.GetParameter("layer3")?.IsLoaded);
    }

    [Fact]
    public void OnLayerLoadedEvent_WhenLayerLoads_FiresEvent()
    {
        // Arrange
        var loader = CreateTestLoader();
        var data = new float[] { 1f, 2f, 3f, 4f };
        var shape = new[] { 2, 2 };
        var weightPath = CreateTempWeightFile(data, shape);
        loader.RegisterParameter("layer1", weightPath, shape);

        string? loadedLayer = null;
        loader.OnLayerLoaded += (sender, args) => loadedLayer = args.LayerName;

        // Act
        loader.PrefetchLayer("layer1");

        // Assert
        Assert.Equal("layer1", loadedLayer);
    }

    [Fact]
    public void OnProgressChangedEvent_WhenLayersLoad_FiresEvent()
    {
        // Arrange
        var loader = CreateTestLoader();
        var data = new double[3]; // Track progress changes
        var shape = new[] { 2, 2 };

        var path1 = CreateTempWeightFile(new float[] { 1, 2, 3, 4 }, shape);
        var path2 = CreateTempWeightFile(new float[] { 5, 6, 7, 8 }, shape);
        var path3 = CreateTempWeightFile(new float[] { 9, 10, 11, 12 }, shape);

        loader.RegisterParameter("layer1", path1, shape);
        loader.RegisterParameter("layer2", path2, shape);
        loader.RegisterParameter("layer3", path3, shape);

        int eventCount = 0;
        loader.OnProgressChanged += (sender, args) =>
        {
            if (eventCount < 3)
            {
                data[eventCount] = args.Progress;
                eventCount++;
            }
        };

        // Act
        loader.PrefetchLayer("layer1");
        loader.PrefetchLayer("layer2");
        loader.PrefetchLayer("layer3");

        // Assert
        Assert.True(eventCount >= 3);
        Assert.Equal(1.0 / 3.0, data[0]);
        Assert.Equal(2.0 / 3.0, data[1]);
        Assert.Equal(1.0, data[2]);
    }

    [Fact]
    public void OnFullyLoadedEvent_WhenAllLayersLoad_FiresEvent()
    {
        // Arrange
        var loader = CreateTestLoader();
        var data = new float[] { 1f, 2f, 3f, 4f };
        var shape = new[] { 2, 2 };

        var path1 = CreateTempWeightFile(data, shape);
        var path2 = CreateTempWeightFile(data, shape);

        loader.RegisterParameter("layer1", path1, shape);
        loader.RegisterParameter("layer2", path2, shape);

        bool fullyLoadedFired = false;
        loader.OnFullyLoaded += (sender, args) => fullyLoadedFired = true;

        // Act
        loader.PrefetchLayer("layer1");
        loader.PrefetchLayer("layer2");

        // Assert
        Assert.True(fullyLoadedFired);
        Assert.True(loader.IsFullyLoaded());
    }

    [Fact]
    public void ProgressiveLoadOptions_DefaultValues_AreCorrect()
    {
        // Act
        var options = new ProgressiveLoadOptions();

        // Assert
        Assert.Equal(LayerLoadingStrategy.OnDemand, options.Strategy);
        Assert.Equal(3, options.MaxConcurrentLoads);
        Assert.Equal(1, options.PrefetchCount);
        Assert.Equal(UnloadStrategy.Never, options.UnloadStrategy);
        Assert.Equal(-1, options.MaxLoadedLayers);
    }

    [Fact]
    public void ProgressiveLoadOptions_WithStrategy_SetsCorrectly()
    {
        // Act
        var options = new ProgressiveLoadOptions(LayerLoadingStrategy.Parallel);

        // Assert
        Assert.Equal(LayerLoadingStrategy.Parallel, options.Strategy);
    }

    [Fact]
    public void LayerLoadOrder_ForCNN_CreatesCorrectOrder()
    {
        // Act
        var loadOrder = LayerLoadOrder.ForCNN();

        // Assert
        Assert.NotNull(loadOrder);
        Assert.Equal(ModelArchitectureType.CNN, loadOrder.ArchitectureType);
        Assert.NotNull(loadOrder.OrderedLayers);
    }

    [Fact]
    public void LayerLoadOrder_ForTransformer_CreatesCorrectOrder()
    {
        // Act
        var loadOrder = LayerLoadOrder.ForTransformer();

        // Assert
        Assert.NotNull(loadOrder);
        Assert.Equal(ModelArchitectureType.Transformer, loadOrder.ArchitectureType);
    }

    [Fact]
    public void LayerLoadOrder_ForRNN_CreatesCorrectOrder()
    {
        // Act
        var loadOrder = LayerLoadOrder.ForRNN();

        // Assert
        Assert.NotNull(loadOrder);
        Assert.Equal(ModelArchitectureType.RNN, loadOrder.ArchitectureType);
    }

    [Fact]
    public void LayerLoadOrder_AddDependency_RespectedInGetLoadOrder()
    {
        // Arrange
        var loadOrder = new LayerLoadOrder(ModelArchitectureType.Custom);
        loadOrder.AddLayer("layer1");
        loadOrder.AddLayer("layer2");
        loadOrder.AddLayer("layer3");
        loadOrder.AddDependency("layer3", "layer1");
        loadOrder.AddDependency("layer3", "layer2");

        // Act
        var result = loadOrder.GetLoadOrder();

        // Assert
        Assert.Contains("layer1", result);
        Assert.Contains("layer2", result);
        Assert.Contains("layer3", result);
        Assert.True(result.IndexOf("layer1") < result.IndexOf("layer3"));
        Assert.True(result.IndexOf("layer2") < result.IndexOf("layer3"));
    }

    [Fact]
    public void MemoryManager_RegisterLayer_IncreasesCount()
    {
        // Arrange
        var options = new ProgressiveLoadOptions();
        var memoryManager = new MemoryManager(options);

        // Act
        memoryManager.RegisterLayer("layer1", 1024);

        // Assert
        Assert.Equal(1, memoryManager.LoadedLayerCount);
        Assert.Equal(1024L, memoryManager.TotalLoadedBytes);
    }

    [Fact]
    public void MemoryManager_UnregisterLayer_DecreasesCount()
    {
        // Arrange
        var options = new ProgressiveLoadOptions();
        var memoryManager = new MemoryManager(options);
        memoryManager.RegisterLayer("layer1", 1024);

        // Act
        long freedBytes = memoryManager.UnregisterLayer("layer1");

        // Assert
        Assert.Equal(0, memoryManager.LoadedLayerCount);
        Assert.Equal(0L, memoryManager.TotalLoadedBytes);
        Assert.Equal(1024L, freedBytes);
    }

    [Fact]
    public void MemoryManager_HasMemoryPressure_WithUnloadStrategyNever_ReturnsFalse()
    {
        // Arrange
        var options = new ProgressiveLoadOptions { UnloadStrategy = UnloadStrategy.Never };
        var memoryManager = new MemoryManager(options);
        memoryManager.RegisterLayer("layer1", 1024 * 1024 * 1024); // 1GB

        // Act
        bool hasPressure = memoryManager.HasMemoryPressure();

        // Assert
        Assert.False(hasPressure);
    }

    [Fact]
    public void MemoryManager_HasMemoryPressure_WithMemoryPressureAndHighLoad_ReturnsTrue()
    {
        // Arrange
        var options = new ProgressiveLoadOptions
        {
            UnloadStrategy = UnloadStrategy.MemoryPressure,
            MemoryPressureThreshold = 100 * 1024 * 1024 // 100MB
        };
        var memoryManager = new MemoryManager(options);
        memoryManager.RegisterLayer("layer1", 200 * 1024 * 1024); // 200MB

        // Act
        bool hasPressure = memoryManager.HasMemoryPressure();

        // Assert
        Assert.True(hasPressure);
    }

    [Fact]
    public void MemoryManager_GetLayersToUnload_WithLRU_ReturnsOldest()
    {
        // Arrange
        var options = new ProgressiveLoadOptions
        {
            UnloadStrategy = UnloadStrategy.LRU,
            MaxLoadedLayers = 2
        };
        var memoryManager = new MemoryManager(options);
        memoryManager.RegisterLayer("layer1", 1024);
        memoryManager.RegisterLayer("layer2", 1024);
        memoryManager.RegisterLayer("layer3", 1024);

        // Access layer2 to make it more recent
        memoryManager.RecordAccess("layer2");

        // Act
        var layersToUnload = memoryManager.GetLayersToUnload(1);

        // Assert
        Assert.Single(layersToUnload);
        Assert.Contains("layer1", layersToUnload); // Least recently used
    }

    [Fact]
    public void LazyParameter_Unload_ReleasesMemory()
    {
        // Arrange
        var loader = CreateTestLoader();
        var data = new float[] { 1f, 2f, 3f, 4f };
        var shape = new[] { 2, 2 };
        var weightPath = CreateTempWeightFile(data, shape);
        var param = loader.RegisterParameter("layer1", weightPath, shape);

        // Load the parameter
        _ = param.Data;

        // Act
        param.Unload();

        // Assert
        Assert.False(param.IsLoaded);
        Assert.Equal(0, loader.Context.MemoryManager.LoadedLayerCount);
    }

    [Fact]
    public void ProgressiveModelLoader_ManageMemory_WithPressure_UnloadsLayers()
    {
        // Arrange
        var options = new ProgressiveLoadOptions
        {
            UnloadStrategy = UnloadStrategy.LRU,
            MaxLoadedLayers = 1
        };
        var loader = CreateTestLoader("test_model", options);
        var data = new float[] { 1f, 2f, 3f, 4f };
        var shape = new[] { 2, 2 };

        var path1 = CreateTempWeightFile(data, shape);
        var path2 = CreateTempWeightFile(data, shape);

        loader.RegisterParameter("layer1", path1, shape);
        loader.RegisterParameter("layer2", path2, shape);

        // Load both layers
        loader.PrefetchLayer("layer1");
        loader.PrefetchLayer("layer2");

        // Act
        loader.ManageMemory();

        // Assert
        // One layer should be unloaded (the least recently used)
        var loadedCount = loader.Context.MemoryManager.LoadedLayerCount;
        Assert.True(loadedCount <= 1);
    }

    [Fact]
    public void ProgressiveModelLoader_WaitForFullyLoaded_WithAllLoaded_ReturnsTrue()
    {
        // Arrange
        var loader = CreateTestLoader();
        var data = new float[] { 1f, 2f, 3f, 4f };
        var shape = new[] { 2, 2 };
        var path1 = CreateTempWeightFile(data, shape);
        var path2 = CreateTempWeightFile(data, shape);

        loader.RegisterParameter("layer1", path1, shape);
        loader.RegisterParameter("layer2", path2, shape);

        // Load all layers
        loader.PrefetchLayer("layer1");
        loader.PrefetchLayer("layer2");

        // Act
        bool result = loader.WaitForFullyLoaded(1000);

        // Assert
        Assert.True(result);
        Assert.True(loader.IsFullyLoaded());
    }

    [Fact]
    public void ProgressiveModelLoader_PrefetchNextLayers_WithCurrentLayer_LoadsNextLayers()
    {
        // Arrange
        var options = new ProgressiveLoadOptions
        {
            Strategy = LayerLoadingStrategy.Prefetch,
            PrefetchCount = 2
        };
        var loader = CreateTestLoader("test_model", options);

        // Add layers to load order
        loader.Context.LayerLoadOrder.AddLayer("layer1");
        loader.Context.LayerLoadOrder.AddLayer("layer2");
        loader.Context.LayerLoadOrder.AddLayer("layer3");

        var data = new float[] { 1f, 2f, 3f, 4f };
        var shape = new[] { 2, 2 };

        var path1 = CreateTempWeightFile(data, shape);
        var path2 = CreateTempWeightFile(data, shape);
        var path3 = CreateTempWeightFile(data, shape);

        loader.RegisterParameter("layer1", path1, shape);
        loader.RegisterParameter("layer2", path2, shape);
        loader.RegisterParameter("layer3", path3, shape);

        // Act
        loader.PrefetchNextLayers("layer1");

        // Assert
        Assert.True(loader.GetParameter("layer2")?.IsLoaded);
        Assert.True(loader.GetParameter("layer3")?.IsLoaded);
    }

    [Fact]
    public void LazyParameter_DataAccess_MultipleTimes_OnlyLoadsOnce()
    {
        // Arrange
        var loader = CreateTestLoader();
        var data = new float[] { 1f, 2f, 3f, 4f };
        var shape = new[] { 2, 2 };
        var weightPath = CreateTempWeightFile(data, shape);
        var param = loader.RegisterParameter("layer1", weightPath, shape);

        // Act
        var data1 = param.Data;
        var data2 = param.Data;
        var data3 = param.Data;

        // Assert
        Assert.NotNull(data1);
        Assert.NotNull(data2);
        Assert.NotNull(data3);
        Assert.Same(data1, data2); // Same reference
        Assert.Same(data2, data3);
    }

    [Fact]
    public void ProgressiveLoadContext_IsFullyLoaded_WithNoLayers_ReturnsTrue()
    {
        // Arrange
        var options = new ProgressiveLoadOptions();
        var metadata = new ModelMetadata { ModelName = "test" };
        var device = Device.CreateCpu();
        var loadOrder = LayerLoadOrder.ForCNN();
        var context = new ProgressiveLoadContext(metadata, device, "test.bin", options, loadOrder);

        // Act
        bool isFullyLoaded = context.IsFullyLoaded();

        // Assert
        Assert.True(isFullyLoaded);
    }

    [Fact]
    public void ProgressiveLoadContext_GetNextLayersToLoad_WithSequentialStrategy_ReturnsNextLayers()
    {
        // Arrange
        var options = new ProgressiveLoadOptions
        {
            Strategy = LayerLoadingStrategy.Sequential,
            PrefetchCount = 2
        };
        var metadata = new ModelMetadata { ModelName = "test" };
        var device = Device.CreateCpu();
        var loadOrder = LayerLoadOrder.ForCNN();

        loadOrder.AddLayer("layer1");
        loadOrder.AddLayer("layer2");
        loadOrder.AddLayer("layer3");
        loadOrder.AddLayer("layer4");

        var context = new ProgressiveLoadContext(metadata, device, "test.bin", options, loadOrder);

        // Act
        var nextLayers = context.GetNextLayersToLoad("layer1");

        // Assert
        Assert.Equal(2, nextLayers.Count);
        Assert.Contains("layer2", nextLayers);
        Assert.Contains("layer3", nextLayers);
    }

    [Fact]
    public void ModelZooProgressiveExtensions_LoadProgressive_WorksCorrectly()
    {
        // Arrange
        var modelZoo = new ModelZoo();

        // Act
        var loader = modelZoo.LoadProgressive("test_model", LayerLoadingStrategy.Parallel);

        // Assert
        Assert.NotNull(loader);
        Assert.Equal(LayerLoadingStrategy.Parallel, loader.Context.LoadingStrategy);
    }
}
