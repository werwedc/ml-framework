# Spec: Adapter Management Tests

## Overview
Implement comprehensive unit tests for adapter registry, switching, and merging functionality.

## Test File Structure

### File: `tests/LoRA/AdapterManagementTests.cs`

## Test Categories

### 1. Registry Save/Load Tests
```csharp
[TestClass]
public class AdapterRegistrySaveLoadTests
{
    private const string TestRegistryPath = "./test_adapters";

    [TestInitialize]
    public void Setup()
    {
        if (Directory.Exists(TestRegistryPath))
            Directory.Delete(TestRegistryPath, recursive: true);
    }

    [TestCleanup]
    public void Cleanup()
    {
        if (Directory.Exists(TestRegistryPath))
            Directory.Delete(TestRegistryPath, recursive: true);
    }

    [TestMethod]
    public void SaveAdapter_SavesToDisk()
    {
        var model = CreateTestModel();
        model.ApplyLoRA(rank: 8, alpha: 16);

        var registry = new LoRAAdapterRegistry(TestRegistryPath);
        registry.SetModel(model);

        // Act
        registry.SaveAdapter("test_adapter");

        // Assert
        var adapterPath = Path.Combine(TestRegistryPath, "test_adapter");
        Assert.IsTrue(Directory.Exists(adapterPath));
        Assert.IsTrue(File.Exists(Path.Combine(adapterPath, "metadata.json")));
        Assert.IsTrue(File.Exists(Path.Combine(adapterPath, "weights.bin")));
    }

    [TestMethod]
    public void LoadAdapter_LoadsFromDisk()
    {
        var model1 = CreateTestModel();
        model1.ApplyLoRA(rank: 8, alpha: 16);

        var registry = new LoRAAdapterRegistry(TestRegistryPath);
        registry.SetModel(model1);

        // Save initial adapter
        registry.SaveAdapter("adapter1");

        // Load into new model
        var model2 = CreateTestModel();
        model2.ApplyLoRA(rank: 8, alpha: 16);

        registry.SetModel(model2);

        // Act
        registry.LoadAdapter("adapter1");

        // Assert - weights should be loaded
        var adapters = model2.GetLoRAAdapters().ToList();
        Assert.IsTrue(adapters.Count > 0);
    }

    [TestMethod]
    public void SaveLoadAdapter_PreservesWeights()
    {
        var model1 = CreateTestModel();
        model1.ApplyLoRA(rank: 8, alpha: 16);

        var input = Tensor.Random(new[] { 32, 64 }, new Random(42));
        var outputBefore = model1.Forward(input);

        var registry = new LoRAAdapterRegistry(TestRegistryPath);
        registry.SetModel(model1);
        registry.SaveAdapter("test_adapter");

        // Load into new model
        var model2 = CreateTestModel();
        model2.ApplyLoRA(rank: 8, alpha: 16);
        registry.SetModel(model2);
        registry.LoadAdapter("test_adapter");

        // Act
        var outputAfter = model2.Forward(input);

        // Assert - outputs should be similar
        var diff = Math.Abs(outputBefore - outputAfter).Max().ToScalar<float>();
        Assert.Less(diff, 1e-4f);
    }
}
```

### 2. Metadata Tests
```csharp
[TestClass]
public class AdapterRegistryMetadataTests
{
    private const string TestRegistryPath = "./test_adapters";

    [TestInitialize]
    public void Setup()
    {
        if (Directory.Exists(TestRegistryPath))
            Directory.Delete(TestRegistryPath, recursive: true);
    }

    [TestCleanup]
    public void Cleanup()
    {
        if (Directory.Exists(TestRegistryPath))
            Directory.Delete(TestRegistryPath, recursive: true);
    }

    [TestMethod]
    public void SaveAdapter_WithMetadata_SavesMetadata()
    {
        var model = CreateTestModel();
        model.ApplyLoRA(rank: 8, alpha: 16);

        var metadata = new AdapterMetadata
        {
            Id = "test_adapter",
            Name = "Test Adapter",
            Description = "This is a test adapter",
            TaskType = "classification",
            TrainingSteps = 1000,
            Metrics = new Dictionary<string, float> { { "accuracy", 0.95f } }
        };

        var registry = new LoRAAdapterRegistry(TestRegistryPath);
        registry.SetModel(model);

        // Act
        registry.SaveAdapter("test_adapter", metadata);

        // Assert
        var loadedMetadata = registry.GetAdapterMetadata("test_adapter");
        Assert.IsNotNull(loadedMetadata);
        Assert.AreEqual("Test Adapter", loadedMetadata.Name);
        Assert.AreEqual("classification", loadedMetadata.TaskType);
        Assert.AreEqual(1000, loadedMetadata.TrainingSteps);
    }

    [TestMethod]
    public void GetAdapterMetadata_WithMissingAdapter_ReturnsNull()
    {
        var registry = new LoRAAdapterRegistry(TestRegistryPath);

        // Act
        var metadata = registry.GetAdapterMetadata("nonexistent");

        // Assert
        Assert.IsNull(metadata);
    }
}
```

### 3. List/Delete Tests
```csharp
[TestClass]
public class AdapterRegistryListDeleteTests
{
    private const string TestRegistryPath = "./test_adapters";

    [TestInitialize]
    public void Setup()
    {
        if (Directory.Exists(TestRegistryPath))
            Directory.Delete(TestRegistryPath, recursive: true);
    }

    [TestCleanup]
    public void Cleanup()
    {
        if (Directory.Exists(TestRegistryPath))
            Directory.Delete(TestRegistryPath, recursive: true);
    }

    [TestMethod]
    public void ListAdapters_ReturnsAllAdapters()
    {
        var model = CreateTestModel();
        model.ApplyLoRA(rank: 8, alpha: 16);

        var registry = new LoRAAdapterRegistry(TestRegistryPath);
        registry.SetModel(model);

        registry.SaveAdapter("adapter1");
        registry.SaveAdapter("adapter2");
        registry.SaveAdapter("adapter3");

        // Act
        var adapters = registry.ListAdapters();

        // Assert
        Assert.AreEqual(3, adapters.Count);
        Assert.IsTrue(adapters.Any(a => a.Id == "adapter1"));
        Assert.IsTrue(adapters.Any(a => a.Id == "adapter2"));
        Assert.IsTrue(adapters.Any(a => a.Id == "adapter3"));
    }

    [TestMethod]
    public void DeleteAdapter_RemovesFromDisk()
    {
        var model = CreateTestModel();
        model.ApplyLoRA(rank: 8, alpha: 16);

        var registry = new LoRAAdapterRegistry(TestRegistryPath);
        registry.SetModel(model);
        registry.SaveAdapter("adapter1");

        // Act
        registry.DeleteAdapter("adapter1");

        // Assert
        var adapterPath = Path.Combine(TestRegistryPath, "adapter1");
        Assert.IsFalse(Directory.Exists(adapterPath));

        var adapters = registry.ListAdapters();
        Assert.IsFalse(adapters.Any(a => a.Id == "adapter1"));
    }
}
```

### 4. Adapter Switching Tests
```csharp
[TestClass]
public class AdapterSwitchingTests
{
    private const string TestRegistryPath = "./test_adapters";

    [TestInitialize]
    public void Setup()
    {
        if (Directory.Exists(TestRegistryPath))
            Directory.Delete(TestRegistryPath, recursive: true);
    }

    [TestCleanup]
    public void Cleanup()
    {
        if (Directory.Exists(TestRegistryPath))
            Directory.Delete(TestRegistryPath, recursive: true);
    }

    [TestMethod]
    public void SwitchAdapter_ChangesModelBehavior()
    {
        var model = CreateTestModel();
        model.ApplyLoRA(rank: 8, alpha: 16);

        var registry = new LoRAAdapterRegistry(TestRegistryPath);
        registry.SetModel(model);

        // Save two different adapters
        var input = Tensor.Random(new[] { 32, 64 }, new Random(42));

        registry.SaveAdapter("adapter1");

        // Modify adapter weights
        var adapters = model.GetLoRAAdapters().ToList();
        foreach (var adapter in adapters)
        {
            var (a, b) = adapter.GetAdapterWeights();
            var newA = a!.Clone().Add(0.1f);
            var newB = b!.Clone().Add(0.1f);
            adapter.SetAdapterWeights(newA, newB);
        }

        registry.SaveAdapter("adapter2");

        var switcher = new AdapterSwitcher(model, registry);

        // Act - switch between adapters
        switcher.SwitchAdapter("adapter1");
        var output1 = model.Forward(input);

        switcher.SwitchAdapter("adapter2");
        var output2 = model.Forward(input);

        // Assert - outputs should differ
        var diff = Math.Abs(output1 - output2).Max().ToScalar<float>();
        Assert.IsTrue(diff > 0.0f);
    }

    [TestMethod]
    public void SwitchAdapter_IsFast()
    {
        var model = CreateTestModel();
        model.ApplyLoRA(rank: 8, alpha: 16);

        var registry = new LoRAAdapterRegistry(TestRegistryPath);
        registry.SetModel(model);
        registry.SaveAdapter("adapter1");
        registry.SaveAdapter("adapter2");

        var switcher = new AdapterSwitcher(model, registry);

        // Act
        var time = switcher.SwitchAdapter("adapter2");

        // Assert - switching should be fast (<100ms target)
        Assert.Less(time, 100);
    }

    [TestMethod]
    public void PreloadAdapter_LoadsIntoCache()
    {
        var model = CreateTestModel();
        model.ApplyLoRA(rank: 8, alpha: 16);

        var registry = new LoRAAdapterRegistry(TestRegistryPath);
        registry.SetModel(model);
        registry.SaveAdapter("adapter1");
        registry.SaveAdapter("adapter2");

        var switcher = new AdapterSwitcher(model, registry);

        // Act
        switcher.PreloadAdapter("adapter1");
        var stats = switcher.GetCacheStats();

        // Assert
        Assert.AreEqual(1, stats.CachedAdapters);
        Assert.IsTrue(stats.CachedAdapters > 0);
    }
}
```

### 5. Adapter Merging Tests
```csharp
[TestClass]
public class AdapterMergingTests
{
    [TestMethod]
    public void MergeAllAdapters_MergesAllAdapters()
    {
        var model = CreateTestModel();
        model.ApplyLoRA(rank: 8, alpha: 16);

        var merger = new AdapterMerger(model);

        // Act
        var count = merger.MergeAllAdapters();

        // Assert
        Assert.IsTrue(count > 0);
        var summary = merger.GetMergeSummary();
        Assert.AreEqual(count, summary.MergedAdapters);
        Assert.IsTrue(merger.AreAllAdaptersMerged());
    }

    [TestMethod]
    public void MergeAllAdapters_VerifiesCorrectness()
    {
        var model = CreateTestModel();
        model.ApplyLoRA(rank: 8, alpha: 16);

        var input = Tensor.Random(new[] { 32, 64 }, new Random(42));
        var outputBefore = model.Forward(input);

        var merger = new AdapterMerger(model);

        // Act
        merger.MergeAllAdapters();
        var outputAfter = model.Forward(input);

        // Assert - outputs should be similar
        var diff = Math.Abs(outputBefore - outputAfter).Max().ToScalar<float>();
        Assert.Less(diff, 1e-4f);
    }

    [TestMethod]
    public void ResetAllAdapters_RestoresWeights()
    {
        var model = CreateTestModel();
        model.ApplyLoRA(rank: 8, alpha: 16);

        var input = Tensor.Random(new[] { 32, 64 }, new Random(42));
        var outputBefore = model.Forward(input);

        var merger = new AdapterMerger(model);

        // Merge and reset
        merger.MergeAllAdapters();
        merger.ResetAllAdapters();

        // Act
        var outputAfter = model.Forward(input);

        // Assert - outputs should be similar to original
        var diff = Math.Abs(outputBefore - outputAfter).Max().ToScalar<float>();
        Assert.Less(diff, 1e-4f);
    }

    [TestMethod]
    public void PartialMergeAll_MergesPartially()
    {
        var model = CreateTestModel();
        model.ApplyLoRA(rank: 8, alpha: 16);

        var merger = new AdapterMerger(model);

        // Act - partial merge at 50%
        merger.PartialMergeAll(0.5f);

        // Assert - adapters should still be enabled (not fully merged)
        var adapters = model.GetLoRAAdapters().ToList();
        foreach (var adapter in adapters)
        {
            Assert.IsTrue(adapter.IsEnabled);
        }
    }
}
```

### 6. Adapter Composition Tests
```csharp
[TestClass]
public class AdapterCompositionTests
{
    private const string TestRegistryPath = "./test_adapters";

    [TestInitialize]
    public void Setup()
    {
        if (Directory.Exists(TestRegistryPath))
            Directory.Delete(TestRegistryPath, recursive: true);
    }

    [TestCleanup]
    public void Cleanup()
    {
        if (Directory.Exists(TestRegistryPath))
            Directory.Delete(TestRegistryPath, recursive: true);
    }

    [TestMethod]
    public void AddAdapters_CombinesWeights()
    {
        var model = CreateTestModel();
        model.ApplyLoRA(rank: 8, alpha: 16);

        var registry = new LoRAAdapterRegistry(TestRegistryPath);
        registry.SetModel(model);

        // Save two adapters
        registry.SaveAdapter("adapter1");

        var adapters = model.GetLoRAAdapters().ToList();
        foreach (var adapter in adapters)
        {
            var (a, b) = adapter.GetAdapterWeights();
            adapter.SetAdapterWeights(a!.Clone().Add(0.1f), b!.Clone().Add(0.1f));
        }

        registry.SaveAdapter("adapter2");

        // Act - compose adapters
        var composer = new AdapterComposer(model, registry);
        composer.AddAdapters(new[] { "adapter1", "adapter2" }, outputAdapterId: "composed");

        // Assert - composed adapter should exist
        var metadata = registry.GetAdapterMetadata("composed");
        Assert.IsNotNull(metadata);
    }

    [TestMethod]
    public void InterpolateAdapters_InterpolatesWeights()
    {
        var model = CreateTestModel();
        model.ApplyLoRA(rank: 8, alpha: 16);

        var registry = new LoRAAdapterRegistry(TestRegistryPath);
        registry.SetModel(model);

        registry.SaveAdapter("adapter1");

        var adapters = model.GetLoRAAdapters().ToList();
        foreach (var adapter in adapters)
        {
            var (a, b) = adapter.GetAdapterWeights();
            adapter.SetAdapterWeights(a!.Clone().Add(1.0f), b!.Clone().Add(1.0f));
        }

        registry.SaveAdapter("adapter2");

        // Act - interpolate at 50%
        var composer = new AdapterComposer(model, registry);
        composer.InterpolateAdapters("adapter1", "adapter2", 0.5f, "interpolated");

        // Assert
        var metadata = registry.GetAdapterMetadata("interpolated");
        Assert.IsNotNull(metadata);
    }
}
```

## Success Criteria
- All tests pass consistently
- Adapter save/load works correctly
- Switching is fast and reliable
- Merging produces correct results
- Composition works as expected

## Dependencies
- LoRAAdapterRegistry (from spec 008)
- AdapterSwitcher (from spec 010)
- AdapterMerger (from spec 011)
- AdapterComposer (from spec 012)
- Testing framework (xUnit/NUnit/etc.)

## Estimated Time
60 minutes

## Notes
- Use temporary directories for test isolation
- Test with different adapter configurations
- Verify persistence across registry instances
- Test concurrent operations where applicable
