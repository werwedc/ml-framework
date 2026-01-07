# Spec: LoRA Unit Tests

## Overview
Comprehensive unit tests for LoRA implementation covering all components.

## Test Structure
- Tests for `LoraConfig`
- Tests for `LoraLinear` layer
- Tests for `LoraAdapter`
- Tests for `AdapterManager`
- Tests for `LoraInjector`
- Tests for `AdapterSerializer`
- Integration tests

## Test Files

### 1. LoraConfigTests.cs
```csharp
[TestFixture]
public class LoraConfigTests
{
    [Test]
    public void DefaultValues_AreCorrect()
    {
        var config = new LoraConfig();
        Assert.That(config.Rank, Is.EqualTo(8));
        Assert.That(config.Alpha, Is.EqualTo(16));
        Assert.That(config.Dropout, Is.EqualTo(0.0f));
    }

    [Test]
    public void Rank_MustBePositive()
    {
        var config = new LoraConfig { Rank = 0 };
        Assert.Throws<ValidationException>(() => config.Validate());
    }

    [Test]
    public void Alpha_MustBePositive()
    {
        var config = new LoraConfig { Alpha = 0 };
        Assert.Throws<ValidationException>(() => config.Validate());
    }

    [Test]
    public void Dropout_MustBeBetween0And1()
    {
        var config = new LoraConfig { Dropout = 1.5f };
        Assert.Throws<ValidationException>(() => config.Validate());
    }

    [Test]
    public void Preset_ForLLaMA_ReturnsCorrectConfig()
    {
        var config = LoraConfig.ForLLaMA();
        Assert.That(config.TargetModules, Does.Contain("q_proj"));
        Assert.That(config.TargetModules, Does.Contain("v_proj"));
    }
}
```

### 2. LoraLinearTests.cs
```csharp
[TestFixture]
public class LoraLinearTests
{
    [Test]
    public void Forward_ProducesCorrectOutputShape()
    {
        var baseLinear = new Linear(inFeatures: 64, outFeatures: 128);
        var loraLayer = new LoraLinear(baseLinear, rank: 8, alpha: 16, dropout: 0.0f);

        var input = new Tensor(new[] { 10, 64 }); // batch=10, in=64
        var output = loraLayer.Forward(input);

        Assert.That(output.Shape, Is.EqualTo(new[] { 10, 128 }));
    }

    [Test]
    public void Forward_WithLoRAOnly_ProducesDifferentOutputThanBase()
    {
        var baseLinear = new Linear(64, 128);
        baseLinear.Weight.Data.RandomNormal(0, 0.02);

        var loraLayer = new LoraLinear(baseLinear, rank: 8, alpha: 16, dropout: 0.0f);

        var input = new Tensor(new[] { 10, 64 });
        input.Data.RandomNormal(0, 1);

        var baseOutput = baseLinear.Forward(input);
        var loraOutput = loraLayer.Forward(input);

        Assert.That(loraOutput, Is.Not.EqualTo(baseOutput));
    }

    [Test]
    public void MergeAndUnmerge_ProduceSameOutputAsBase()
    {
        var baseLinear = new Linear(64, 128);
        baseLinear.Weight.Data.RandomNormal(0, 0.02);

        var loraLayer = new LoraLinear(baseLinear, rank: 8, alpha: 16, dropout: 0.0f);

        var input = new Tensor(new[] { 10, 64 });
        input.Data.RandomNormal(0, 1);

        var outputBeforeMerge = loraLayer.Forward(input);
        loraLayer.Merge();
        var outputAfterMerge = loraLayer.Forward(input);
        loraLayer.Unmerge();
        var outputAfterUnmerge = loraLayer.Forward(input);

        Assert.That(outputAfterMerge, Is.EqualTo(outputBeforeMerge).Within(1e-6));
        Assert.That(outputAfterUnmerge, Is.EqualTo(outputBeforeMerge).Within(1e-6));
    }

    [Test]
    public void TrainableParameters_ReturnsOnlyLoRAMatrices()
    {
        var baseLinear = new Linear(64, 128);
        var loraLayer = new LoraLinear(baseLinear, rank: 8, alpha: 16, dropout: 0.0f);

        var trainableParams = loraLayer.TrainableParameters().ToList();

        Assert.That(trainableParams.Count, Is.EqualTo(2));
        Assert.That(trainableParams[0], Is.EqualTo(loraLayer.LoraA));
        Assert.That(trainableParams[1], Is.EqualTo(loraLayer.LoraB));
    }

    [Test]
    public void Gradients_FlowThroughLoRAMatrices()
    {
        var baseLinear = new Linear(64, 128);
        baseLinear.Weight.Data.RandomNormal(0, 0.02);

        var loraLayer = new LoraLinear(baseLinear, rank: 8, alpha: 16, dropout: 0.0f);

        var input = new Tensor(new[] { 10, 64 });
        input.Data.RandomNormal(0, 1);

        var output = loraLayer.Forward(input);
        var loss = output.Sum();
        loss.Backward();

        Assert.That(loraLayer.LoraA.Grad, Is.Not.Null);
        Assert.That(loraLayer.LoraB.Grad, Is.Not.Null);
        Assert.That(baseLinear.Weight.Grad, Is.Null); // Frozen
    }
}
```

### 3. LoraInjectorTests.cs
```csharp
[TestFixture]
public class LoraInjectorTests
{
    private class MockModel : IModule
    {
        public Linear Layer1 { get; }
        public Linear Layer2 { get; }

        public MockModel()
        {
            Layer1 = new Linear(64, 128) { Name = "q_proj" };
            Layer2 = new Linear(128, 64) { Name = "k_proj" };
        }
    }

    [Test]
    public void Inject_ReplacesTargetLinearLayers()
    {
        var model = new MockModel();
        var config = new LoraConfig { TargetModules = new[] { "q_proj" } };

        LoraInjector.Inject(model, config);

        Assert.That(model.Layer1, Is.InstanceOf<LoraLinear>());
        Assert.That(model.Layer2, Is.Not.InstanceOf<LoraLinear>());
    }

    [Test]
    public void Inject_WithWildcard_ReplacesMatchingLayers()
    {
        var model = new MockModel();
        var config = new LoraConfig { TargetModules = new[] { "*_proj" } };

        LoraInjector.Inject(model, config);

        Assert.That(model.Layer1, Is.InstanceOf<LoraLinear>());
        Assert.That(model.Layer2, Is.InstanceOf<LoraLinear>());
    }

    [Test]
    public void Remove_RestoresOriginalLayers()
    {
        var model = new MockModel();
        var config = new LoraConfig { TargetModules = new[] { "q_proj" } };

        LoraInjector.Inject(model, config);
        Assert.That(model.Layer1, Is.InstanceOf<LoraLinear>());

        LoraInjector.Remove(model);
        Assert.That(model.Layer1, Is.Not.InstanceOf<LoraLinear>());
    }

    [Test]
    public void HasLoRA_ReturnsTrueAfterInjection()
    {
        var model = new MockModel();
        var config = new LoraConfig();

        Assert.That(LoraInjector.HasLoRA(model), Is.False);

        LoraInjector.Inject(model, config);
        Assert.That(LoraInjector.HasLoRA(model), Is.True);
    }
}
```

### 4. AdapterManagerTests.cs
```csharp
[TestFixture]
public class AdapterManagerTests
{
    private MockModel _model;
    private AdapterManager _manager;
    private LoraAdapter _testAdapter;

    [SetUp]
    public void SetUp()
    {
        _model = new MockModel();
        _manager = new AdapterManager(_model);
        _testAdapter = CreateTestAdapter();
    }

    [Test]
    public void LoadAdapter_AddsToLoadedAdapters()
    {
        _manager.LoadAdapter(_testAdapter);

        Assert.That(_manager.ListAdapters(), Does.Contain("test_adapter"));
    }

    [Test]
    public void SetActiveAdapter_SetsSingleActiveAdapter()
    {
        _manager.LoadAdapter(_testAdapter);
        _manager.SetActiveAdapter("test_adapter");

        Assert.That(_manager.ListActiveAdapters(), Is.EqualTo(new[] { "test_adapter" }));
    }

    [Test]
    public void SetActiveAdapter_WithMultiple_SetsAllActive()
    {
        var adapter1 = CreateTestAdapter("adapter1");
        var adapter2 = CreateTestAdapter("adapter2");

        _manager.LoadAdapter(adapter1);
        _manager.LoadAdapter(adapter2);

        _manager.SetActiveAdapter("adapter1", "adapter2");

        Assert.That(_manager.ListActiveAdapters().Count, Is.EqualTo(2));
    }

    [Test]
    public void UnloadAdapter_RemovesFromLoadedAndActive()
    {
        _manager.LoadAdapter(_testAdapter);
        _manager.SetActiveAdapter("test_adapter");

        _manager.UnloadAdapter("test_adapter");

        Assert.That(_manager.ListAdapters(), Does.Not.Contain("test_adapter"));
        Assert.That(_manager.ListActiveAdapters(), Does.Not.Contain("test_adapter"));
    }

    [Test]
    public void ActivateAdapter_AddsToActiveSet()
    {
        var adapter1 = CreateTestAdapter("adapter1");
        var adapter2 = CreateTestAdapter("adapter2");

        _manager.LoadAdapter(adapter1);
        _manager.LoadAdapter(adapter2);

        _manager.SetActiveAdapter("adapter1");
        _manager.ActivateAdapter("adapter2");

        Assert.That(_manager.ListActiveAdapters(), Does.Contain("adapter2"));
        Assert.That(_manager.ListActiveAdapters(), Does.Contain("adapter1"));
    }

    [Test]
    public void DeactivateAdapter_RemovesFromActiveSet()
    {
        var adapter1 = CreateTestAdapter("adapter1");
        var adapter2 = CreateTestAdapter("adapter2");

        _manager.LoadAdapter(adapter1);
        _manager.LoadAdapter(adapter2);

        _manager.SetActiveAdapter("adapter1", "adapter2");
        _manager.DeactivateAdapter("adapter1");

        Assert.That(_manager.ListActiveAdapters(), Is.EqualTo(new[] { "adapter2" }));
    }
}
```

### 5. AdapterSerializerTests.cs
```csharp
[TestFixture]
public class AdapterSerializerTests
{
    [Test]
    public void SaveAndLoad_Binary_PreservesAllData()
    {
        var adapter = CreateTestAdapter();
        var tempPath = Path.GetTempFileName();

        AdapterSerializer.Save(adapter, tempPath);
        var loadedAdapter = AdapterSerializer.Load(tempPath);

        Assert.That(loadedAdapter.Name, Is.EqualTo(adapter.Name));
        Assert.That(loadedAdapter.Config.Rank, Is.EqualTo(adapter.Config.Rank));
        Assert.That(loadedAdapter.Weights.Count, Is.EqualTo(adapter.Weights.Count));
        Assert.That(loadedAdapter.Metadata.CreatedAt, Is.EqualTo(adapter.Metadata.CreatedAt));

        File.Delete(tempPath);
    }

    [Test]
    public void SaveAndLoad_Json_PreservesAllData()
    {
        var adapter = CreateTestAdapter();
        var tempPath = Path.GetTempFileName();

        AdapterSerializer.SaveJson(adapter, tempPath);
        var loadedAdapter = AdapterSerializer.LoadJson(tempPath);

        Assert.That(loadedAdapter.Name, Is.EqualTo(adapter.Name));
        Assert.That(loadedAdapter.Config.Rank, Is.EqualTo(adapter.Config.Rank));

        File.Delete(tempPath);
    }

    [Test]
    public void Load_InvalidFormat_ThrowsException()
    {
        var tempPath = Path.GetTempFileName();
        File.WriteAllBytes(tempPath, new byte[] { 0x00, 0x01, 0x02, 0x03 });

        Assert.Throws<InvalidDataException>(() => AdapterSerializer.Load(tempPath));

        File.Delete(tempPath);
    }
}
```

### 6. IntegrationTests.cs
```csharp
[TestFixture]
public class LoRAIntegrationTests
{
    [Test]
    public void EndToEnd_TrainWithLoRA()
    {
        // Create model with LoRA
        var model = new SimpleTransformer();
        var config = new LoraConfig { Rank = 4, Alpha = 8 };
        model.ApplyLoRA(config);

        // Create optimizer with only LoRA parameters
        var loraParams = model.GetLoRAParameters();
        var optimizer = new Adam(loraParams);

        // Training step
        var input = new Tensor(new[] { 10, 32 });
        input.Data.RandomNormal(0, 1);

        var output = model.Forward(input);
        var loss = output.Sum();
        loss.Backward();
        optimizer.Step();

        // Verify gradients
        foreach (var param in loraParams)
        {
            Assert.That(param.Grad, Is.Not.Null);
        }
    }

    [Test]
    public void EndToEnd_SaveLoadAdapter()
    {
        var model = new SimpleTransformer();
        model.ApplyLoRA(new LoraConfig { Rank = 4 });

        // Train model (simplified)
        // ...

        // Save adapter
        var manager = new AdapterManager(model);
        manager.SaveAdapter("test", "test_adapter.lora");

        // Load adapter onto fresh model
        var newModel = new SimpleTransformer();
        newModel.ApplyLoRA(new LoraConfig { Rank = 4 });
        var newManager = new AdapterManager(newModel);
        newManager.LoadAdapter("test", "test_adapter.lora");

        // Verify outputs match
        var input = new Tensor(new[] { 1, 32 });
        var output1 = model.Forward(input);
        var output2 = newModel.Forward(input);

        Assert.That(output2, Is.EqualTo(output1).Within(1e-6));

        File.Delete("test_adapter.lora");
    }
}
```

## Test Coverage Requirements
- Unit test coverage > 90%
- Integration tests for key workflows
- Edge cases: empty models, invalid configs, missing files
- Performance tests: large adapters, many adapters

## Deliverables
- `tests/Core/LoRA/LoraConfigTests.cs`
- `tests/Core/LoRA/LoraLinearTests.cs`
- `tests/Core/LoRA/LoraAdapterTests.cs`
- `tests/Core/LoRA/AdapterManagerTests.cs`
- `tests/Core/LoRA/LoraInjectorTests.cs`
- `tests/Core/LoRA/AdapterSerializerTests.cs`
- `tests/Core/LoRA/IntegrationTests.cs`
