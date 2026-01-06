# Spec: Model Conversion API Tests

## Overview
Implement comprehensive unit tests for the model conversion API and LoRA injection functionality.

## Test File Structure

### File: `tests/LoRA/LoRAInjectorTests.cs`

## Test Categories

### 1. Injection Tests
```csharp
[TestClass]
public class LoRAInjectorInjectionTests
{
    [TestMethod]
    public void ApplyLoRA_WithSimpleModel_InjectsAdapters()
    {
        // Arrange - create simple model with linear layers
        var model = new SimpleModel();
        var config = new LoRAConfig(rank: 8, alpha: 16);

        // Act
        var injector = new LoRAInjector(config);
        var count = injector.ApplyLoRA(model);

        // Assert
        Assert.IsTrue(count > 0);
        Assert.AreEqual(count, injector.InjectedAdapters.Count);
    }

    [TestMethod]
    public void ApplyLoRA_WithComplexModel_InjectsCorrectly()
    {
        // Arrange - create hierarchical model
        var model = CreateComplexModel();
        var config = new LoRAConfig(rank: 8, alpha: 16);

        // Act
        var injector = new LoRAInjector(config);
        var count = injector.ApplyLoRA(model);

        // Assert - all linear layers should have adapters
        var allAdapters = model.GetLoRAAdapters().ToList();
        Assert.IsTrue(allAdapters.Count >= count);
    }

    [TestMethod]
    public void ApplyLoRA_WithTargetModules_InjectsSelectively()
    {
        var model = CreateModelWithNamedLayers();
        var config = new LoRAConfig(rank: 8, alpha: 16)
        {
            TargetModules = new[] { "layer1", "layer3" }
        };

        // Act
        var injector = new LoRAInjector(config);
        var count = injector.ApplyLoRA(model);

        // Assert - should only inject into target modules
        var adapters = model.GetLoRAAdapters().ToList();
        Assert.AreEqual(2, adapters.Count);
    }

    [TestMethod]
    public void ApplyLoRA_WithTargetLayerTypes_InjectsSelectively()
    {
        var model = CreateModelWithMixedLayers();
        var config = new LoRAConfig(rank: 8, alpha: 16)
        {
            TargetLayerTypes = new[] { "Linear" }
        };

        // Act
        var injector = new LoRAInjector(config);
        injector.ApplyLoRA(model);

        // Assert - only linear layers should have adapters
        var adapters = model.GetLoRAAdapters().ToList();
        foreach (var adapter in adapters)
        {
            Assert.IsInstanceOfType(adapter.BaseLayer, typeof(Linear));
        }
    }

    [TestMethod]
    public void ApplyLoRA_OnAlreadyWrappedModel_DoesNotDoubleWrap()
    {
        var model = new SimpleModel();
        var config1 = new LoRAConfig(rank: 8, alpha: 16);

        var injector1 = new LoRAInjector(config1);
        injector1.ApplyLoRA(model);

        var countBefore = model.GetLoRAAdapters().Count();

        // Act - try to inject again
        var config2 = new LoRAConfig(rank: 4, alpha: 8);
        var injector2 = new LoRAInjector(config2);
        injector2.ApplyLoRA(model);

        var countAfter = model.GetLoRAAdapters().Count();

        // Assert - no new adapters should be created
        Assert.AreEqual(countBefore, countAfter);
    }
}
```

### 2. Configuration Tests
```csharp
[TestClass]
public class LoRAInjectorConfigurationTests
{
    [TestMethod]
    public void ApplyLoRA_WithDifferentRank_UsesCorrectRank()
    {
        var model = new SimpleModel();
        var config = new LoRAConfig(rank: 16, alpha: 32);

        // Act
        var injector = new LoRAInjector(config);
        injector.ApplyLoRA(model);

        // Assert
        var adapters = model.GetLoRAAdapters().ToList();
        foreach (var adapter in adapters)
        {
            Assert.AreEqual(16, adapter.Rank);
        }
    }

    [TestMethod]
    public void ApplyLoRA_WithDifferentAlpha_UsesCorrectScalingFactor()
    {
        var model = new SimpleModel();
        var config = new LoRAConfig(rank: 8, alpha: 24);

        // Act
        var injector = new LoRAInjector(config);
        injector.ApplyLoRA(model);

        // Assert - scaling factor should be alpha/rank = 24/8 = 3.0
        var adapters = model.GetLoRAAdapters().ToList();
        foreach (var adapter in adapters)
        {
            Assert.AreEqual(3.0f, adapter.ScalingFactor, 1e-6f);
        }
    }

    [TestMethod]
    public void ApplyLoRA_WithDropout_IncludesDropout()
    {
        var model = new SimpleModel();
        var config = new LoRAConfig(rank: 8, alpha: 16)
        {
            Dropout = 0.5f
        };

        // Act
        var injector = new LoRAInjector(config);
        injector.ApplyLoRA(model);

        // Assert
        var adapters = model.GetLoRAAdapters().ToList();
        Assert.IsTrue(adapters.Count > 0);
    }

    [TestMethod]
    public void ApplyLoRA_WithDifferentInitialization_InitializesCorrectly()
    {
        var model1 = new SimpleModel();
        var model2 = new SimpleModel();

        var config1 = new LoRAConfig(rank: 8, alpha: 16)
        {
            Initialization = LoRAInitializationStrategy.Standard
        };

        var config2 = new LoRAConfig(rank: 8, alpha: 16)
        {
            Initialization = LoRAInitializationStrategy.Zero
        };

        // Act
        var injector1 = new LoRAInjector(config1);
        var injector2 = new LoRAInjector(config2);

        injector1.ApplyLoRA(model1);
        injector2.ApplyLoRA(model2);

        // Assert - zero initialization should have zero weights
        var adapters2 = model2.GetLoRAAdapters().OfType<LoRALinear>().ToList();
        foreach (var adapter in adapters2)
        {
            var (a, b) = adapter.GetAdapterWeights();
            var aSum = a!.Sum().ToScalar<float>();
            var bSum = b!.Sum().ToScalar<float>();
            Assert.AreEqual(0.0f, aSum, 1e-6f);
            Assert.AreEqual(0.0f, bSum, 1e-6f);
        }
    }
}
```

### 3. Parameter Management Tests
```csharp
[TestClass]
public class LoRAInjectorParameterTests
{
    [TestMethod]
    public void GetTrainableParameters_WithFrozenBase_ReturnsOnlyAdapters()
    {
        var model = new SimpleModel();
        var config = new LoRAConfig(rank: 8, alpha: 16);

        var injector = new LoRAInjector(config);
        injector.ApplyLoRA(model);
        injector.FreezeAllBaseLayers();

        // Act
        var trainableParams = injector.GetTrainableParameters().ToList();

        // Assert - should only have adapter parameters
        Assert.IsTrue(trainableParams.Count > 0);
        foreach (var param in trainableParams)
        {
            Assert.IsTrue(param.RequiresGrad);
        }
    }

    [TestMethod]
    public void GetFrozenParameters_WithFrozenBase_ReturnsBaseParams()
    {
        var model = new SimpleModel();
        var config = new LoRAConfig(rank: 8, alpha: 16);

        var injector = new LoRAInjector(config);
        injector.ApplyLoRA(model);
        injector.FreezeAllBaseLayers();

        // Act
        var frozenParams = injector.GetFrozenParameters().ToList();

        // Assert - should have base layer parameters
        Assert.IsTrue(frozenParams.Count > 0);
    }

    [TestMethod]
    public void FreezeAllBaseLayers_FreezesAllBaseLayers()
    {
        var model = new SimpleModel();
        var config = new LoRAConfig(rank: 8, alpha: 16);

        var injector = new LoRAInjector(config);
        injector.ApplyLoRA(model);

        // Act
        injector.FreezeAllBaseLayers();

        // Assert
        var adapters = model.GetLoRAAdapters().ToList();
        var paramManager = new ParameterManager(model);
        var stats = paramManager.GetParameterStats();

        Assert.IsTrue(stats.TrainableParameters < stats.TotalParameters);
        Assert.IsTrue(stats.ReductionPercentage > 80.0); // >80% reduction
    }

    [TestMethod]
    public void UnfreezeAllBaseLayers_UnfreezesAllBaseLayers()
    {
        var model = new SimpleModel();
        var config = new LoRAConfig(rank: 8, alpha: 16);

        var injector = new LoRAInjector(config);
        injector.ApplyLoRA(model);
        injector.FreezeAllBaseLayers();

        // Act
        injector.UnfreezeAllBaseLayers();

        // Assert
        var paramManager = new ParameterManager(model);
        var stats = paramManager.GetParameterStats();

        Assert.AreEqual(stats.TrainableParameters, stats.TotalParameters);
    }
}
```

### 4. Adapter Control Tests
```csharp
[TestClass]
public class LoRAInjectorAdapterControlTests
{
    [TestMethod]
    public void EnableAllAdapters_EnableAllAdapters()
    {
        var model = new SimpleModel();
        var config = new LoRAConfig(rank: 8, alpha: 16);

        var injector = new LoRAInjector(config);
        injector.ApplyLoRA(model);

        // Disable all first
        injector.DisableAllAdapters();

        // Act
        injector.EnableAllAdapters();

        // Assert
        var adapters = model.GetLoRAAdapters().ToList();
        foreach (var adapter in adapters)
        {
            Assert.IsTrue(adapter.IsEnabled);
        }
    }

    [TestMethod]
    public void DisableAllAdapters_DisablesAllAdapters()
    {
        var model = new SimpleModel();
        var config = new LoRAConfig(rank: 8, alpha: 16);

        var injector = new LoRAInjector(config);
        injector.ApplyLoRA(model);

        // Act
        injector.DisableAllAdapters();

        // Assert
        var adapters = model.GetLoRAAdapters().ToList();
        foreach (var adapter in adapters)
        {
            Assert.IsFalse(adapter.IsEnabled);
        }
    }

    [TestMethod]
    public void InjectedAdapters_ReturnsCorrectList()
    {
        var model = new SimpleModel();
        var config = new LoRAConfig(rank: 8, alpha: 16);

        var injector = new LoRAInjector(config);
        var count = injector.ApplyLoRA(model);

        // Act
        var injectedAdapters = injector.InjectedAdapters;

        // Assert
        Assert.AreEqual(count, injectedAdapters.Count);
    }
}
```

### 5. Extension Method Tests
```csharp
[TestClass]
public class LoRAExtensionMethodTests
{
    [TestMethod]
    public void ApplyLoRA_ExtensionMethod_WorksCorrectly()
    {
        var model = new SimpleModel();

        // Act
        var injector = model.ApplyLoRA(rank: 8, alpha: 16);

        // Assert
        Assert.IsNotNull(injector);
        Assert.IsTrue(injector.InjectedAdapters.Count > 0);
    }

    [TestMethod]
    public void ApplyLoRAToModules_ExtensionMethod_WorksCorrectly()
    {
        var model = CreateModelWithNamedLayers();

        // Act
        var injector = model.ApplyLoRAToModules(
            rank: 8,
            alpha: 16,
            targetModules: new[] { "layer1", "layer2" }
        );

        // Assert
        Assert.AreEqual(2, injector.InjectedAdapters.Count);
    }

    [TestMethod]
    public void GetLoRAAdapters_ExtensionMethod_ReturnsAllAdapters()
    {
        var model = new SimpleModel();
        model.ApplyLoRA(rank: 8, alpha: 16);

        // Act
        var adapters = model.GetLoRAAdapters().ToList();

        // Assert
        Assert.IsTrue(adapters.Count > 0);
    }

    [TestMethod]
    public void FreezeLoRABaseLayers_ExtensionMethod_FreezesBaseLayers()
    {
        var model = new SimpleModel();
        model.ApplyLoRA(rank: 8, alpha: 16);

        // Act
        model.FreezeLoRABaseLayers();

        // Assert
        var paramManager = new ParameterManager(model);
        var stats = paramManager.GetParameterStats();

        Assert.IsTrue(stats.TrainableParameters < stats.TotalParameters);
    }
}
```

### 6. Model Structure Tests
```csharp
[TestClass]
public class LoRAInjectorModelStructureTests
{
    [TestMethod]
    public void ApplyLoRA_PreservesModelStructure()
    {
        var model = CreateComplexModel();
        var config = new LoRAConfig(rank: 8, alpha: 16);

        var injector = new LoRAInjector(config);
        injector.ApplyLoRA(model);

        // Assert - model should still have same hierarchy
        Assert.IsNotNull(model);
        // Additional structure validation depends on model implementation
    }

    [TestMethod]
    public void ApplyLoRA_MaintainsInputOutputCompatibility()
    {
        var model = new SimpleModel();
        var config = new LoRAConfig(rank: 8, alpha: 16);

        var input = Tensor.Random(new[] { 32, 64 }, new Random(42));
        var originalOutput = model.Forward(input);

        var injector = new LoRAInjector(config);
        injector.ApplyLoRA(model);

        // Act
        var newOutput = model.Forward(input);

        // Assert - shapes should match
        Assert.AreEqual(originalOutput.Shape, newOutput.Shape);
    }
}
```

## Helper Methods
```csharp
private class SimpleModel : IModule, IHasSubmodules
{
    private readonly Linear _layer1;
    private readonly Linear _layer2;
    private readonly Linear _layer3;

    public SimpleModel()
    {
        _layer1 = new Linear(64, 128);
        _layer2 = new Linear(128, 256);
        _layer3 = new Linear(256, 128);
    }

    public ITensor Forward(ITensor input)
    {
        var x = _layer1.Forward(input);
        x = x.Relu();
        x = _layer2.Forward(x);
        x = x.Relu();
        x = _layer3.Forward(x);
        return x;
    }

    public IEnumerable<(string Name, IModule Module)> NamedChildren()
    {
        yield return ("layer1", _layer1);
        yield return ("layer2", _layer2);
        yield return ("layer3", _layer3);
    }

    public void SetModule(string name, IModule module)
    {
        switch (name)
        {
            case "layer1": _layer1 = (Linear)module; break;
            case "layer2": _layer2 = (Linear)module; break;
            case "layer3": _layer3 = (Linear)module; break;
        }
    }
}
```

## Success Criteria
- All tests pass consistently
- LoRA injection works for various model structures
- Parameter management functions correctly
- Extension methods work as expected
- Model structure is preserved

## Dependencies
- LoRAInjector implementation (from spec 005)
- LoRAConfig (from spec 001)
- ParameterManager (from spec 006)
- Testing framework (xUnit/NUnit/etc.)

## Estimated Time
60 minutes

## Notes
- Test with different model architectures
- Ensure module hierarchy is preserved
- Validate that existing functionality is not broken
- Consider adding integration tests with real models
