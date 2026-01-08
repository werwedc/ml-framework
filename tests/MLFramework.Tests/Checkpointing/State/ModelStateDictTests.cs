namespace MachineLearning.Checkpointing.Tests;

using Xunit;

/// <summary>
/// Tests for ModelStateDict
/// </summary>
public class ModelStateDictTests
{
    [Fact]
    public void Create_WithParameters_CreatesValidState()
    {
        // Act
        var state = ModelStateDict.Create("Transformer", 12);

        // Assert
        Assert.Equal("Transformer", state.ModelType);
        Assert.Equal(12, state.LayerCount);
    }

    [Fact]
    public void GetLayerState_WithExistingLayer_ReturnsCorrectState()
    {
        // Arrange
        var state = new ModelStateDict
        {
            ModelType = "MLP",
            LayerCount = 2
        };

        var weightTensor = new MockTensor(new long[] { 128, 256 }, TensorDataType.Float32, 131072);
        var biasTensor = new MockTensor(new long[] { 256 }, TensorDataType.Float32, 1024);

        state["layer1.weight"] = weightTensor;
        state["layer1.bias"] = biasTensor;
        state["layer2.weight"] = new MockTensor(new long[] { 256, 128 }, TensorDataType.Float32, 131072);

        // Act
        var layerState = state.GetLayerState("layer1");

        // Assert
        Assert.Equal(2, layerState.Count);
        Assert.Contains("weight", layerState.Keys);
        Assert.Contains("bias", layerState.Keys);
        Assert.Equal(weightTensor, layerState["weight"]);
        Assert.Equal(biasTensor, layerState["bias"]);
    }

    [Fact]
    public void SetLayerState_AddsLayerStateCorrectly()
    {
        // Arrange
        var state = new ModelStateDict();
        var layerState = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10, 20 }, TensorDataType.Float32, 800),
            ["bias"] = new MockTensor(new long[] { 20 }, TensorDataType.Float32, 80)
        };

        // Act
        state.SetLayerState("layer1", layerState);

        // Assert
        Assert.True(state.ContainsKey("layer1.weight"));
        Assert.True(state.ContainsKey("layer1.bias"));
    }

    [Fact]
    public void SetLayerState_ReplacesExistingLayerState()
    {
        // Arrange
        var state = new ModelStateDict();
        var oldLayerState = new StateDict
        {
            ["old_param"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };
        var newLayerState = new StateDict
        {
            ["new_param"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };

        // Act
        state.SetLayerState("layer1", oldLayerState);
        state.SetLayerState("layer1", newLayerState);

        // Assert
        Assert.False(state.ContainsKey("layer1.old_param"));
        Assert.True(state.ContainsKey("layer1.new_param"));
    }

    [Fact]
    public void GetLayerNames_WithMultipleLayers_ReturnsAllNames()
    {
        // Arrange
        var state = new ModelStateDict();
        state["layer1.weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40);
        state["layer1.bias"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40);
        state["layer2.weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40);
        state["layer3.weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40);
        state["model_param"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40);

        // Act
        var layerNames = state.GetLayerNames();

        // Assert
        Assert.Equal(3, layerNames.Count());
        Assert.Contains("layer1", layerNames);
        Assert.Contains("layer2", layerNames);
        Assert.Contains("layer3", layerNames);
        Assert.DoesNotContain("model", layerNames);
    }

    [Fact]
    public void GetModelLevelState_WithMixedParameters_ReturnsOnlyModelLevel()
    {
        // Arrange
        var state = new ModelStateDict();
        state["layer1.weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40);
        state["model_param1"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40);
        state["model_param2"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40);

        // Act
        var modelState = state.GetModelLevelState();

        // Assert
        Assert.Equal(2, modelState.Count);
        Assert.Contains("model_param1", modelState.Keys);
        Assert.Contains("model_param2", modelState.Keys);
        Assert.DoesNotContain("layer1.weight", modelState.Keys);
    }

    [Fact]
    public void SetModelLevelState_AddsModelLevelParameters()
    {
        // Arrange
        var state = new ModelStateDict();
        var modelState = new StateDict
        {
            ["param1"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40),
            ["param2"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };

        // Act
        state.SetModelLevelState(modelState);

        // Assert
        Assert.True(state.ContainsKey("param1"));
        Assert.True(state.ContainsKey("param2"));
    }

    [Fact]
    public void GetLayerState_WithEmptyLayerName_ThrowsException()
    {
        // Arrange
        var state = new ModelStateDict();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => state.GetLayerState(""));
    }

    [Fact]
    public void SetLayerState_WithEmptyLayerName_ThrowsException()
    {
        // Arrange
        var state = new ModelStateDict();
        var layerState = new StateDict();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => state.SetLayerState("", layerState));
    }

    [Fact]
    public void SetLayerState_WithNullState_ThrowsException()
    {
        // Arrange
        var state = new ModelStateDict();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => state.SetLayerState("layer1", null!));
    }
}
