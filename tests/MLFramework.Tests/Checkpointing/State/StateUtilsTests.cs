namespace MachineLearning.Checkpointing.Tests;

using Xunit;

/// <summary>
/// Tests for StateUtils
/// </summary>
public class StateUtilsTests
{
    [Fact]
    public void KeysMatch_WithMatchingKeys_ReturnsTrue()
    {
        // Arrange
        var state1 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10, 20 }, TensorDataType.Float32, 800),
            ["bias"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };
        var state2 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10, 20 }, TensorDataType.Float32, 800),
            ["bias"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };

        // Act
        var result = StateUtils.KeysMatch(state1, state2);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void KeysMatch_WithDifferentKeys_ReturnsFalse()
    {
        // Arrange
        var state1 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };
        var state2 = new StateDict
        {
            ["bias"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };

        // Act
        var result = StateUtils.KeysMatch(state1, state2);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void KeysMatch_WithNullStates_ReturnsFalse()
    {
        // Act
        var result1 = StateUtils.KeysMatch(null!, new StateDict());
        var result2 = StateUtils.KeysMatch(new StateDict(), null!);
        var result3 = StateUtils.KeysMatch(null!, null!);

        // Assert
        Assert.False(result1);
        Assert.False(result2);
        Assert.False(result3);
    }

    [Fact]
    public void GetMissingKeys_WithMissingKeys_ReturnsCorrectKeys()
    {
        // Arrange
        var state1 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40),
            ["bias"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };
        var state2 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };

        // Act
        var missing = StateUtils.GetMissingKeys(state1, state2);

        // Assert
        Assert.Single(missing);
        Assert.Contains("bias", missing);
    }

    [Fact]
    public void GetUnexpectedKeys_WithExtraKeys_ReturnsCorrectKeys()
    {
        // Arrange
        var state1 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };
        var state2 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40),
            ["extra_param"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };

        // Act
        var unexpected = StateUtils.GetUnexpectedKeys(state1, state2);

        // Assert
        Assert.Single(unexpected);
        Assert.Contains("extra_param", unexpected);
    }

    [Fact]
    public void ShapesMatch_WithMatchingShapes_ReturnsTrue()
    {
        // Arrange
        var state1 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10, 20 }, TensorDataType.Float32, 800)
        };
        var state2 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10, 20 }, TensorDataType.Float32, 800)
        };

        // Act
        var result = StateUtils.ShapesMatch(state1, state2);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void ShapesMatch_WithDifferentShapes_ReturnsFalse()
    {
        // Arrange
        var state1 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10, 20 }, TensorDataType.Float32, 800)
        };
        var state2 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10, 30 }, TensorDataType.Float32, 1200)
        };

        // Act
        var result = StateUtils.ShapesMatch(state1, state2);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void DataTypesMatch_WithMatchingTypes_ReturnsTrue()
    {
        // Arrange
        var state1 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };
        var state2 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };

        // Act
        var result = StateUtils.DataTypesMatch(state1, state2);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void DataTypesMatch_WithDifferentTypes_ReturnsFalse()
    {
        // Arrange
        var state1 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };
        var state2 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float64, 80)
        };

        // Act
        var result = StateUtils.DataTypesMatch(state1, state2);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void GetTotalSize_WithMultipleTensors_ReturnsCorrectTotal()
    {
        // Arrange
        var state = new StateDict
        {
            ["tensor1"] = new MockTensor(new long[] { 10, 20 }, TensorDataType.Float32, 800),
            ["tensor2"] = new MockTensor(new long[] { 30 }, TensorDataType.Float32, 120)
        };

        // Act
        var totalSize = StateUtils.GetTotalSize(state);

        // Assert
        Assert.Equal(200 + 30, totalSize); // 10*20 + 30 = 200 + 30 = 230
    }

    [Fact]
    public void GetTotalSizeInBytes_WithMultipleTensors_ReturnsCorrectTotal()
    {
        // Arrange
        var state = new StateDict
        {
            ["tensor1"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40),
            ["tensor2"] = new MockTensor(new long[] { 20 }, TensorDataType.Float32, 80)
        };

        // Act
        var totalBytes = StateUtils.GetTotalSizeInBytes(state);

        // Assert
        Assert.Equal(120, totalBytes); // 40 + 80 = 120
    }

    [Fact]
    public void Clone_WithStateDict_ReturnsShallowCopy()
    {
        // Arrange
        var tensor = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40);
        var state = new StateDict { ["weight"] = tensor };

        // Act
        var cloned = StateUtils.Clone(state);

        // Assert
        Assert.NotSame(state, cloned);
        Assert.Equal(tensor, cloned["weight"]); // Shallow copy: same tensor reference
    }

    [Fact]
    public void GetSummary_WithStateDict_ReturnsFormattedSummary()
    {
        // Arrange
        var state = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10, 20 }, TensorDataType.Float32, 800),
            ["bias"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };

        // Act
        var summary = StateUtils.GetSummary(state);

        // Assert
        Assert.Contains("StateDict: 2 tensors", summary);
        Assert.Contains("weight", summary);
        Assert.Contains("bias", summary);
    }

    [Fact]
    public void Filter_WithPredicate_ReturnsFilteredState()
    {
        // Arrange
        var state = new StateDict
        {
            ["weight1"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40),
            ["weight2"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40),
            ["bias"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };

        // Act
        var filtered = StateUtils.Filter(state, (key, _) => key.StartsWith("weight"));

        // Assert
        Assert.Equal(2, filtered.Count);
        Assert.Contains("weight1", filtered.Keys);
        Assert.Contains("weight2", filtered.Keys);
        Assert.DoesNotContain("bias", filtered.Keys);
    }

    [Fact]
    public void FilterByPrefix_WithPrefix_ReturnsFilteredState()
    {
        // Arrange
        var state = new StateDict
        {
            ["layer1.weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40),
            ["layer1.bias"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40),
            ["layer2.weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };

        // Act
        var filtered = StateUtils.FilterByPrefix(state, "layer1.");

        // Assert
        Assert.Equal(2, filtered.Count);
        Assert.Contains("layer1.weight", filtered.Keys);
        Assert.Contains("layer1.bias", filtered.Keys);
    }

    [Fact]
    public void RemovePrefix_WithPrefix_RemovesPrefixFromKeys()
    {
        // Arrange
        var state = new StateDict
        {
            ["layer1.weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40),
            ["layer1.bias"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40),
            ["layer2.weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };

        // Act
        var result = StateUtils.RemovePrefix(state, "layer1.");

        // Assert
        Assert.Equal(3, result.Count);
        Assert.Contains("weight", result.Keys);
        Assert.Contains("bias", result.Keys);
        Assert.Contains("layer2.weight", result.Keys);
    }

    [Fact]
    public void AddPrefix_WithPrefix_AddsPrefixToKeys()
    {
        // Arrange
        var state = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40),
            ["bias"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };

        // Act
        var result = StateUtils.AddPrefix(state, "layer1.");

        // Assert
        Assert.Equal(2, result.Count);
        Assert.Contains("layer1.weight", result.Keys);
        Assert.Contains("layer1.bias", result.Keys);
    }
}
