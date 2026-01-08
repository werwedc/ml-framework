namespace MachineLearning.Checkpointing.Tests;

using Xunit;

/// <summary>
/// Tests for StateDict and StateUtils
/// </summary>
public class StateDictTests
{
    [Fact]
    public void GetTensor_WithExistingKey_ReturnsTensor()
    {
        // Arrange
        var stateDict = new StateDict();
        var tensor = new MockTensor(new long[] { 10, 20 }, TensorDataType.Float32, 1024);
        stateDict["weight"] = tensor;

        // Act
        var result = stateDict.GetTensor("weight");

        // Assert
        Assert.Equal(tensor, result);
    }

    [Fact]
    public void GetTensor_WithNonExistingKey_ThrowsException()
    {
        // Arrange
        var stateDict = new StateDict();

        // Act & Assert
        Assert.Throws<KeyNotFoundException>(() => stateDict.GetTensor("nonexistent"));
    }

    [Fact]
    public void GetTensorOrNull_WithExistingKey_ReturnsTensor()
    {
        // Arrange
        var stateDict = new StateDict();
        var tensor = new MockTensor(new long[] { 10, 20 }, TensorDataType.Float32, 1024);
        stateDict["weight"] = tensor;

        // Act
        var result = stateDict.GetTensorOrNull("weight");

        // Assert
        Assert.Equal(tensor, result);
    }

    [Fact]
    public void GetTensorOrNull_WithNonExistingKey_ReturnsNull()
    {
        // Arrange
        var stateDict = new StateDict();

        // Act
        var result = stateDict.GetTensorOrNull("nonexistent");

        // Assert
        Assert.Null(result);
    }

    [Fact]
    public void StateUtils_KeysMatch_WithSameKeys_ReturnsTrue()
    {
        // Arrange
        var state1 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10, 20 }, TensorDataType.Float32, 1024),
            ["bias"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 512)
        };
        var state2 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10, 20 }, TensorDataType.Float32, 1024),
            ["bias"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 512)
        };

        // Act
        var result = StateUtils.KeysMatch(state1, state2);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void StateUtils_ShapesMatch_WithMatchingShapes_ReturnsTrue()
    {
        // Arrange
        var state1 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10, 20 }, TensorDataType.Float32, 1024)
        };
        var state2 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10, 20 }, TensorDataType.Float32, 1024)
        };

        // Act
        var result = StateUtils.ShapesMatch(state1, state2);

        // Assert
        Assert.True(result);
    }
}
