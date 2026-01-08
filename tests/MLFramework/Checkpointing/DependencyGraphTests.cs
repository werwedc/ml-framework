namespace MLFramework.Checkpointing.Tests;

using System;
using Xunit;

/// <summary>
/// Tests for DependencyGraph
/// </summary>
public class DependencyGraphTests
{
    [Fact]
    public void AddDependency_WithValidData_AddsDependency()
    {
        // Arrange
        var graph = new DependencyGraph();

        // Act
        graph.AddDependency("layer1", new[] { "input" });

        // Assert - no exception thrown
    }

    [Fact]
    public void AddDependency_WithNullLayerId_ThrowsException()
    {
        // Arrange
        var graph = new DependencyGraph();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            graph.AddDependency(null!, new[] { "input" }));
    }

    [Fact]
    public void AddDependency_WithNullDependsOn_ThrowsException()
    {
        // Arrange
        var graph = new DependencyGraph();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            graph.AddDependency("layer1", null!));
    }

    [Fact]
    public void GetTopologicalOrder_WithSimpleDependencies_ReturnsCorrectOrder()
    {
        // Arrange
        var graph = new DependencyGraph();
        graph.AddDependency("layer1", new[] { "input" });
        graph.AddDependency("layer2", new[] { "layer1" });

        // Act
        var order = graph.GetTopologicalOrder(new[] { "layer2", "layer1", "input" });

        // Assert
        Assert.Equal(3, order.Count);
        Assert.Equal("input", order[0]); // input has no dependencies
    }

    [Fact]
    public void GetTopologicalOrder_WithNoDependencies_ReturnsAllLayers()
    {
        // Arrange
        var graph = new DependencyGraph();

        // Act
        var order = graph.GetTopologicalOrder(new[] { "layer1", "layer2", "layer3" });

        // Assert
        Assert.Equal(3, order.Count);
        Assert.Contains("layer1", order);
        Assert.Contains("layer2", order);
        Assert.Contains("layer3", order);
    }

    [Fact]
    public void GetTopologicalOrder_WithEmptyList_ReturnsEmptyList()
    {
        // Arrange
        var graph = new DependencyGraph();

        // Act
        var order = graph.GetTopologicalOrder(Array.Empty<string>());

        // Assert
        Assert.Empty(order);
    }

    [Fact]
    public void GetTopologicalOrder_WithNullList_ThrowsException()
    {
        // Arrange
        var graph = new DependencyGraph();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            graph.GetTopologicalOrder(null!));
    }

    [Fact]
    public void GetTopologicalOrder_WithCycle_ThrowsException()
    {
        // Arrange
        var graph = new DependencyGraph();
        graph.AddDependency("layer1", new[] { "layer2" });
        graph.AddDependency("layer2", new[] { "layer1" }); // Creates cycle

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            graph.GetTopologicalOrder(new[] { "layer1", "layer2" }));
    }

    [Fact]
    public void HasCycle_WithNoDependencies_ReturnsFalse()
    {
        // Arrange
        var graph = new DependencyGraph();

        // Act
        var hasCycle = graph.HasCycle();

        // Assert
        Assert.False(hasCycle);
    }

    [Fact]
    public void HasCycle_WithAcyclicGraph_ReturnsFalse()
    {
        // Arrange
        var graph = new DependencyGraph();
        graph.AddDependency("layer1", new[] { "input" });
        graph.AddDependency("layer2", new[] { "layer1" });

        // Act
        var hasCycle = graph.HasCycle();

        // Assert
        Assert.False(hasCycle);
    }

    [Fact]
    public void HasCycle_WithCycle_ReturnsTrue()
    {
        // Arrange
        var graph = new DependencyGraph();
        graph.AddDependency("layer1", new[] { "layer2" });
        graph.AddDependency("layer2", new[] { "layer1" });

        // Act
        var hasCycle = graph.HasCycle();

        // Assert
        Assert.True(hasCycle);
    }

    [Fact]
    public void Clear_RemovesAllDependencies()
    {
        // Arrange
        var graph = new DependencyGraph();
        graph.AddDependency("layer1", new[] { "input" });

        // Act
        graph.Clear();

        // Assert
        Assert.False(graph.HasCycle()); // Should be acyclic (no dependencies)
    }

    [Fact]
    public void GetTopologicalOrder_WithComplexGraph_RespectsDependencies()
    {
        // Arrange
        var graph = new DependencyGraph();
        graph.AddDependency("layer3", new[] { "layer1", "layer2" });
        graph.AddDependency("layer1", new[] { "input" });

        // Act
        var order = graph.GetTopologicalOrder(new[] { "layer3", "layer1", "layer2", "input" });

        // Assert
        var inputIndex = order.IndexOf("input");
        var layer1Index = order.IndexOf("layer1");
        var layer3Index = order.IndexOf("layer3");

        Assert.True(inputIndex < layer1Index); // input before layer1
        Assert.True(layer1Index < layer3Index); // layer1 before layer3
    }

    [Fact]
    public void GetTopologicalOrder_WithTransitiveDependencies_RespectsAll()
    {
        // Arrange
        var graph = new DependencyGraph();
        graph.AddDependency("layer3", new[] { "layer2" });
        graph.AddDependency("layer2", new[] { "layer1" });
        graph.AddDependency("layer1", new[] { "input" });

        // Act
        var order = graph.GetTopologicalOrder(new[] { "layer3", "layer2", "layer1", "input" });

        // Assert
        Assert.Equal("input", order[0]);
        Assert.Equal("layer1", order[1]);
        Assert.Equal("layer2", order[2]);
        Assert.Equal("layer3", order[3]);
    }

    [Fact]
    public void GetTopologicalOrder_WithMultipleDependencies_RespectsAll()
    {
        // Arrange
        var graph = new DependencyGraph();
        graph.AddDependency("output", new[] { "layer1", "layer2", "layer3" });

        // Act
        var order = graph.GetTopologicalOrder(new[] { "output", "layer1", "layer2", "layer3" });

        // Assert
        var outputIndex = order.IndexOf("output");
        var layer1Index = order.IndexOf("layer1");
        var layer2Index = order.IndexOf("layer2");
        var layer3Index = order.IndexOf("layer3");

        Assert.True(layer1Index < outputIndex);
        Assert.True(layer2Index < outputIndex);
        Assert.True(layer3Index < outputIndex);
    }
}
