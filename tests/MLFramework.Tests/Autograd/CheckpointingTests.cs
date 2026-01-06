using NUnit.Framework;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;

namespace MLFramework.Tests.Autograd;

/// <summary>
/// Unit tests for gradient checkpointing functionality.
/// </summary>
[TestFixture]
public class CheckpointingTests
{
    [SetUp]
    public void Setup()
    {
        // Clear any existing checkpoint marks before each test
        TensorCheckpointExtensions.ClearAllCheckpointMarks();
    }

    [TearDown]
    public void TearDown()
    {
        // Clean up after each test
        TensorCheckpointExtensions.ClearAllCheckpointMarks();
    }

    #region CheckpointScope Tests

    [Test]
    public void CheckpointScope_Creation_InitializesCorrectly()
    {
        // Arrange & Act
        using var scope = new CheckpointScope("test_scope", true);

        // Assert
        Assert.AreEqual("test_scope", scope.Name);
        Assert.IsTrue(scope.IsEnabled);
        Assert.IsTrue(scope.UseRecomputation);
        Assert.IsNotNull(scope.CheckpointedNodes);
        Assert.AreEqual(0, scope.CheckpointedNodes.Count);
    }

    [Test]
    public void CheckpointScope_CreationDisabled_DisabledByDefault()
    {
        // Arrange & Act
        using var scope = new CheckpointScope("disabled_scope", false);

        // Assert
        Assert.IsFalse(scope.IsEnabled);
    }

    [Test]
    public void CheckpointScope_GetActiveScope_ReturnsActiveScope()
    {
        // Arrange
        using var scope = new CheckpointScope("active_scope");

        // Act
        var retrieved = CheckpointScope.GetActiveScope("active_scope");

        // Assert
        Assert.IsNotNull(retrieved);
        Assert.AreEqual("active_scope", retrieved.Name);
    }

    [Test]
    public void CheckpointScope_Dispose_DisablesScope()
    {
        // Arrange
        var scope = new CheckpointScope("dispose_test");

        // Act
        scope.Dispose();

        // Assert
        var retrieved = CheckpointScope.GetActiveScope("dispose_test");
        Assert.IsNull(retrieved);
    }

    [Test]
    public void CheckpointScope_GetMemoryUsage_ReturnsCorrectValue()
    {
        // Arrange
        using var scope = new CheckpointScope("memory_test");
        var tensor = new Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 4 });
        var operation = new OperationContext("test", gradOutput => new Tensor[] { gradOutput });
        var node = new CheckpointNode(tensor, operation, scope);
        node.SaveActivations(tensor);

        // Act
        var memoryMB = scope.GetMemoryUsageMB();

        // Assert
        Assert.Greater(memoryMB, 0);
    }

    #endregion

    #region CheckpointNode Tests

    [Test]
    public void CheckpointNode_Creation_InitializesCorrectly()
    {
        // Arrange
        var tensor = new Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 4 });
        var operation = new OperationContext("test", gradOutput => new Tensor[] { gradOutput });

        // Act
        var node = new CheckpointNode(tensor, operation);

        // Assert
        Assert.IsNotNull(node);
        Assert.IsFalse(node.IsCheckpoint);
        Assert.IsNotNull(node.SavedActivations);
        Assert.IsNotNull(node.RecomputeFunctions);
        Assert.IsFalse(node.HasRecomputed);
    }

    [Test]
    public void CheckpointNode_WithScope_SetsCheckpointProperty()
    {
        // Arrange
        using var scope = new CheckpointScope("scope_test", true);
        var tensor = new Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 4 });
        var operation = new OperationContext("test", gradOutput => new Tensor[] { gradOutput });

        // Act
        var node = new CheckpointNode(tensor, operation, scope);

        // Assert
        Assert.IsTrue(node.IsCheckpoint);
        Assert.AreEqual(scope, node.Scope);
    }

    [Test]
    public void CheckpointNode_SaveActivations_SavesTensors()
    {
        // Arrange
        var tensor1 = new Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 4 });
        var tensor2 = new Tensor(new float[] { 5, 6, 7, 8 }, new int[] { 4 });
        var operation = new OperationContext("test", gradOutput => new Tensor[] { gradOutput });
        var node = new CheckpointNode(tensor1, operation);

        // Act
        node.SaveActivations(tensor1, tensor2);

        // Assert
        Assert.AreEqual(2, node.SavedActivationCount);
        Assert.IsTrue(node.HasSavedActivations);
    }

    [Test]
    public void CheckpointNode_AddRecomputeFunction_AddsFunction()
    {
        // Arrange
        var tensor = new Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 4 });
        var operation = new OperationContext("test", gradOutput => new Tensor[] { gradOutput });
        var node = new CheckpointNode(tensor, operation);

        // Act
        node.AddRecomputeFunction(() => new Tensor[] { tensor });

        // Assert
        Assert.IsTrue(node.HasRecomputeFunction);
    }

    [Test]
    public void CheckpointNode_Recompute_ExecutesRecomputeFunction()
    {
        // Arrange
        var tensor = new Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 4 });
        var operation = new OperationContext("test", gradOutput => new Tensor[] { gradOutput });
        var node = new CheckpointNode(tensor, operation);
        node.AddRecomputeFunction(() => new Tensor[] { tensor.Clone() });

        // Act
        var recomputed = node.Recompute();

        // Assert
        Assert.AreEqual(1, recomputed.Length);
        Assert.IsTrue(node.HasRecomputed);
    }

    [Test]
    public void CheckpointNode_Recompute_NoFunction_ThrowsException()
    {
        // Arrange
        var tensor = new Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 4 });
        var operation = new OperationContext("test", gradOutput => new Tensor[] { gradOutput });
        var node = new CheckpointNode(tensor, operation);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => node.Recompute());
    }

    [Test]
    public void CheckpointNode_ClearSavedActivations_ClearsAll()
    {
        // Arrange
        var tensor = new Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 4 });
        var operation = new OperationContext("test", gradOutput => new Tensor[] { gradOutput });
        var node = new CheckpointNode(tensor, operation);
        node.SaveActivations(tensor);

        // Act
        node.ClearSavedActivations();

        // Assert
        Assert.AreEqual(0, node.SavedActivationCount);
        Assert.IsFalse(node.HasSavedActivations);
    }

    [Test]
    public void CheckpointNode_GetSavedMemorySize_ReturnsCorrectSize()
    {
        // Arrange
        var tensor = new Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 4 });
        var operation = new OperationContext("test", gradOutput => new Tensor[] { gradOutput });
        var node = new CheckpointNode(tensor, operation);
        node.SaveActivations(tensor);

        // Act
        var size = node.GetSavedMemorySize();

        // Assert
        Assert.AreEqual(4 * sizeof(float), size);
    }

    #endregion

    #region CheckpointManager Tests

    [Test]
    public void CheckpointManager_Creation_InitializesCorrectly()
    {
        // Act
        var manager = new CheckpointManager();

        // Assert
        Assert.AreEqual(1024, manager.MaxMemoryMB);
        Assert.IsFalse(manager.AutoCheckpoint);
        Assert.AreEqual(0.8f, manager.MemoryThreshold);
        Assert.AreEqual(0, manager.CurrentMemoryMB);
    }

    [Test]
    public void CheckpointManager_CreationWithParameters_UsesParameters()
    {
        // Act
        var manager = new CheckpointManager(512, true, 0.7f);

        // Assert
        Assert.AreEqual(512, manager.MaxMemoryMB);
        Assert.IsTrue(manager.AutoCheckpoint);
        Assert.AreEqual(0.7f, manager.MemoryThreshold);
    }

    [Test]
    public void CheckpointManager_RegisterScope_RegistersScope()
    {
        // Arrange
        using var scope = new CheckpointScope("manager_scope");
        var manager = new CheckpointManager();

        // Act
        manager.RegisterScope(scope);

        // Assert
        Assert.AreEqual(1, manager.RegisteredScopeCount);
    }

    [Test]
    public void CheckpointManager_UnregisterScope_RemovesScope()
    {
        // Arrange
        using var scope = new CheckpointScope("unreg_scope");
        var manager = new CheckpointManager();
        manager.RegisterScope(scope);

        // Act
        manager.UnregisterScope(scope);

        // Assert
        Assert.AreEqual(0, manager.RegisteredScopeCount);
    }

    [Test]
    public void CheckpointManager_RegisterNode_UpdatesMemory()
    {
        // Arrange
        var tensor = new Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 4 });
        var operation = new OperationContext("test", gradOutput => new Tensor[] { gradOutput });
        var node = new CheckpointNode(tensor, operation);
        var manager = new CheckpointManager();

        // Act
        manager.RegisterNode(node);

        // Assert
        Assert.AreEqual(1, manager.RegisteredNodeCount);
        Assert.Greater(manager.CurrentMemoryMB, 0);
    }

    [Test]
    public void CheckpointManager_ShouldCheckpoint_ReturnsFalseWhenDisabled()
    {
        // Arrange
        var tensor = new Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 4 });
        var operation = new OperationContext("test", gradOutput => new Tensor[] { gradOutput });
        var node = new CheckpointNode(tensor, operation);
        var manager = new CheckpointManager { AutoCheckpoint = false };

        // Act
        var shouldCheckpoint = manager.ShouldCheckpoint(node);

        // Assert
        Assert.IsFalse(shouldCheckpoint);
    }

    [Test]
    public void CheckpointManager_ClearAllCheckpoints_ClearsAll()
    {
        // Arrange
        using var scope = new CheckpointScope("clear_scope");
        var tensor = new Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 4 });
        var operation = new OperationContext("test", gradOutput => new Tensor[] { gradOutput });
        var node = new CheckpointNode(tensor, operation, scope);
        node.SaveActivations(tensor);
        var manager = new CheckpointManager();
        manager.RegisterScope(scope);

        // Act
        manager.ClearAllCheckpoints();

        // Assert
        Assert.AreEqual(0, manager.CurrentMemoryMB, 0.01);
        Assert.AreEqual(0, scope.CheckpointedNodes.Sum(n => n.SavedActivationCount));
    }

    [Test]
    public void CheckpointManager_GetScopeMemoryUsage_ReturnsCorrectValue()
    {
        // Arrange
        using var scope = new CheckpointScope("mem_scope");
        var tensor = new Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 4 });
        var operation = new OperationContext("test", gradOutput => new Tensor[] { gradOutput });
        var node = new CheckpointNode(tensor, operation, scope);
        node.SaveActivations(tensor);
        var manager = new CheckpointManager();
        manager.RegisterScope(scope);

        // Act
        var memoryMB = manager.GetScopeMemoryUsage("mem_scope");

        // Assert
        Assert.Greater(memoryMB, 0);
    }

    #endregion

    #region TensorCheckpointExtensions Tests

    [Test]
    public void Tensor_MarkCheckpoint_MarksTensor()
    {
        // Arrange
        var tensor = new Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 4 });

        // Act
        tensor.MarkCheckpoint();

        // Assert
        Assert.IsTrue(tensor.IsCheckpoint());
    }

    [Test]
    public void Tensor_IsCheckpoint_ReturnsFalseForUnmarked()
    {
        // Arrange
        var tensor = new Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 4 });

        // Act & Assert
        Assert.IsFalse(tensor.IsCheckpoint());
    }

    [Test]
    public void Tensor_UnmarkCheckpoint_RemovesMark()
    {
        // Arrange
        var tensor = new Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 4 });
        tensor.MarkCheckpoint();

        // Act
        tensor.UnmarkCheckpoint();

        // Assert
        Assert.IsFalse(tensor.IsCheckpoint());
    }

    [Test]
    public void Tensor_MarkCheckpointWithScope_RequiresActiveScope()
    {
        // Arrange
        var tensor = new Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 4 });

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => tensor.MarkCheckpoint("nonexistent_scope"));
    }

    [Test]
    public void Tensor_MarkCheckpointWithActiveScope_Succeeds()
    {
        // Arrange
        using var scope = new CheckpointScope("active_test");
        var tensor = new Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 4 });

        // Act
        tensor.MarkCheckpoint("active_test");

        // Assert
        Assert.IsTrue(tensor.IsCheckpoint());
    }

    [Test]
    public void Tensor_RequiresRecomputation_ReturnsCorrectValue()
    {
        // Arrange
        var tensor = new Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 4 }, requiresGrad: true);

        // Act
        tensor.MarkCheckpoint();

        // Assert
        Assert.IsTrue(tensor.RequiresRecomputation());
    }

    [Test]
    public void Tensor_CreateCheckpointNode_ReturnsNode()
    {
        // Arrange
        var tensor = new Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 4 });
        var operation = new OperationContext("test", gradOutput => new Tensor[] { gradOutput });

        // Act
        var node = tensor.CreateCheckpointNode(operation);

        // Assert
        Assert.IsNotNull(node);
        Assert.IsTrue(tensor.IsCheckpoint());
    }

    [Test]
    public void ClearAllCheckpointMarks_ClearsAllMarks()
    {
        // Arrange
        var tensor1 = new Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 4 });
        var tensor2 = new Tensor(new float[] { 5, 6, 7, 8 }, new int[] { 4 });
        tensor1.MarkCheckpoint();
        tensor2.MarkCheckpoint();

        // Act
        TensorCheckpointExtensions.ClearAllCheckpointMarks();

        // Assert
        Assert.IsFalse(tensor1.IsCheckpoint());
        Assert.IsFalse(tensor2.IsCheckpoint());
    }

    #endregion

    #region Integration Tests

    [Test]
    public void Integration_MultipleCheckpoints_SavesMemory()
    {
        // Arrange
        using var scope1 = new CheckpointScope("scope1");
        using var scope2 = new CheckpointScope("scope2");
        var manager = new CheckpointManager();

        manager.RegisterScope(scope1);
        manager.RegisterScope(scope2);

        var tensor1 = new Tensor(new float[1000], new int[] { 1000 });
        var tensor2 = new Tensor(new float[1000], new int[] { 1000 });
        var operation = new OperationContext("test", gradOutput => new Tensor[] { gradOutput });

        var node1 = new CheckpointNode(tensor1, operation, scope1);
        var node2 = new CheckpointNode(tensor2, operation, scope2);

        node1.SaveActivations(tensor1);
        node2.SaveActivations(tensor2);

        // Act
        var totalMemory = manager.GetTotalMemoryUsage();

        // Assert
        Assert.Greater(totalMemory, 0);
    }

    [Test]
    public void Integration_DisabledCheckpoint_DoesNotSave()
    {
        // Arrange
        using var disabledScope = new CheckpointScope("disabled", false);
        var tensor = new Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 4 });
        var operation = new OperationContext("test", gradOutput => new Tensor[] { gradOutput });
        var node = new CheckpointNode(tensor, operation, disabledScope);

        // Act
        node.SaveActivations(tensor);

        // Assert
        Assert.AreEqual(0, disabledScope.CheckpointedNodes.Count);
    }

    #endregion
}
