using NUnit.Framework;
using MLFramework.Core;
using MLFramework.Fusion;
using FusionBackends = MLFramework.Fusion.Backends;
using MLFramework.Tests.Fusion;

namespace MLFramework.Tests.Fusion.Backends;

/// <summary>
/// Tests for FusionBackendRegistry
/// </summary>
[TestFixture]
public class FusionBackendRegistryTests
{
    [Test]
    public void Registry_RegisterAndRetrieveBackend()
    {
        var registry = new FusionBackendRegistry();
        var backend = CreateMockBackend("TestBackend");

        registry.Register(backend);
        var retrieved = registry.GetBackend("TestBackend");

        Assert.IsNotNull(retrieved);
        Assert.AreEqual("TestBackend", retrieved.Name);
    }

    [Test]
    public void Registry_SetDefaultBackend()
    {
        var registry = new FusionBackendRegistry();
        var backend1 = CreateMockBackend("Backend1");
        var backend2 = CreateMockBackend("Backend2");

        registry.Register(backend1);
        registry.Register(backend2);

        registry.SetDefaultBackend("Backend2");
        var defaultBackend = registry.GetDefaultBackend();

        Assert.AreEqual("Backend2", defaultBackend.Name);
    }

    [Test]
    public void Registry_SetDefaultBackend_NotRegistered_ThrowsException()
    {
        var registry = new FusionBackendRegistry();
        var backend1 = CreateMockBackend("Backend1");

        registry.Register(backend1);

        var ex = Assert.Throws<InvalidOperationException>(() =>
            registry.SetDefaultBackend("NonExistentBackend"));

        Assert.That(ex.Message, Does.Contain("is not registered"));
    }

    [Test]
    public void Registry_GetDefaultBackend_NotSet_ThrowsException()
    {
        var registry = new FusionBackendRegistry();

        var ex = Assert.Throws<InvalidOperationException>(() =>
            registry.GetDefaultBackend());

        Assert.That(ex.Message, Does.Contain("not registered"));
    }

    [Test]
    public void Registry_UnregisterBackend()
    {
        var registry = new FusionBackendRegistry();
        var backend = CreateMockBackend("TestBackend");

        registry.Register(backend);
        Assert.IsNotNull(registry.GetBackend("TestBackend"));

        registry.Unregister("TestBackend");
        Assert.IsNull(registry.GetBackend("TestBackend"));
    }

    [Test]
    public void Registry_GetAllBackends()
    {
        var registry = new FusionBackendRegistry();
        var backend1 = CreateMockBackend("Backend1");
        var backend2 = CreateMockBackend("Backend2");

        registry.Register(backend1);
        registry.Register(backend2);

        var allBackends = registry.GetAllBackends();

        Assert.AreEqual(2, allBackends.Count);
        Assert.IsTrue(allBackends.Any(b => b.Name == "Backend1"));
        Assert.IsTrue(allBackends.Any(b => b.Name == "Backend2"));
    }

    [Test]
    public void Registry_FindCapableBackend()
    {
        var registry = new FusionBackendRegistry();
        var tritonBackend = new TritonBackend(new MockTritonCompiler(), new MockTritonAutotuner(), new ConsoleLogger());
        var operations = new[] { OperationTestHelper.CreateAddOp(), OperationTestHelper.CreateAddOp() };

        registry.Register(tritonBackend);

        var capableBackend = registry.FindCapableBackend(operations);

        Assert.IsNotNull(capableBackend);
        Assert.AreEqual("Triton", capableBackend.Name);
    }

    [Test]
    public void Registry_FindCapableBackend_NoCapableBackend()
    {
        var registry = new FusionBackendRegistry();
        var tritonBackend = new TritonBackend(new MockTritonCompiler(), new MockTritonAutotuner(), new ConsoleLogger());
        // Create operations that cannot be fused (empty list)
        var operations = Array.Empty<Operation>();

        registry.Register(tritonBackend);

        var capableBackend = registry.FindCapableBackend(operations);

        Assert.IsNull(capableBackend);
    }

    private FusionBackends.IFusionBackend CreateMockBackend(string name)
    {
        return new MockFusionBackend(name);
    }
}

/// <summary>
/// Mock backend for testing
/// </summary>
internal class MockFusionBackend : FusionBackends.IFusionBackend
{
    public string Name { get; }
    public FusionBackends.FusionBackendType Type => FusionBackends.FusionBackendType.Custom;

    public MockFusionBackend(string name)
    {
        Name = name;
    }

    public bool CanFuse(IReadOnlyList<Operation> operations)
    {
        return operations.Count > 0;
    }

    public FusionBackends.FusionResult Fuse(IReadOnlyList<Operation> operations, FusionBackends.FusionOptions options)
    {
        throw new NotImplementedException();
    }

    public FusionBackends.CompiledKernel Compile(FusionBackends.FusedOperation fusedOp, FusionBackends.CompilationOptions options)
    {
        throw new NotImplementedException();
    }

    public FusionBackends.BackendCapabilities GetCapabilities()
    {
        return new FusionBackends.BackendCapabilities
        {
            SupportedPatterns = new HashSet<FusionPatternType> { FusionPatternType.ElementWise },
            SupportedDataTypes = new HashSet<DataType> { DataType.Float32 },
            SupportsAutotuning = false,
            SupportsProfiling = false,
            SupportsJITCompilation = false,
            SupportsBinaryCache = false
        };
    }

    public void Initialize(FusionBackends.BackendConfiguration config)
    {
        // No-op for mock
    }
}
