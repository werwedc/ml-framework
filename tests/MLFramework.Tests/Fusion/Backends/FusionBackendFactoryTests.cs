using NUnit.Framework;
using MLFramework.Core;
using MLFramework.Fusion;
using MLFramework.Fusion.Backends;

namespace MLFramework.Tests.Fusion.Backends;

/// <summary>
/// Tests for FusionBackendFactory
/// </summary>
[TestFixture]
public class FusionBackendFactoryTests
{
    [Test]
    public void Factory_CreateBackend_Triton_ReturnsTritonBackend()
    {
        var config = new BackendConfiguration
        {
            DeviceId = "cuda:0",
            Options = new Dictionary<string, object>()
        };

        var backend = FusionBackendFactory.CreateBackend(FusionBackendType.Triton, config);

        Assert.IsNotNull(backend);
        Assert.AreEqual("Triton", backend.Name);
        Assert.AreEqual(FusionBackendType.Triton, backend.Type);
    }

    [Test]
    public void Factory_CreateBackend_XLA_ReturnsXLABackend()
    {
        var config = new BackendConfiguration
        {
            DeviceId = "cuda:0",
            Options = new Dictionary<string, object>()
        };

        var backend = FusionBackendFactory.CreateBackend(FusionBackendType.XLA, config);

        Assert.IsNotNull(backend);
        Assert.AreEqual("XLA", backend.Name);
        Assert.AreEqual(FusionBackendType.XLA, backend.Type);
    }

    [Test]
    public void Factory_CreateBackend_NotRegistered_ThrowsException()
    {
        var config = new BackendConfiguration
        {
            DeviceId = "cuda:0",
            Options = new Dictionary<string, object>()
        };

        var ex = Assert.Throws<ArgumentException>(() =>
            FusionBackendFactory.CreateBackend(FusionBackendType.Custom, config));

        Assert.That(ex.Message, Does.Contain("is not registered"));
    }

    [Test]
    public void Factory_RegisterBackend_CustomBackend()
    {
        var config = new BackendConfiguration
        {
            DeviceId = "cuda:0",
            Options = new Dictionary<string, object>()
        };

        FusionBackendFactory.RegisterBackend(FusionBackendType.Custom, () =>
            new MockFusionBackend("CustomBackend"));

        var backend = FusionBackendFactory.CreateBackend(FusionBackendType.Custom, config);

        Assert.IsNotNull(backend);
        Assert.AreEqual("CustomBackend", backend.Name);
        Assert.AreEqual(FusionBackendType.Custom, backend.Type);
    }

    [Test]
    public void Factory_RegisterBackend_OverwritesExisting()
    {
        var config = new BackendConfiguration
        {
            DeviceId = "cuda:0",
            Options = new Dictionary<string, object>()
        };

        FusionBackendFactory.RegisterBackend(FusionBackendType.Triton, () =>
            new MockFusionBackend("CustomTriton"));

        var backend = FusionBackendFactory.CreateBackend(FusionBackendType.Triton, config);

        Assert.AreEqual("CustomTriton", backend.Name);

        // Restore default
        FusionBackendFactory.RegisterDefaults();
    }

    [Test]
    public void Factory_CreateBackend_InitializesBackend()
    {
        var testBackend = new TestableFusionBackend();
        var config = new BackendConfiguration
        {
            DeviceId = "cuda:0",
            Options = new Dictionary<string, object>()
        };

        FusionBackendFactory.RegisterBackend(FusionBackendType.Custom, () => testBackend);

        var backend = FusionBackendFactory.CreateBackend(FusionBackendType.Custom, config);

        Assert.IsTrue(testBackend.Initialized);
        Assert.AreEqual("cuda:0", testBackend.LastConfig?.DeviceId);
    }

    /// <summary>
    /// Testable backend that tracks initialization
    /// </summary>
    private class TestableFusionBackend : IFusionBackend
    {
        public string Name => "Testable";
        public FusionBackendType Type => FusionBackendType.Custom;
        public bool Initialized { get; private set; }
        public BackendConfiguration? LastConfig { get; private set; }

        public bool CanFuse(IReadOnlyList<Operation> operations)
        {
            return false;
        }

        public FusionResult Fuse(IReadOnlyList<Operation> operations, FusionOptions options)
        {
            throw new NotImplementedException();
        }

        public CompiledKernel Compile(FusedOperation fusedOp, CompilationOptions options)
        {
            throw new NotImplementedException();
        }

        public BackendCapabilities GetCapabilities()
        {
            throw new NotImplementedException();
        }

        public void Initialize(BackendConfiguration config)
        {
            Initialized = true;
            LastConfig = config;
        }
    }
}
