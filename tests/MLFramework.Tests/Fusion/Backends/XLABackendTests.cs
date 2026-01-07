using NUnit.Framework;
using MLFramework.Fusion;
using MLFramework.Fusion.Backends;
using MLFramework.Tests.Fusion;

namespace MLFramework.Tests.Fusion.Backends;

/// <summary>
/// Tests for XLABackend
/// </summary>
[TestFixture]
public class XLABackendTests
{
    private XLABackend _backend = null!;
    private MockXLACompiler _compiler = null!;
    private ConsoleLogger _logger = null!;

    [SetUp]
    public void SetUp()
    {
        _compiler = new MockXLACompiler();
        _logger = new ConsoleLogger();
        _backend = new XLABackend(_compiler, _logger);
    }

    [Test]
    public void XLABackend_Name_ReturnsXLA()
    {
        Assert.AreEqual("XLA", _backend.Name);
    }

    [Test]
    public void XLABackend_Type_ReturnsXLAType()
    {
        Assert.AreEqual(FusionBackendType.XLA, _backend.Type);
    }

    [Test]
    public void XLABackend_CanFuse_SmallOperationCount()
    {
        var operations = Enumerable.Range(0, 10)
            .Select(_ => OperationTestHelper.CreateAddOp())
            .ToArray();

        Assert.IsTrue(_backend.CanFuse(operations));
    }

    [Test]
    public void XLABackend_CanFuse_MaximumOperationCount()
    {
        var operations = Enumerable.Range(0, 20)
            .Select(_ => OperationTestHelper.CreateAddOp())
            .ToArray();

        Assert.IsTrue(_backend.CanFuse(operations));
    }

    [Test]
    public void XLABackend_CanFuse_ExceedsLimit_ReturnsFalse()
    {
        var operations = Enumerable.Range(0, 21)
            .Select(_ => OperationTestHelper.CreateAddOp())
            .ToArray();

        Assert.IsFalse(_backend.CanFuse(operations));
    }

    [Test]
    public void XLABackend_CanFuse_EmptyOperations_ReturnsFalse()
    {
        var operations = Array.Empty<Operation>();

        Assert.IsFalse(_backend.CanFuse(operations));
    }

    [Test]
    public void XLABackend_Fuse_CreatesFusedOperation()
    {
        var operations = Enumerable.Range(0, 5)
            .Select(_ => OperationTestHelper.CreateAddOp())
            .ToArray();
        var options = new FusionOptions();

        var result = _backend.Fuse(operations, options);

        Assert.IsNotNull(result);
        Assert.AreEqual(5, result.OriginalOpCount);
        Assert.AreEqual(1, result.FusedOpCount);
        Assert.AreEqual(1, result.FusedOperations.Count);
    }

    [Test]
    public void XLABackend_Fuse_CannotFuse_ThrowsException()
    {
        var operations = Array.Empty<Operation>();
        var options = new FusionOptions();

        var ex = Assert.Throws<InvalidOperationException>(() =>
            _backend.Fuse(operations, options));

        Assert.That(ex.Message, Does.Contain("Cannot fuse these operations"));
    }

    [Test]
    public void XLABackend_Compile_CreatesCompiledKernel()
    {
        var operations = Enumerable.Range(0, 5)
            .Select(_ => OperationTestHelper.CreateAddOp())
            .ToArray();
        var fuseOptions = new FusionOptions();
        var compileOptions = new CompilationOptions { OptimizationLevel = 2 };

        var fuseResult = _backend.Fuse(operations, fuseOptions);
        var fusedOp = fuseResult.FusedOperations[0];

        var compiledKernel = _backend.Compile(fusedOp, compileOptions);

        Assert.IsNotNull(compiledKernel);
        Assert.IsNotNull(compiledKernel.KernelId);
        Assert.IsNotNull(compiledKernel.Binary);
        Assert.IsNotNull(compiledKernel.LaunchConfig);
        Assert.IsNotNull(compiledKernel.Metrics);
        Assert.AreEqual(compileOptions.OptimizationLevel, compiledKernel.Metrics.OptimizationLevel);
        Assert.That(compiledKernel.KernelId, Does.EndWith("_xla"));
    }

    [Test]
    public void XLABackend_GetCapabilities_ReturnsExpectedCapabilities()
    {
        var capabilities = _backend.GetCapabilities();

        Assert.IsNotNull(capabilities);
        Assert.IsTrue(capabilities.SupportsAutotuning);
        Assert.IsTrue(capabilities.SupportsProfiling);
        Assert.IsTrue(capabilities.SupportsJITCompilation);
        Assert.IsTrue(capabilities.SupportsBinaryCache);
        Assert.Contains(FusionPatternType.ElementWise, capabilities.SupportedPatterns);
        Assert.Contains(FusionPatternType.ConvActivation, capabilities.SupportedPatterns);
        Assert.Contains(FusionPatternType.Mixed, capabilities.SupportedPatterns);
        Assert.Contains(DataType.Float32, capabilities.SupportedDataTypes);
        Assert.Contains(DataType.Float16, capabilities.SupportedDataTypes);
        Assert.Contains(DataType.BFloat16, capabilities.SupportedDataTypes);
    }

    [Test]
    public void XLABackend_Initialize_WithConfig()
    {
        var config = new BackendConfiguration
        {
            DeviceId = "cuda:0",
            Options = new Dictionary<string, object>()
        };

        Assert.DoesNotThrow(() => _backend.Initialize(config));
    }
}
