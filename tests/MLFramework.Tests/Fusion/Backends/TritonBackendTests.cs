using NUnit.Framework;
using MLFramework.Fusion;
using MLFramework.Fusion.Backends;
using MLFramework.Tests.Fusion;

namespace MLFramework.Tests.Fusion.Backends;

/// <summary>
/// Tests for TritonBackend
/// </summary>
[TestFixture]
public class TritonBackendTests
{
    private TritonBackend _backend = null!;
    private MockTritonCompiler _compiler = null!;
    private MockTritonAutotuner _autotuner = null!;
    private ConsoleLogger _logger = null!;

    [SetUp]
    public void SetUp()
    {
        _compiler = new MockTritonCompiler();
        _autotuner = new MockTritonAutotuner();
        _logger = new ConsoleLogger();
        _backend = new TritonBackend(_compiler, _autotuner, _logger);
    }

    [Test]
    public void TritonBackend_Name_ReturnsTriton()
    {
        Assert.AreEqual("Triton", _backend.Name);
    }

    [Test]
    public void TritonBackend_Type_ReturnsTritonType()
    {
        Assert.AreEqual(FusionBackendType.Triton, _backend.Type);
    }

    [Test]
    public void TritonBackend_CanFuse_ElementWiseOperations()
    {
        var operations = new[] { OperationTestHelper.CreateAddOp(), OperationTestHelper.CreateAddOp() };

        Assert.IsTrue(_backend.CanFuse(operations));
    }

    [Test]
    public void TritonBackend_CanFuse_SingleOperation()
    {
        var operations = new[] { OperationTestHelper.CreateAddOp() };

        Assert.IsTrue(_backend.CanFuse(operations));
    }

    [Test]
    public void TritonBackend_CanFuse_EmptyOperations_ReturnsFalse()
    {
        var operations = Array.Empty<Operation>();

        Assert.IsFalse(_backend.CanFuse(operations));
    }

    [Test]
    public void TritonBackend_Fuse_CreatesFusedOperation()
    {
        var operations = new[] { OperationTestHelper.CreateAddOp(), OperationTestHelper.CreateAddOp() };
        var options = new FusionOptions();

        var result = _backend.Fuse(operations, options);

        Assert.IsNotNull(result);
        Assert.AreEqual(2, result.OriginalOpCount);
        Assert.AreEqual(1, result.FusedOpCount);
        Assert.AreEqual(1, result.FusedOperations.Count);
    }

    [Test]
    public void TritonBackend_Fuse_CannotFuse_ThrowsException()
    {
        var operations = Array.Empty<Operation>();
        var options = new FusionOptions();

        var ex = Assert.Throws<InvalidOperationException>(() =>
            _backend.Fuse(operations, options));

        Assert.That(ex.Message, Does.Contain("Cannot fuse these operations"));
    }

    [Test]
    public void TritonBackend_Compile_CreatesCompiledKernel()
    {
        var operations = new[] { OperationTestHelper.CreateAddOp() };
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
    }

    [Test]
    public void TritonBackend_GetCapabilities_ReturnsExpectedCapabilities()
    {
        var capabilities = _backend.GetCapabilities();

        Assert.IsNotNull(capabilities);
        Assert.IsTrue(capabilities.SupportsAutotuning);
        Assert.IsTrue(capabilities.SupportsProfiling);
        Assert.IsTrue(capabilities.SupportsJITCompilation);
        Assert.IsTrue(capabilities.SupportsBinaryCache);
        Assert.Contains(FusionPatternType.ElementWise, capabilities.SupportedPatterns);
        Assert.Contains(FusionPatternType.ConvActivation, capabilities.SupportedPatterns);
        Assert.Contains(DataType.Float32, capabilities.SupportedDataTypes);
        Assert.Contains(DataType.Float16, capabilities.SupportedDataTypes);
    }

    [Test]
    public void TritonBackend_Initialize_WithConfig()
    {
        var config = new BackendConfiguration
        {
            DeviceId = "cuda:0",
            Options = new Dictionary<string, object>()
        };

        Assert.DoesNotThrow(() => _backend.Initialize(config));
    }
}
