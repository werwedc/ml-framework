using NUnit.Framework;
using MLFramework.Fusion;

namespace MLFramework.Tests.Fusion.KernelCode;

/// <summary>
/// Tests for kernel code generator
/// </summary>
[TestFixture]
public class KernelCodeGeneratorTests
{
    private CodeTemplateEngine _templateEngine = null!;
    private SimpleTemplateRenderer _renderer = null!;
    private TestKernelCodeGenerator _generator = null!;

    [SetUp]
    public void Setup()
    {
        _renderer = new SimpleTemplateRenderer();
        _templateEngine = new CodeTemplateEngine(_renderer);
        _generator = new TestKernelCodeGenerator(_templateEngine);

        // Register test template
        var testTemplate = @"template<typename T>
__global__ void {{kernel_name}}(
    {{parameters}},
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements)
        return;

    // Load input
    T val = input[idx];

    {{kernel_body}}

    // Store output
    output[idx] = val;
}";
        _templateEngine.RegisterTemplate("test_template", testTemplate);
    }

    [Test]
    public void GenerateKernel_ProducesValidCode()
    {
        var fusedOp = CreateSimpleFusedOperation();
        var options = new GenerationOptions();

        var result = _generator.GenerateKernel(fusedOp, options);

        Assert.IsNotNull(result.KernelSourceCode);
        Assert.IsNotEmpty(result.KernelSourceCode);
        Assert.IsNotEmpty(result.Parameters);
        Assert.IsTrue(result.KernelSourceCode.Contains("__global__"));
        Assert.IsTrue(result.KernelSourceCode.Contains("test_kernel"));
    }

    [Test]
    public void GenerateKernel_ComputesParameters()
    {
        var fusedOp = CreateFusedOperationWithVariables();
        var options = new GenerationOptions();

        var result = _generator.GenerateKernel(fusedOp, options);

        Assert.AreEqual(2, result.Parameters.Count);
        Assert.AreEqual(ParameterDirection.Input, result.Parameters[0].Direction);
        Assert.AreEqual(ParameterDirection.Output, result.Parameters[1].Direction);
        Assert.AreEqual(DataType.Float32, result.Parameters[0].DataType);
    }

    [Test]
    public void GenerateKernel_ComputesCompilationMetadata()
    {
        var fusedOp = CreateSimpleFusedOperation();
        var options = new GenerationOptions();

        var result = _generator.GenerateKernel(fusedOp, options);

        Assert.Greater(result.Metadata.ThreadBlockSize, 0);
        Assert.Greater(result.Metadata.GridSize, 0);
        Assert.IsNotEmpty(result.Metadata.RequiredCapabilities);
    }

    [Test]
    public void TemplateRenderer_ReplacesPlaceholders()
    {
        var template = "{{kernel_name}} - {{shared_memory}}";
        var context = new TemplateContext
        {
            KernelName = "test_kernel",
            MemoryLayout = new MemoryLayout
            {
                TensorLayout = TensorLayout.NCHW,
                SharedMemoryBytes = 1024,
                RegisterBytes = 512
            },
            ComputeRequirements = new ComputeRequirements
            {
                ThreadBlocks = 1,
                ThreadsPerBlock = 256,
                RequiresSharedMemory = false,
                RequiresAtomicOps = false
            },
            Parameters = Array.Empty<KernelParameter>(),
            Nodes = Array.Empty<FusionOpNode>(),
            Options = new GenerationOptions()
        };

        var result = _renderer.Render(template, context);

        Assert.IsTrue(result.Contains("test_kernel"));
        Assert.IsTrue(result.Contains("1024"));
    }

    [Test]
    public void CanCompile_SupportedOp_ReturnsTrue()
    {
        var fusedOp = CreateFusedOpWithSupportedOps();
        Assert.IsTrue(_generator.CanCompile(fusedOp));
    }

    [Test]
    public void CanCompile_UnsupportedOp_ReturnsFalse()
    {
        var fusedOp = CreateFusedOpWithUnsupportedOps();
        Assert.IsFalse(_generator.CanCompile(fusedOp));
    }

    private FusedOperation CreateSimpleFusedOperation()
    {
        var ir = new FusionIR
        {
            Id = "test_ir",
            Nodes = new[]
            {
                new FusionOpNode
                {
                    Id = "node1",
                    OriginalOpType = "Add",
                    InputVars = new[] { "input1", "input2" },
                    OutputVar = "temp1",
                    Attributes = new Dictionary<string, object>()
                }
            },
            Variables = new[]
            {
                new FusionVariable
                {
                    Name = "input1",
                    Shape = new TensorShape(32, 32),
                    DataType = DataType.Float32,
                    Location = MemoryLocation.Input
                },
                new FusionVariable
                {
                    Name = "input2",
                    Shape = new TensorShape(32, 32),
                    DataType = DataType.Float32,
                    Location = MemoryLocation.Input
                },
                new FusionVariable
                {
                    Name = "temp1",
                    Shape = new TensorShape(32, 32),
                    DataType = DataType.Float32,
                    Location = MemoryLocation.Output
                }
            },
            MemoryLayout = new MemoryLayout
            {
                TensorLayout = TensorLayout.NCHW,
                SharedMemoryBytes = 0,
                RegisterBytes = 512
            },
            ComputeRequirements = new ComputeRequirements
            {
                ThreadBlocks = 1,
                ThreadsPerBlock = 256,
                RequiresSharedMemory = false,
                RequiresAtomicOps = false
            }
        };

        var kernelSpec = new KernelSpecification
        {
            KernelName = "test_kernel",
            Strategy = FusionStrategy.Merge,
            InputTensors = ir.Variables.Where(v => v.Location == MemoryLocation.Input).ToList(),
            OutputTensors = ir.Variables.Where(v => v.Location == MemoryLocation.Output).ToList(),
            TemporaryMemoryBytes = 0,
            RegisterBytes = 512,
            ThreadBlockConfig = new ThreadBlockConfiguration { X = 256, Y = 1, Z = 1 },
            CompilationFlags = Array.Empty<string>()
        };

        return FusedOperation.Create(
            "test_op",
            Array.Empty<Operation>(),
            new FusionPatternDefinition { Name = "TestPattern" },
            ir,
            kernelSpec);
    }

    private FusedOperation CreateFusedOperationWithVariables()
    {
        var ir = new FusionIR
        {
            Id = "test_ir",
            Nodes = Array.Empty<FusionOpNode>(),
            Variables = new[]
            {
                new FusionVariable
                {
                    Name = "input",
                    Shape = new TensorShape(32, 32),
                    DataType = DataType.Float32,
                    Location = MemoryLocation.Input
                },
                new FusionVariable
                {
                    Name = "output",
                    Shape = new TensorShape(32, 32),
                    DataType = DataType.Float32,
                    Location = MemoryLocation.Output
                }
            },
            MemoryLayout = new MemoryLayout
            {
                TensorLayout = TensorLayout.NCHW,
                SharedMemoryBytes = 0,
                RegisterBytes = 512
            },
            ComputeRequirements = new ComputeRequirements
            {
                ThreadBlocks = 1,
                ThreadsPerBlock = 256,
                RequiresSharedMemory = false,
                RequiresAtomicOps = false
            }
        };

        var kernelSpec = new KernelSpecification
        {
            KernelName = "test_kernel",
            Strategy = FusionStrategy.Merge,
            InputTensors = ir.Variables.Where(v => v.Location == MemoryLocation.Input).ToList(),
            OutputTensors = ir.Variables.Where(v => v.Location == MemoryLocation.Output).ToList(),
            TemporaryMemoryBytes = 0,
            RegisterBytes = 512,
            ThreadBlockConfig = new ThreadBlockConfiguration { X = 256, Y = 1, Z = 1 },
            CompilationFlags = Array.Empty<string>()
        };

        return FusedOperation.Create(
            "test_op",
            Array.Empty<Operation>(),
            new FusionPatternDefinition { Name = "TestPattern" },
            ir,
            kernelSpec);
    }

    private FusedOperation CreateFusedOpWithSupportedOps()
    {
        var ir = new FusionIR
        {
            Id = "test_ir",
            Nodes = new[]
            {
                new FusionOpNode
                {
                    Id = "node1",
                    OriginalOpType = "Add",
                    InputVars = Array.Empty<string>(),
                    OutputVar = "output",
                    Attributes = new Dictionary<string, object>()
                }
            },
            Variables = new[]
            {
                new FusionVariable
                {
                    Name = "output",
                    Shape = new TensorShape(32, 32),
                    DataType = DataType.Float32,
                    Location = MemoryLocation.Output
                }
            },
            MemoryLayout = new MemoryLayout
            {
                TensorLayout = TensorLayout.NCHW,
                SharedMemoryBytes = 0,
                RegisterBytes = 512
            },
            ComputeRequirements = new ComputeRequirements
            {
                ThreadBlocks = 1,
                ThreadsPerBlock = 256,
                RequiresSharedMemory = false,
                RequiresAtomicOps = false
            }
        };

        var kernelSpec = new KernelSpecification
        {
            KernelName = "test_kernel",
            Strategy = FusionStrategy.Merge,
            InputTensors = Array.Empty<FusionVariable>(),
            OutputTensors = ir.Variables.Where(v => v.Location == MemoryLocation.Output).ToList(),
            TemporaryMemoryBytes = 0,
            RegisterBytes = 512,
            ThreadBlockConfig = new ThreadBlockConfiguration { X = 256, Y = 1, Z = 1 },
            CompilationFlags = Array.Empty<string>()
        };

        return FusedOperation.Create(
            "test_op",
            Array.Empty<Operation>(),
            new FusionPatternDefinition { Name = "TestPattern" },
            ir,
            kernelSpec);
    }

    private FusedOperation CreateFusedOpWithUnsupportedOps()
    {
        var ir = new FusionIR
        {
            Id = "test_ir",
            Nodes = new[]
            {
                new FusionOpNode
                {
                    Id = "node1",
                    OriginalOpType = "UnsupportedOp",
                    InputVars = Array.Empty<string>(),
                    OutputVar = "output",
                    Attributes = new Dictionary<string, object>()
                }
            },
            Variables = new[]
            {
                new FusionVariable
                {
                    Name = "output",
                    Shape = new TensorShape(32, 32),
                    DataType = DataType.Float32,
                    Location = MemoryLocation.Output
                }
            },
            MemoryLayout = new MemoryLayout
            {
                TensorLayout = TensorLayout.NCHW,
                SharedMemoryBytes = 0,
                RegisterBytes = 512
            },
            ComputeRequirements = new ComputeRequirements
            {
                ThreadBlocks = 1,
                ThreadsPerBlock = 256,
                RequiresSharedMemory = false,
                RequiresAtomicOps = false
            }
        };

        var kernelSpec = new KernelSpecification
        {
            KernelName = "test_kernel",
            Strategy = FusionStrategy.Merge,
            InputTensors = Array.Empty<FusionVariable>(),
            OutputTensors = ir.Variables.Where(v => v.Location == MemoryLocation.Output).ToList(),
            TemporaryMemoryBytes = 0,
            RegisterBytes = 512,
            ThreadBlockConfig = new ThreadBlockConfiguration { X = 256, Y = 1, Z = 1 },
            CompilationFlags = Array.Empty<string>()
        };

        return FusedOperation.Create(
            "test_op",
            Array.Empty<Operation>(),
            new FusionPatternDefinition { Name = "TestPattern" },
            ir,
            kernelSpec);
    }

    /// <summary>
    /// Test generator for unit tests
    /// </summary>
    private class TestKernelCodeGenerator : KernelCodeGeneratorBase
    {
        public override KernelBackendType BackendType => KernelBackendType.CUDA;

        public TestKernelCodeGenerator(ICodeTemplateEngine templateEngine)
            : base(templateEngine)
        {
        }

        public override bool CanCompile(FusedOperation fusedOp)
        {
            // Check if all operations are supported
            return fusedOp.IntermediateRepresentation.Nodes.All(op =>
                op.OriginalOpType == "Add" || op.OriginalOpType == "Mul" ||
                op.OriginalOpType == "ReLU" || op.OriginalOpType == "Sigmoid");
        }

        protected override string GetTemplate(FusionIR ir)
        {
            return LoadTemplate("test_template");
        }

        protected override TemplateContext BuildTemplateContext(
            FusionIR ir,
            IReadOnlyList<KernelParameter> parameters,
            GenerationOptions options)
        {
            return new TemplateContext
            {
                KernelName = $"test_kernel_{ir.Id}",
                Parameters = parameters,
                Nodes = ir.Nodes,
                MemoryLayout = ir.MemoryLayout,
                ComputeRequirements = ir.ComputeRequirements,
                Options = options
            };
        }
    }
}
