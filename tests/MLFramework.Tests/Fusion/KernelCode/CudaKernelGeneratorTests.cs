using NUnit.Framework;
using MLFramework.Core;
using MLFramework.Fusion;

namespace MLFramework.Tests.Fusion.KernelCode;

/// <summary>
/// Tests for CUDA kernel generator
/// </summary>
[TestFixture]
public class CudaKernelGeneratorTests
{
    private CodeTemplateEngine _templateEngine = null!;
    private SimpleTemplateRenderer _renderer = null!;
    private CudaKernelGenerator _generator = null!;
    private MemoryAccessOptimizer _optimizer = null!;
    private SharedMemoryPlanner _memoryPlanner = null!;

    [SetUp]
    public void Setup()
    {
        _renderer = new SimpleTemplateRenderer();
        _templateEngine = new CodeTemplateEngine(_renderer);
        _optimizer = new MemoryAccessOptimizer();
        _memoryPlanner = new SharedMemoryPlanner();

        _generator = new CudaKernelGenerator(
            _templateEngine,
            _optimizer,
            _memoryPlanner);

        // Register CUDA templates
        CudaTemplateRegistrar.RegisterTemplates(_templateEngine);
    }

    [Test]
    public void GenerateKernel_ElementWise_ProducesValidCode()
    {
        var fusedOp = CreateElementWiseFusedOperation();
        var options = new GenerationOptions { EnableVectorization = true };

        var result = _generator.GenerateKernel(fusedOp, options);

        Assert.IsNotNull(result.KernelSourceCode);
        Assert.IsTrue(result.KernelSourceCode.Contains("__global__"));
        Assert.IsTrue(result.KernelSourceCode.Contains("float*"));
        Assert.IsTrue(result.KernelSourceCode.Contains("fused_cuda"));
    }

    [Test]
    public void GenerateKernel_ConvWithActivation_ProducesValidCode()
    {
        var fusedOp = CreateConvActivationFusedOperation();
        var options = new GenerationOptions { EnableSharedMemory = true };

        var result = _generator.GenerateKernel(fusedOp, options);

        Assert.IsNotNull(result.KernelSourceCode);
        Assert.IsTrue(result.KernelSourceCode.Contains("__global__"));
        Assert.IsTrue(result.KernelSourceCode.Contains("bias"));
        Assert.IsTrue(result.KernelSourceCode.Contains("weight"));
        Assert.IsTrue(result.KernelSourceCode.Contains("extern __shared__"));
    }

    [Test]
    public void MemoryAccessOptimizer_VectorizesFloat()
    {
        var context = CreateContextWithFloat32();
        var options = new GenerationOptions { EnableVectorization = true };

        var optimized = _optimizer.Optimize(context, options);

        Assert.IsTrue(optimized.Options.EnableVectorization);
    }

    [Test]
    public void SharedMemoryPlanner_ComputesTilingForConv()
    {
        var ir = CreateConvIR();
        var options = new GenerationOptions { EnableSharedMemory = true };

        var layout = _memoryPlanner.PlanMemory(ir, options);

        Assert.Greater(layout.SharedMemoryBytes, 0);
        Assert.Less(layout.SharedMemoryBytes, 48 * 1024);
    }

    [Test]
    public void SharedMemoryPlanner_ElementWise_NoTiling()
    {
        var ir = CreateElementWiseIR();
        var options = new GenerationOptions { EnableSharedMemory = true };

        var layout = _memoryPlanner.PlanMemory(ir, options);

        Assert.AreEqual(0, layout.SharedMemoryBytes);
    }

    [Test]
    public void GenerateReLUCode_ProducesCorrectCode()
    {
        var node = CreateFusionOpNode("ReLU", new[] { "input" }, "output");
        var code = CudaOperationCodeGenerator.GenerateReLUCode(node);

        Assert.IsTrue(code.Contains("fmaxf"));
        Assert.IsTrue(code.Contains("0.0f"));
    }

    [Test]
    public void GenerateSigmoidCode_ProducesCorrectCode()
    {
        var node = CreateFusionOpNode("Sigmoid", new[] { "input" }, "output");
        var code = CudaOperationCodeGenerator.GenerateSigmoidCode(node);

        Assert.IsTrue(code.Contains("expf"));
        Assert.IsTrue(code.Contains("1.0f"));
    }

    [Test]
    public void GenerateAddCode_TwoInputs_ProducesCorrectCode()
    {
        var node = CreateFusionOpNode("Add", new[] { "input1", "input2" }, "output");
        var code = CudaOperationCodeGenerator.GenerateAddCode(node);

        Assert.IsTrue(code.Contains("+"));
        Assert.IsTrue(code.Contains("input1"));
        Assert.IsTrue(code.Contains("input2"));
    }

    [Test]
    public void CudaTypeMapper_Float32_ReturnsCorrectType()
    {
        var typeName = CudaTypeMapper.GetCudaTypeName(DataType.Float32);

        Assert.AreEqual("float", typeName);
    }

    [Test]
    public void CudaTypeMapper_Float16_ReturnsCorrectType()
    {
        var typeName = CudaTypeMapper.GetCudaTypeName(DataType.Float16);

        Assert.AreEqual("half", typeName);
    }

    [Test]
    public void CudaTypeMapper_BFloat16_ReturnsCorrectType()
    {
        var typeName = CudaTypeMapper.GetCudaTypeName(DataType.BFloat16);

        Assert.AreEqual("__nv_bfloat16", typeName);
    }

    [Test]
    public void CudaTypeMapper_Int32_ReturnsCorrectType()
    {
        var typeName = CudaTypeMapper.GetCudaTypeName(DataType.Int32);

        Assert.AreEqual("int", typeName);
    }

    [Test]
    public void CanCompile_SupportedOps_ReturnsTrue()
    {
        var fusedOp = CreateElementWiseFusedOperation();

        Assert.IsTrue(_generator.CanCompile(fusedOp));
    }

    [Test]
    public void CanCompile_UnsupportedOps_ReturnsFalse()
    {
        var fusedOp = CreateUnsupportedFusedOperation();

        Assert.IsFalse(_generator.CanCompile(fusedOp));
    }

    [Test]
    public void GenerateKernel_ComputesParameters()
    {
        var fusedOp = CreateElementWiseFusedOperation();
        var options = new GenerationOptions();

        var result = _generator.GenerateKernel(fusedOp, options);

        Assert.IsNotNull(result.Parameters);
        Assert.IsNotEmpty(result.Parameters);
    }

    [Test]
    public void GenerateKernel_ComputesCompilationMetadata()
    {
        var fusedOp = CreateElementWiseFusedOperation();
        var options = new GenerationOptions();

        var result = _generator.GenerateKernel(fusedOp, options);

        Assert.Greater(result.Metadata.ThreadBlockSize, 0);
        Assert.Greater(result.Metadata.GridSize, 0);
        Assert.IsNotEmpty(result.Metadata.RequiredCapabilities);
    }

    #region Helper Methods

    private TemplateContext CreateContextWithFloat32()
    {
        return new TemplateContext
        {
            KernelName = "test_kernel",
            Parameters = new[]
            {
                new KernelParameter
                {
                    Name = "input",
                    Direction = ParameterDirection.Input,
                    DataType = DataType.Float32
                }
            },
            Nodes = Array.Empty<FusionOpNode>(),
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
            },
            Options = new GenerationOptions()
        };
    }

    private FusedOperation CreateElementWiseFusedOperation()
    {
        var ir = new FusionIR
        {
            Id = "test_elementwise",
            Nodes = new[]
            {
                new FusionOpNode
                {
                    Id = "node1",
                    OriginalOpType = "Add",
                    InputVars = new[] { "input1", "input2" },
                    OutputVar = "temp1",
                    Attributes = new Dictionary<string, object>()
                },
                new FusionOpNode
                {
                    Id = "node2",
                    OriginalOpType = "ReLU",
                    InputVars = new[] { "temp1" },
                    OutputVar = "output",
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
                    Location = MemoryLocation.Temporary
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
            KernelName = "elementwise_kernel",
            Strategy = FusionStrategy.Merge,
            InputTensors = ir.Variables.Where(v => v.Location == MemoryLocation.Input).ToList(),
            OutputTensors = ir.Variables.Where(v => v.Location == MemoryLocation.Output).ToList(),
            TemporaryMemoryBytes = 0,
            RegisterBytes = 512,
            ThreadBlockConfig = new ThreadBlockConfiguration { X = 256, Y = 1, Z = 1 },
            CompilationFlags = Array.Empty<string>(),
            Parameters = Array.Empty<MLFramework.Fusion.Backends.KernelLaunchParameter>()
        };

        var op1 = new TestOperation("op1", "Add", DataType.Float32);
        var op2 = new TestOperation("op2", "ReLU", DataType.Float32);

        return FusedOperation.Create(
            "test_op",
            new[] { op1, op2 },
            new FusionPatternDefinition { Name = "ElementWise" },
            ir,
            kernelSpec);
    }

    private FusedOperation CreateConvActivationFusedOperation()
    {
        var ir = new FusionIR
        {
            Id = "test_conv_relu",
            Nodes = new[]
            {
                new FusionOpNode
                {
                    Id = "node1",
                    OriginalOpType = "Conv2D",
                    InputVars = new[] { "input" },
                    OutputVar = "conv_output",
                    Attributes = new Dictionary<string, object>()
                },
                new FusionOpNode
                {
                    Id = "node2",
                    OriginalOpType = "ReLU",
                    InputVars = new[] { "conv_output" },
                    OutputVar = "output",
                    Attributes = new Dictionary<string, object>()
                }
            },
            Variables = new[]
            {
                new FusionVariable
                {
                    Name = "input",
                    Shape = new TensorShape(1, 3, 32, 32),
                    DataType = DataType.Float32,
                    Location = MemoryLocation.Input
                },
                new FusionVariable
                {
                    Name = "conv_output",
                    Shape = new TensorShape(1, 64, 30, 30),
                    DataType = DataType.Float32,
                    Location = MemoryLocation.Temporary
                },
                new FusionVariable
                {
                    Name = "output",
                    Shape = new TensorShape(1, 64, 30, 30),
                    DataType = DataType.Float32,
                    Location = MemoryLocation.Output
                }
            },
            MemoryLayout = new MemoryLayout
            {
                TensorLayout = TensorLayout.NCHW,
                SharedMemoryBytes = 1024,
                RegisterBytes = 1024
            },
            ComputeRequirements = new ComputeRequirements
            {
                ThreadBlocks = 64,
                ThreadsPerBlock = 128,
                RequiresSharedMemory = true,
                RequiresAtomicOps = false
            }
        };

        var kernelSpec = new KernelSpecification
        {
            KernelName = "conv_relu_kernel",
            Strategy = FusionStrategy.Merge,
            InputTensors = ir.Variables.Where(v => v.Location == MemoryLocation.Input).ToList(),
            OutputTensors = ir.Variables.Where(v => v.Location == MemoryLocation.Output).ToList(),
            TemporaryMemoryBytes = 1024,
            RegisterBytes = 1024,
            ThreadBlockConfig = new ThreadBlockConfiguration { X = 128, Y = 1, Z = 1 },
            CompilationFlags = Array.Empty<string>(),
            Parameters = Array.Empty<MLFramework.Fusion.Backends.KernelLaunchParameter>()
        };

        var op1 = new TestOperation("op1", "Conv2D", DataType.Float32);
        var op2 = new TestOperation("op2", "ReLU", DataType.Float32);

        return FusedOperation.Create(
            "test_op",
            new[] { op1, op2 },
            new FusionPatternDefinition { Name = "ConvActivation" },
            ir,
            kernelSpec);
    }

    private FusedOperation CreateUnsupportedFusedOperation()
    {
        var ir = new FusionIR
        {
            Id = "test_unsupported",
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
            CompilationFlags = Array.Empty<string>(),
            Parameters = Array.Empty<MLFramework.Fusion.Backends.KernelLaunchParameter>()
        };

        var op1 = new TestOperation("op1", "UnsupportedOp", DataType.Float32);

        return FusedOperation.Create(
            "test_op",
            new[] { op1 },
            new FusionPatternDefinition { Name = "TestPattern" },
            ir,
            kernelSpec);
    }

    private FusionIR CreateConvIR()
    {
        return new FusionIR
        {
            Id = "conv_ir",
            Nodes = new[]
            {
                new FusionOpNode
                {
                    Id = "node1",
                    OriginalOpType = "Conv2D",
                    InputVars = new[] { "input" },
                    OutputVar = "output",
                    Attributes = new Dictionary<string, object>()
                }
            },
            Variables = new[]
            {
                new FusionVariable
                {
                    Name = "input",
                    Shape = new TensorShape(1, 3, 32, 32),
                    DataType = DataType.Float32,
                    Location = MemoryLocation.Input
                },
                new FusionVariable
                {
                    Name = "output",
                    Shape = new TensorShape(1, 64, 30, 30),
                    DataType = DataType.Float32,
                    Location = MemoryLocation.Output
                }
            },
            MemoryLayout = new MemoryLayout
            {
                TensorLayout = TensorLayout.NCHW,
                SharedMemoryBytes = 0,
                RegisterBytes = 1024
            },
            ComputeRequirements = new ComputeRequirements
            {
                ThreadBlocks = 64,
                ThreadsPerBlock = 128,
                RequiresSharedMemory = true,
                RequiresAtomicOps = false
            }
        };
    }

    private FusionIR CreateElementWiseIR()
    {
        return new FusionIR
        {
            Id = "elementwise_ir",
            Nodes = new[]
            {
                new FusionOpNode
                {
                    Id = "node1",
                    OriginalOpType = "Add",
                    InputVars = new[] { "input1", "input2" },
                    OutputVar = "output",
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
    }

    private FusionOpNode CreateFusionOpNode(string opType, string[] inputVars, string outputVar)
    {
        return new FusionOpNode
        {
            Id = Guid.NewGuid().ToString(),
            OriginalOpType = opType,
            InputVars = inputVars,
            OutputVar = outputVar,
            Attributes = new Dictionary<string, object>()
        };
    }

    /// <summary>
    /// Test operation for unit tests
    /// </summary>
    private record TestOperation : Operation
    {
        public required string TestType { get; init; }

        public TestOperation(string id, string type, DataType dataType)
        {
            Id = id;
            Type = type;
            Name = type;
            DataType = dataType;
            Layout = TensorLayout.NCHW;
            InputShape = new TensorShape(32, 32);
            OutputShape = new TensorShape(32, 32);
            Inputs = Array.Empty<string>();
            Outputs = Array.Empty<string>();
            Attributes = new Dictionary<string, object>();
            TestType = type;
        }
    }

    #endregion
}
