using System;
using System.Collections.Generic;
using Xunit;
using MLFramework.IR;
using MLFramework.IR.Backend;
using MLFramework.IR.Backend.CPU;
using MLFramework.IR.Backend.GPU;

namespace MLFramework.Tests.IR.Backend
{
    public class BackendAbstractionTests
    {
        [Fact]
        public void BackendRegistry_Singleton_ShouldBeSameInstance()
        {
            // Act & Assert
            var instance1 = BackendRegistry.Instance;
            var instance2 = BackendRegistry.Instance;
            Assert.Same(instance1, instance2);
        }

        [Fact]
        public void BackendRegistry_RegisterAndRetrieve_ShouldWork()
        {
            // Arrange
            var registry = BackendRegistry.Instance;
            registry.Clear();
            var backend = new X86CPUBackend();

            // Act
            registry.RegisterBackend(backend);
            var retrievedBackend = registry.GetBackend("x86_64");

            // Assert
            Assert.NotNull(retrievedBackend);
            Assert.Same(backend, retrievedBackend);
        }

        [Fact]
        public void BackendRegistry_HasBackend_ShouldReturnTrueAfterRegistration()
        {
            // Arrange
            var registry = BackendRegistry.Instance;
            registry.Clear();
            var backend = new X86CPUBackend();

            // Act
            registry.RegisterBackend(backend);
            var hasBackend = registry.HasBackend("x86_64");

            // Assert
            Assert.True(hasBackend);
        }

        [Fact]
        public void BackendRegistry_GetAllBackends_ShouldReturnAllRegisteredBackends()
        {
            // Arrange
            var registry = BackendRegistry.Instance;
            registry.Clear();
            var cpuBackend = new X86CPUBackend();
            var cudaBackend = new CUDABackend();

            // Act
            registry.RegisterBackend(cpuBackend);
            registry.RegisterBackend(cudaBackend);
            var allBackends = registry.GetAllBackends();

            // Assert
            Assert.Equal(2, allBackends.Count());
        }

        [Fact]
        public void BackendRegistry_UnregisterBackend_ShouldRemoveBackend()
        {
            // Arrange
            var registry = BackendRegistry.Instance;
            registry.Clear();
            var backend = new X86CPUBackend();
            registry.RegisterBackend(backend);

            // Act
            var removed = registry.UnregisterBackend("x86_64");

            // Assert
            Assert.True(removed);
            Assert.Null(registry.GetBackend("x86_64"));
        }

        [Fact]
        public void X86CPUBackend_Properties_ShouldBeCorrect()
        {
            // Arrange
            var backend = new X86CPUBackend();

            // Act & Assert
            Assert.Equal("x86_64", backend.Name);
            Assert.Equal("x86_64-unknown-linux-gnu", backend.TargetTriple);
        }

        [Fact]
        public void X86CPUBackend_CanCompile_ShouldReturnTrue()
        {
            // Arrange
            var backend = new X86CPUBackend();
            var module = new HLIRModule("test_module");

            // Act
            var canCompile = backend.CanCompile(module);

            // Assert
            Assert.True(canCompile);
        }

        [Fact]
        public void X86CPUBackend_LowerToBackendIR_ShouldReturnModule()
        {
            // Arrange
            var backend = new X86CPUBackend();
            var module = new HLIRModule("test_module");

            // Act
            var loweredModule = backend.LowerToBackendIR(module);

            // Assert
            Assert.NotNull(loweredModule);
            Assert.Same(module, loweredModule);
        }

        [Fact]
        public void X86CPUBackend_GenerateCode_ShouldReturnCode()
        {
            // Arrange
            var backend = new X86CPUBackend();
            var module = new HLIRModule("test_module");
            module.Operations.Add(new HIROperation("add", new List<HIRValue>(), "test_add"));

            // Act
            var code = backend.GenerateCode(module);

            // Assert
            Assert.NotNull(code);
            Assert.Contains("// Generated code for x86_64 backend", code);
        }

        [Fact]
        public void X86CPUBackend_GenerateBinary_ShouldThrowNotImplemented()
        {
            // Arrange
            var backend = new X86CPUBackend();
            var module = new HLIRModule("test_module");

            // Act & Assert
            Assert.Throws<NotImplementedException>(() => backend.GenerateBinary(module));
        }

        [Fact]
        public void CUDABackend_Properties_ShouldBeCorrect()
        {
            // Arrange
            var backend = new CUDABackend(ComputeCapability.SM_80);

            // Act & Assert
            Assert.Equal("CUDA", backend.Name);
            Assert.Equal("nvptx64-nvidia-cuda", backend.TargetTriple);
            Assert.Equal(ComputeCapability.SM_80, backend.ComputeCapability);
        }

        [Fact]
        public void CUDABackend_CanCompile_ShouldReturnTrue()
        {
            // Arrange
            var backend = new CUDABackend();
            var module = new HLIRModule("test_module");

            // Act
            var canCompile = backend.CanCompile(module);

            // Assert
            Assert.True(canCompile);
        }

        [Fact]
        public void CUDABackend_GenerateCode_ShouldReturnPTXCode()
        {
            // Arrange
            var backend = new CUDABackend();
            var module = new HLIRModule("test_module");
            module.Operations.Add(new HIROperation("matmul", new List<HIRValue>(), "test_matmul"));

            // Act
            var code = backend.GenerateCode(module);

            // Assert
            Assert.NotNull(code);
            Assert.Contains("// Generated PTX for CUDA backend", code);
        }

        [Fact]
        public void CUDABackend_GenerateBinary_ShouldThrowNotImplemented()
        {
            // Arrange
            var backend = new CUDABackend();
            var module = new HLIRModule("test_module");

            // Act & Assert
            Assert.Throws<NotImplementedException>(() => backend.GenerateBinary(module));
        }

        [Fact]
        public void CompilationOptions_DefaultValues_ShouldBeCorrect()
        {
            // Arrange & Act
            var options = new CompilationOptions();

            // Assert
            Assert.Equal(OptimizationLevel.Standard, options.OptimizationLevel);
            Assert.False(options.DebugSymbols);
            Assert.False(options.Verbose);
            Assert.Equal(-1, options.VectorWidth);
            Assert.Equal(MemoryLayout.RowMajor, options.PreferredMemoryLayout);
        }

        [Fact]
        public void CompilationResult_Success_ShouldBeTrueByDefault()
        {
            // Arrange & Act
            var result = new CompilationResult();

            // Assert
            Assert.True(result.Success);
            Assert.Null(result.ErrorMessage);
        }

        [Fact]
        public void BackendCompiler_Compile_ShouldSucceed()
        {
            // Arrange
            var registry = BackendRegistry.Instance;
            registry.Clear();
            var backend = new X86CPUBackend();
            registry.RegisterBackend(backend);

            var module = new HLIRModule("test_module");
            module.Operations.Add(new HIROperation("add", new List<HIRValue>(), "test_add"));

            var options = new CompilationOptions
            {
                OptimizationLevel = OptimizationLevel.Basic,
                Verbose = false
            };

            // Act
            var result = BackendCompiler.Compile(module, "x86_64", options);

            // Assert
            Assert.True(result.Success);
            Assert.NotNull(result.Code);
            Assert.Contains("// Generated code for x86_64 backend", result.Code);
        }

        [Fact]
        public void BackendCompiler_Compile_WithInvalidBackend_ShouldThrow()
        {
            // Arrange
            var registry = BackendRegistry.Instance;
            registry.Clear();

            var module = new HLIRModule("test_module");

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                BackendCompiler.Compile(module, "nonexistent_backend"));
        }

        [Fact]
        public void BackendCompiler_Compile_WithNullModule_ShouldThrow()
        {
            // Arrange
            var registry = BackendRegistry.Instance;
            registry.Clear();
            var backend = new X86CPUBackend();
            registry.RegisterBackend(backend);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                BackendCompiler.Compile(null, "x86_64"));
        }

        [Fact]
        public void X86CPUBackend_Compile_ShouldNotThrow()
        {
            // Arrange
            var backend = new X86CPUBackend();
            var module = new HLIRModule("test_module");
            module.Operations.Add(new HIROperation("mul", new List<HIRValue>(), "test_mul"));
            var options = new CompilationOptions();

            // Act & Assert
            backend.Compile(module, options);
        }

        [Fact]
        public void CUDABackend_Compile_ShouldNotThrow()
        {
            // Arrange
            var backend = new CUDABackend();
            var module = new HLIRModule("test_module");
            module.Operations.Add(new HIROperation("add", new List<HIRValue>(), "test_add"));
            var options = new CompilationOptions();

            // Act & Assert
            backend.Compile(module, options);
        }

        [Fact]
        public void IRPassManager_AddPass_ShouldAddPass()
        {
            // Arrange
            var passManager = new IRPassManager();
            var pass = new ConstantFoldingPass();

            // Act
            passManager.AddPass(pass, IRPassManager.PassType.Optimization);

            // Assert
            var passes = passManager.GetPasses(IRPassManager.PassType.Optimization);
            Assert.Single(passes);
            Assert.Same(pass, passes[0]);
        }

        [Fact]
        public void IRPassManager_RunAll_ShouldNotThrow()
        {
            // Arrange
            var passManager = new IRPassManager();
            var pass = new ConstantFoldingPass();
            passManager.AddPass(pass, IRPassManager.PassType.Optimization);
            var module = new HLIRModule("test_module");

            // Act & Assert
            passManager.RunAll(module);
        }

        [Fact]
        public void CPUArchitecture_Enum_ShouldHaveCorrectValues()
        {
            // Act & Assert
            Assert.Equal(0, (int)CPUArchitecture.X32);
            Assert.Equal(1, (int)CPUArchitecture.X64);
        }

        [Fact]
        public void ComputeCapability_Enum_ShouldHaveCorrectValues()
        {
            // Act & Assert
            Assert.Equal(0, (int)ComputeCapability.SM_70);
            Assert.Equal(1, (int)ComputeCapability.SM_75);
            Assert.Equal(2, (int)ComputeCapability.SM_80);
            Assert.Equal(3, (int)ComputeCapability.SM_90);
        }

        [Fact]
        public void MemoryLayout_Enum_ShouldHaveCorrectValues()
        {
            // Act & Assert
            Assert.Equal(0, (int)MemoryLayout.RowMajor);
            Assert.Equal(1, (int)MemoryLayout.ColumnMajor);
        }
    }
}
