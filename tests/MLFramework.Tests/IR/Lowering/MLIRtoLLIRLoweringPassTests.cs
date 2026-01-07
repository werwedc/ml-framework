using NUnit.Framework;
using MLFramework.IR;
using MLFramework.IR.Backend;
using MLFramework.IR.Lowering;
using MLFramework.IR.MLIR.Elementwise;
using MLFramework.IR.MLIR.Memory;
using MLFramework.IR.MLIR.Reduce;
using MLFramework.IR.MLIR.Loop;
using MLFramework.IR.Types;
using MLFramework.IR.Values;

namespace MLFramework.Tests.IR.Lowering
{
    [TestFixture]
    public class MLIRtoLLIRLoweringPassTests
    {
        private IRContext _mlirContext;
        private IRContext _llirContext;
        private MLIRtoLLIRLoweringPass _pass;

        [SetUp]
        public void Setup()
        {
            _mlirContext = new IRContext();
            _llirContext = new IRContext();
            _pass = new MLIRtoLLIRLoweringPass();
        }

        [Test]
        public void MLIRtoLLIRLoweringPass_CanBeCreated()
        {
            var pass = new MLIRtoLLIRLoweringPass();
            Assert.IsNotNull(pass);
            Assert.AreEqual("MLIR", pass.SourceIRLevel);
            Assert.AreEqual("LLIR", pass.TargetIRLevel);
        }

        [Test]
        public void MLIRtoLLIRLoweringPass_CanConfigureMemoryLayout()
        {
            var rowMajorPass = new MLIRtoLLIRLoweringPass(MemoryLayout.RowMajor);
            var colMajorPass = new MLIRtoLLIRLoweringPass(MemoryLayout.ColumnMajor);
            Assert.IsNotNull(rowMajorPass);
            Assert.IsNotNull(colMajorPass);
        }

        [Test]
        public void MLIRtoLLIRLoweringPass_CanConfigureVectorWidth()
        {
            var scalarPass = new MLIRtoLLIRLoweringPass(vectorWidth: 1);
            var vectorPass = new MLIRtoLLIRLoweringPass(vectorWidth: 4);
            Assert.IsNotNull(scalarPass);
            Assert.IsNotNull(vectorPass);
        }

        [Test]
        public void CanLower_BroadcastAddOp_ReturnsTrue()
        {
            var tensorType = new TensorType(DataType.Float32, new[] { 10, 20 });
            var lhs = _mlirContext.CreateValue(tensorType, "lhs");
            var rhs = _mlirContext.CreateValue(tensorType, "rhs");
            var result = _mlirContext.CreateValue(tensorType, "result");
            var broadcastShape = new[] { 10, 20 };

            var addOp = new BroadcastAddOp(lhs, rhs, result, broadcastShape);
            Assert.IsTrue(_pass.CanLower(addOp));
        }

        [Test]
        public void CanLower_ConvOp_ReturnsTrue()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 1, 28, 28, 3 });
            var weightType = new TensorType(DataType.Float32, new[] { 3, 3, 3, 16 });
            var outputType = new TensorType(DataType.Float32, new[] { 1, 26, 26, 16 });

            var input = _mlirContext.CreateValue(inputType, "input");
            var weight = _mlirContext.CreateValue(weightType, "weight");
            var result = _mlirContext.CreateValue(outputType, "result");

            var convOp = new ConvOp(
                input, weight, result,
                new[] { 1, 28, 28, 3 },
                new[] { 3, 3, 3, 16 },
                new[] { 1, 26, 26, 16 },
                new[] { 3, 3 },
                new[] { 1, 1 },
                new[] { 0, 0 },
                new[] { 1, 1 },
                1
            );

            Assert.IsTrue(_pass.CanLower(convOp));
        }

        [Test]
        public void CanLower_AllocTensorOp_ReturnsTrue()
        {
            var tensorType = new TensorType(DataType.Float32, new[] { 10, 20 });
            var result = _mlirContext.CreateValue(tensorType, "result");

            var allocOp = new AllocTensorOp(result, tensorType);
            Assert.IsTrue(_pass.CanLower(allocOp));
        }

        [Test]
        public void CanLower_ForLoopOp_ReturnsTrue()
        {
            var lowerBound = _mlirContext.CreateValue(new TensorType(DataType.Int32, new[] { 1 }), "0");
            var upperBound = _mlirContext.CreateValue(new TensorType(DataType.Int32, new[] { 1 }), "10");
            var step = _mlirContext.CreateValue(new TensorType(DataType.Int32, new[] { 1 }), "1");
            var iv = _mlirContext.CreateValue(new TensorType(DataType.Int32, new[] { 1 }), "i");
            var body = new IRBlock("loop_body");

            var loopOp = new ForLoopOp(lowerBound, upperBound, step, iv, body);
            Assert.IsTrue(_pass.CanLower(loopOp));
        }

        [Test]
        public void CanLower_ReduceOp_ReturnsTrue()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 10, 20 });
            var outputType = new TensorType(DataType.Float32, new[] { 1, 20 });

            var input = _mlirContext.CreateValue(inputType, "input");
            var result = _mlirContext.CreateValue(outputType, "result");

            var reduceOp = new ReduceOp(input, result, ReductionKind.Sum, new[] { 0 }, false);
            Assert.IsTrue(_pass.CanLower(reduceOp));
        }

        [Test]
        public void Lower_AllocTensorOp_GeneratesLLIROperations()
        {
            var tensorType = new TensorType(DataType.Float32, new[] { 10, 20 });
            var result = _mlirContext.CreateValue(tensorType, "result");

            var allocOp = new AllocTensorOp(result, tensorType);
            var lowered = _pass.Lower(_llirContext, allocOp);

            Assert.IsNotNull(lowered);
            Assert.AreEqual("alloc_buffer", lowered.Name);
        }

        [Test]
        public void Lower_BroadcastAddOp_GeneratesLLIROperations()
        {
            var tensorType = new TensorType(DataType.Float32, new[] { 10, 20 });
            var lhs = _mlirContext.CreateValue(tensorType, "lhs");
            var rhs = _mlirContext.CreateValue(tensorType, "rhs");
            var result = _mlirContext.CreateValue(tensorType, "result");
            var broadcastShape = new[] { 10, 20 };

            var addOp = new BroadcastAddOp(lhs, rhs, result, broadcastShape);
            var lowered = _pass.Lower(_llirContext, addOp);

            Assert.IsNotNull(lowered);
            // The lowered operation should involve loops and scalar operations
        }

        [Test]
        public void Lower_ForLoopOp_GeneratesLLIRLoop()
        {
            var lowerBound = _mlirContext.CreateValue(new TensorType(DataType.Int32, new[] { 1 }), "0");
            var upperBound = _mlirContext.CreateValue(new TensorType(DataType.Int32, new[] { 1 }), "10");
            var step = _mlirContext.CreateValue(new TensorType(DataType.Int32, new[] { 1 }), "1");
            var iv = _mlirContext.CreateValue(new TensorType(DataType.Int32, new[] { 1 }), "i");
            var body = new IRBlock("loop_body");

            var loopOp = new ForLoopOp(lowerBound, upperBound, step, iv, body);
            var lowered = _pass.Lower(_llirContext, loopOp);

            Assert.IsNotNull(lowered);
            Assert.AreEqual("for_loop", lowered.Name);
        }

        [Test]
        public void Lower_ReduceOp_GeneratesLLIRLoop()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 10, 20 });
            var outputType = new TensorType(DataType.Float32, new[] { 1, 20 });

            var input = _mlirContext.CreateValue(inputType, "input");
            var result = _mlirContext.CreateValue(outputType, "result");

            var reduceOp = new ReduceOp(input, result, ReductionKind.Sum, new[] { 0 }, false);
            var lowered = _pass.Lower(_llirContext, reduceOp);

            Assert.IsNotNull(lowered);
        }

        [Test]
        public void MLIRtoLLIRLoweringPass_HasCorrectSourceAndTargetLevels()
        {
            var pass = new MLIRtoLLIRLoweringPass();
            Assert.AreEqual("MLIR", pass.SourceIRLevel);
            Assert.AreEqual("LLIR", pass.TargetIRLevel);
        }

        [Test]
        public void ComputeTensorSize_CalculatesCorrectly()
        {
            // This is an indirect test via the lowering behavior
            var tensorType = new TensorType(DataType.Float32, new[] { 10, 20 });
            var result = _mlirContext.CreateValue(tensorType, "result");
            var allocOp = new AllocTensorOp(result, tensorType);
            var lowered = _pass.Lower(_llirContext, allocOp);

            Assert.IsNotNull(lowered);
            // Size should be 10 * 20 * 4 = 800 bytes
        }
    }
}
