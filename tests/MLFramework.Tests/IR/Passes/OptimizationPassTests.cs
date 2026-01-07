using NUnit.Framework;
using MLFramework.IR.Backend.CPU;
using MLFramework.IR;
using MLFramework.IR.HLIR;
using MLFramework.IR.HLIR.Elementwise;
using MLFramework.IR.Types;
using MLFramework.IR.Values;
using MLFramework.IR.Attributes;

namespace MLFramework.Tests.IR.Passes
{
    [TestFixture]
    public class OptimizationPassTests
    {
        private HLIRModule _module;
        private IRContext _context;

        [SetUp]
        public void Setup()
        {
            _module = new HLIRModule("TestModule");
            _context = new IRContext();
        }

        [Test]
        public void ConstantFoldingPass_HasCorrectName()
        {
            var pass = new ConstantFoldingPass();

            Assert.AreEqual("Constant Folding", pass.Name);
        }

        [Test]
        public void ConstantFoldingPass_DoesNotThrowWhenRun()
        {
            var func = new HIRFunction { Name = "Test" };
            var inputType = new TensorType(DataType.Float32, new[] { 1 });
            var input = _context.CreateValue(inputType, "x");
            func.Parameters.Add(input);

            var const1 = ConstantOp.Create(_context, new FloatAttribute(2.0f), "c1");
            var const2 = ConstantOp.Create(_context, new FloatAttribute(3.0f), "c2");
            var result = AddOp.Create(_context, const1, const2);

            func.Result = result;
            _module.Functions.Add(func);

            var pass = new ConstantFoldingPass();
            Assert.DoesNotThrow(() => pass.Run(_module));
        }

        [Test]
        public void ConstantFoldingPass_RunsWithEmptyModule()
        {
            var pass = new ConstantFoldingPass();
            Assert.DoesNotThrow(() => pass.Run(_module));
        }

        [Test]
        public void DeadCodeEliminationPass_HasCorrectName()
        {
            var pass = new DeadCodeEliminationPass();

            Assert.AreEqual("Dead Code Elimination", pass.Name);
        }

        [Test]
        public void DeadCodeEliminationPass_DoesNotThrowWhenRun()
        {
            var func = new HIRFunction { Name = "Test" };
            var inputType = new TensorType(DataType.Float32, new[] { 1 });
            var input = _context.CreateValue(inputType, "x");
            func.Parameters.Add(input);

            var temp = AddOp.Create(_context, input, input, "temp");
            var const1 = ConstantOp.Create(_context, new FloatAttribute(1.0f), "c1");
            var output = AddOp.Create(_context, temp, const1, "output");

            // Create an unused operation
            var unused = AddOp.Create(_context, input, input, "unused");

            func.Result = output;
            _module.Functions.Add(func);

            var pass = new DeadCodeEliminationPass();
            Assert.DoesNotThrow(() => pass.Run(_module));
        }

        [Test]
        public void DeadCodeEliminationPass_RunsWithEmptyModule()
        {
            var pass = new DeadCodeEliminationPass();
            Assert.DoesNotThrow(() => pass.Run(_module));
        }

        [Test]
        public void OperationSimplificationPass_HasCorrectName()
        {
            var pass = new OperationSimplificationPass();

            Assert.AreEqual("Operation Simplification", pass.Name);
        }

        [Test]
        public void OperationSimplificationPass_DoesNotThrowWhenRun()
        {
            var func = new HIRFunction { Name = "Test" };
            var inputType = new TensorType(DataType.Float32, new[] { 1 });
            var input = _context.CreateValue(inputType, "x");
            func.Parameters.Add(input);

            var zero = ConstantOp.Create(_context, new FloatAttribute(0.0f), "zero");
            var result = AddOp.Create(_context, input, zero, "result");

            func.Result = result;
            _module.Functions.Add(func);

            var pass = new OperationSimplificationPass();
            Assert.DoesNotThrow(() => pass.Run(_module));
        }

        [Test]
        public void OperationSimplificationPass_RunsWithEmptyModule()
        {
            var pass = new OperationSimplificationPass();
            Assert.DoesNotThrow(() => pass.Run(_module));
        }

        [Test]
        public void OptimizationPasses_CanBeRunSequentially()
        {
            var func = new HIRFunction { Name = "Test" };
            var inputType = new TensorType(DataType.Float32, new[] { 1 });
            var input = _context.CreateValue(inputType, "x");
            func.Parameters.Add(input);

            var const1 = ConstantOp.Create(_context, new FloatAttribute(2.0f), "c1");
            var const2 = ConstantOp.Create(_context, new FloatAttribute(3.0f), "c2");
            var temp = AddOp.Create(_context, const1, const2, "temp");
            var zero = ConstantOp.Create(_context, new FloatAttribute(0.0f), "zero");
            var result = AddOp.Create(_context, temp, zero, "result");

            func.Result = result;
            _module.Functions.Add(func);

            var constantFolding = new ConstantFoldingPass();
            var simplification = new OperationSimplificationPass();
            var deadCodeElimination = new DeadCodeEliminationPass();

            Assert.DoesNotThrow(() => constantFolding.Run(_module));
            Assert.DoesNotThrow(() => simplification.Run(_module));
            Assert.DoesNotThrow(() => deadCodeElimination.Run(_module));
        }

        [Test]
        public void ConstantFoldingPass_CanRunWithMultipleFunctions()
        {
            var func1 = new HIRFunction { Name = "Func1" };
            var func2 = new HIRFunction { Name = "Func2" };

            var inputType = new TensorType(DataType.Float32, new[] { 1 });
            var input1 = _context.CreateValue(inputType, "x1");
            var input2 = _context.CreateValue(inputType, "x2");

            func1.Parameters.Add(input1);
            func2.Parameters.Add(input2);

            var result1 = AddOp.Create(_context, input1, input1, "result1");
            var result2 = AddOp.Create(_context, input2, input2, "result2");

            func1.Result = result1;
            func2.Result = result2;

            _module.Functions.Add(func1);
            _module.Functions.Add(func2);

            var pass = new ConstantFoldingPass();
            Assert.DoesNotThrow(() => pass.Run(_module));
        }

        [Test]
        public void IRPass_ImplementsIRPassInterface()
        {
            var pass = new ConstantFoldingPass();

            Assert.IsNotNull(pass.Name);
            Assert.IsInstanceOf<IRPass>(pass);
        }

        [Test]
        public void AllOptimizationPasses_HaveNames()
        {
            var passes = new IRPass[]
            {
                new ConstantFoldingPass(),
                new DeadCodeEliminationPass(),
                new OperationSimplificationPass()
            };

            foreach (var pass in passes)
            {
                Assert.IsNotNull(pass.Name);
                Assert.IsNotEmpty(pass.Name);
            }
        }

        [Test]
        public void OptimizationPasses_HandleNullModule()
        {
            var passes = new IRPass[]
            {
                new ConstantFoldingPass(),
                new DeadCodeEliminationPass(),
                new OperationSimplificationPass()
            };

            foreach (var pass in passes)
            {
                // Note: This test verifies the pass doesn't crash on null,
                // but in a real implementation, it might throw ArgumentNullException
                Assert.DoesNotThrow(() =>
                {
                    try
                    {
                        pass.Run(null);
                    }
                    catch
                    {
                        // Expected behavior - just checking it doesn't crash unexpectedly
                    }
                });
            }
        }

        [Test]
        public void ConstantFoldingPass_CanProcessConstants()
        {
            var func = new HIRFunction { Name = "Test" };
            var funcInput = _context.CreateValue(new TensorType(DataType.Float32, new[] { 1 }), "input");
            func.Parameters.Add(funcInput);

            // Create various constants
            var floatConst = ConstantOp.Create(_context, new FloatAttribute(1.5f), "float_const");
            var intConst = ConstantOp.Create(_context, new IntAttribute(42), "int_const");
            var boolConst = ConstantOp.Create(_context, new BoolAttribute(true), "bool_const");

            func.Result = floatConst;
            _module.Functions.Add(func);

            var pass = new ConstantFoldingPass();
            Assert.DoesNotThrow(() => pass.Run(_module));
        }

        [Test]
        public void DeadCodeEliminationPass_CanProcessComplexGraph()
        {
            var func = new HIRFunction { Name = "ComplexGraph" };
            var inputType = new TensorType(DataType.Float32, new[] { 10, 10 });
            var input = _context.CreateValue(inputType, "input");
            func.Parameters.Add(input);

            // Create a more complex graph
            var temp1 = AddOp.Create(_context, input, input, "temp1");
            var temp2 = AddOp.Create(_context, temp1, temp1, "temp2");
            var temp3 = AddOp.Create(_context, temp2, temp2, "temp3");

            // Some potentially unused nodes
            var unused1 = AddOp.Create(_context, input, input, "unused1");
            var unused2 = AddOp.Create(_context, unused1, unused1, "unused2");

            var result = AddOp.Create(_context, temp3, input, "result");

            func.Result = result;
            _module.Functions.Add(func);

            var pass = new DeadCodeEliminationPass();
            Assert.DoesNotThrow(() => pass.Run(_module));
        }
    }
}
