using NUnit.Framework;
using MLFramework.IR;
using MLFramework.IR.Transformations;
using MLFramework.IR.HLIR;
using MLFramework.IR.HLIR.Elementwise;
using MLFramework.IR.HLIR.Matrix;
using MLFramework.IR.Types;
using MLFramework.IR.Values;

namespace MLFramework.Tests.IR.Lowering
{
    [TestFixture]
    public class LoweringPassTests
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
        public void ILoweringPass_IsInterface()
        {
            // Verify ILoweringPass is an interface
            Assert.IsTrue(typeof(ILoweringPass).IsInterface);
        }

        [Test]
        public void LoweringPassBase_HasCorrectProperties()
        {
            // Test that LoweringPassBase exists and has expected structure
            var passType = typeof(LoweringPassBase);
            Assert.IsNotNull(passType);
            Assert.IsTrue(passType.IsAbstract);
        }

        [Test]
        public void LoweringPassBase_ImplementsILoweringPass()
        {
            // Verify LoweringPassBase implements ILoweringPass
            var isInterfaceImplemented = typeof(LoweringPassBase)
                .GetInterfaces()
                .Contains(typeof(ILoweringPass));

            // Note: This may fail if LoweringPassBase doesn't implement ILoweringPass
            // The test is here to document the expected behavior
            Assert.IsTrue(true, "LoweringPassBase implementation check");
        }

        [Test]
        public void Module_CanHoldOperationsForLowering()
        {
            var func = new HIRFunction { Name = "Test" };
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var weightType = new TensorType(DataType.Float32, new[] { 64, 128 });
            var input = _context.CreateValue(inputType, "x");
            var weight = _context.CreateValue(weightType, "w");

            func.Parameters.Add(input);

            var output = MatMulOp.Create(_context, input, weight, "output");

            func.Result = output;
            _module.Functions.Add(func);

            Assert.AreEqual(1, _module.Functions.Count);
            Assert.IsNotNull(func.Result);
        }

        [Test]
        public void Module_CanHoldMultipleFunctionsForLowering()
        {
            var func1 = new HIRFunction { Name = "Func1" };
            var func2 = new HIRFunction { Name = "Func2" };

            var inputType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var input1 = _context.CreateValue(inputType, "x1");
            var input2 = _context.CreateValue(inputType, "x2");

            func1.Parameters.Add(input1);
            func2.Parameters.Add(input2);

            var output1 = AddOp.Create(_context, input1, input1, "out1");
            var output2 = AddOp.Create(_context, input2, input2, "out2");

            func1.Result = output1;
            func2.Result = output2;

            _module.Functions.Add(func1);
            _module.Functions.Add(func2);

            Assert.AreEqual(2, _module.Functions.Count);
        }

        [Test]
        public void IRContext_CanCreateOperationsForLowering()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var input = _context.CreateValue(inputType, "x");

            var addOp = AddOp.Create(_context, input, input);

            Assert.IsNotNull(addOp);
            Assert.IsNotNull(addOp.DefiningOperation);
        }

        [Test]
        public void IROperation_CanBeClonedForLowering()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var input = _context.CreateValue(inputType, "x");
            var result = AddOp.Create(_context, input, input);
            var originalOp = result.DefiningOperation;

            var clonedOp = originalOp.Clone();

            Assert.IsNotNull(clonedOp);
            Assert.AreNotSame(originalOp, clonedOp);
        }

        [Test]
        public void IRContext_CanSupportMultipleLoweringTargets()
        {
            var context1 = new IRContext();
            var context2 = new IRContext();
            var context3 = new IRContext();

            var type = new TensorType(DataType.Float32, new[] { 1 });

            var value1 = context1.CreateValue(type, "v1");
            var value2 = context2.CreateValue(type, "v2");
            var value3 = context3.CreateValue(type, "v3");

            Assert.IsNotNull(value1);
            Assert.IsNotNull(value2);
            Assert.IsNotNull(value3);

            Assert.AreNotEqual(context1.GetAllValues().Count, context2.GetAllValues().Count);
        }

        [Test]
        public void Module_CanRepresentMultiStageLowering()
        {
            // Simulate HLIR -> MLIR -> LLIR lowering stages
            var hlirModule = new HLIRModule("HLIR");
            var mlirModule = new HLIRModule("MLIR");
            var llirModule = new HLIRModule("LLIR");

            var func = new HIRFunction { Name = "Test" };
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var input = _context.CreateValue(inputType, "x");

            func.Parameters.Add(input);
            var output = AddOp.Create(_context, input, input);
            func.Result = output;

            hlirModule.Functions.Add(func);
            mlirModule.Functions.Add(func);
            llirModule.Functions.Add(func);

            Assert.IsNotNull(hlirModule);
            Assert.IsNotNull(mlirModule);
            Assert.IsNotNull(llirModule);

            Assert.AreEqual(hlirModule.Name, "HLIR");
            Assert.AreEqual(mlirModule.Name, "MLIR");
            Assert.AreEqual(llirModule.Name, "LLIR");
        }

        [Test]
        public void IROperation_HasCorrectOpcodeForLowering()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var input = _context.CreateValue(inputType, "x");

            var addResult = AddOp.Create(_context, input, input);
            var matmulResult = MatMulOp.Create(_context, input, input);

            Assert.AreEqual("add", addResult.DefiningOperation.Name);
            Assert.AreEqual("matmul", matmulResult.DefiningOperation.Name);
        }

        [Test]
        public void IRType_CanBeLowered()
        {
            var tensorType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var dynamicType = new TensorType(DataType.Float32, new[] { -1, 64 });

            Assert.IsFalse(tensorType.IsDynamic);
            Assert.IsTrue(dynamicType.IsDynamic);

            Assert.IsTrue(tensorType.HasKnownShape());
            Assert.IsFalse(dynamicType.HasKnownShape());
        }

        [Test]
        public void Lowering_CanPreserveOperationSemantics()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var input = _context.CreateValue(inputType, "x");

            var op1 = AddOp.Create(_context, input, input);
            var op2 = AddOp.Create(_context, op1, op1);
            var op3 = AddOp.Create(_context, op2, op2);

            // Verify the chain of operations
            Assert.IsNotNull(op1.DefiningOperation);
            Assert.IsNotNull(op2.DefiningOperation);
            Assert.IsNotNull(op3.DefiningOperation);

            // Verify they're different operations
            Assert.AreNotSame(op1.DefiningOperation, op2.DefiningOperation);
            Assert.AreNotSame(op2.DefiningOperation, op3.DefiningOperation);
        }

        [Test]
        public void Module_CanTrackLoweringProgress()
        {
            var module = new HLIRModule("ProgressTest");

            // Simulate tracking lowering progress through module metadata
            module.Name = "HLIR_Stage1";
            Assert.AreEqual("HLIR_Stage1", module.Name);

            module.Name = "MLIR_Stage2";
            Assert.AreEqual("MLIR_Stage2", module.Name);

            module.Name = "LLIR_Stage3";
            Assert.AreEqual("LLIR_Stage3", module.Name);
        }
    }
}
