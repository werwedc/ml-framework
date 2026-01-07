using NUnit.Framework;
using MLFramework.IR;
using MLFramework.IR.Backend;
using MLFramework.IR.LLIR;
using MLFramework.IR.LLIR.Operations;
using MLFramework.IR.LLIR.Operations.Arithmetic;
using MLFramework.IR.LLIR.Operations.ControlFlow;
using MLFramework.IR.LLIR.Operations.Memory;
using MLFramework.IR.LLIR.Operations.Vector;
using MLFramework.IR.LLIR.Types;
using MLFramework.IR.LLIR.Values;
using MLFramework.IR.Types;

namespace MLFramework.Tests.IR.LLIR
{
    [TestFixture]
    public class LLIRFoundationTests
    {
        private IRContext _context;

        [SetUp]
        public void Setup()
        {
            _context = new IRContext();
        }

        #region Memory Operations Tests

        [Test]
        public void FreeBufferOp_CreatesCorrectly()
        {
            var pointerType = new PointerType(new ScalarType(DataType.Float32));
            var buffer = new MemoryValue(pointerType, "buffer", 0, 1024);

            var freeOp = new FreeBufferOp(buffer);

            Assert.IsNotNull(freeOp);
            Assert.AreEqual(buffer, freeOp.Buffer);
            Assert.AreEqual("free_buffer", freeOp.Name);
            Assert.AreEqual(IROpcode.FreeBuffer, freeOp.Opcode);
        }

        [Test]
        public void FreeBufferOp_Validate_DoesNotThrowForMemoryLocation()
        {
            var pointerType = new PointerType(new ScalarType(DataType.Float32));
            var buffer = new MemoryValue(pointerType, "buffer", 0, 1024);

            var freeOp = new FreeBufferOp(buffer);

            Assert.DoesNotThrow(() => freeOp.Validate());
        }

        [Test]
        public void FreeBufferOp_Validate_ThrowsForRegister()
        {
            var scalarType = new ScalarType(DataType.Float32);
            var register = new RegisterValue(scalarType, "r0");

            var freeOp = new FreeBufferOp(register);

            Assert.Throws<System.InvalidOperationException>(() => freeOp.Validate());
        }

        [Test]
        public void MemcpyOp_CreatesCorrectly()
        {
            var pointerType = new PointerType(new ScalarType(DataType.Float32));
            var dest = new MemoryValue(pointerType, "dest", 0, 1024);
            var src = new MemoryValue(pointerType, "src", 0, 1024);

            var memcpyOp = new MemcpyOp(dest, src, 1024);

            Assert.IsNotNull(memcpyOp);
            Assert.AreEqual(dest, memcpyOp.Dest);
            Assert.AreEqual(src, memcpyOp.Src);
            Assert.AreEqual(1024, memcpyOp.SizeInBytes);
        }

        [Test]
        public void MemcpyOp_Clone_CreatesIndependentCopy()
        {
            var pointerType = new PointerType(new ScalarType(DataType.Float32));
            var dest = new MemoryValue(pointerType, "dest", 0, 1024);
            var src = new MemoryValue(pointerType, "src", 0, 1024);

            var original = new MemcpyOp(dest, src, 1024);
            var cloned = (MemcpyOp)original.Clone();

            Assert.AreEqual(original.Dest, cloned.Dest);
            Assert.AreEqual(original.Src, cloned.Src);
            Assert.AreEqual(original.SizeInBytes, cloned.SizeInBytes);
        }

        #endregion

        #region Arithmetic Operations Tests

        [Test]
        public void SubScalarOp_CreatesCorrectly()
        {
            var scalarType = new ScalarType(DataType.Float32);
            var lhs = new RegisterValue(scalarType, "lhs");
            var rhs = new RegisterValue(scalarType, "rhs");
            var result = new RegisterValue(scalarType, "result");

            var subOp = new SubScalarOp(lhs, rhs, result);

            Assert.IsNotNull(subOp);
            Assert.AreEqual(lhs, subOp.Lhs);
            Assert.AreEqual(rhs, subOp.Rhs);
            Assert.AreEqual(result, subOp.Result);
        }

        [Test]
        public void DivScalarOp_CreatesCorrectly()
        {
            var scalarType = new ScalarType(DataType.Float32);
            var lhs = new RegisterValue(scalarType, "dividend");
            var rhs = new RegisterValue(scalarType, "divisor");
            var result = new RegisterValue(scalarType, "result");

            var divOp = new DivScalarOp(lhs, rhs, result);

            Assert.IsNotNull(divOp);
            Assert.AreEqual(lhs, divOp.Lhs);
            Assert.AreEqual(rhs, divOp.Rhs);
            Assert.AreEqual(result, divOp.Result);
        }

        [Test]
        public void VectorAddOp_CreatesCorrectly()
        {
            var scalarType = new ScalarType(DataType.Float32);
            var vectorType = new VectorType(scalarType, 4);
            var lhs = new RegisterValue(vectorType, "v_lhs");
            var rhs = new RegisterValue(vectorType, "v_rhs");
            var result = new RegisterValue(vectorType, "v_result");

            var vectorAddOp = new VectorAddOp(lhs, rhs, result, 4);

            Assert.IsNotNull(vectorAddOp);
            Assert.AreEqual(lhs, vectorAddOp.Lhs);
            Assert.AreEqual(rhs, vectorAddOp.Rhs);
            Assert.AreEqual(result, vectorAddOp.Result);
            Assert.AreEqual(4, vectorAddOp.VectorWidth);
        }

        [Test]
        public void VectorMulOp_CreatesCorrectly()
        {
            var scalarType = new ScalarType(DataType.Float32);
            var vectorType = new VectorType(scalarType, 8);
            var lhs = new RegisterValue(vectorType, "v_lhs");
            var rhs = new RegisterValue(vectorType, "v_rhs");
            var result = new RegisterValue(vectorType, "v_result");

            var vectorMulOp = new VectorMulOp(lhs, rhs, result, 8);

            Assert.IsNotNull(vectorMulOp);
            Assert.AreEqual(lhs, vectorMulOp.Lhs);
            Assert.AreEqual(rhs, vectorMulOp.Rhs);
            Assert.AreEqual(result, vectorMulOp.Result);
            Assert.AreEqual(8, vectorMulOp.VectorWidth);
        }

        [Test]
        public void VectorAddOp_ThrowsForInvalidVectorWidth()
        {
            var scalarType = new ScalarType(DataType.Float32);
            var vectorType = new VectorType(scalarType, 4);
            var lhs = new RegisterValue(vectorType, "v_lhs");
            var rhs = new RegisterValue(vectorType, "v_rhs");
            var result = new RegisterValue(vectorType, "v_result");

            Assert.Throws<System.ArgumentOutOfRangeException>(() =>
                new VectorAddOp(lhs, rhs, result, 0));
        }

        #endregion

        #region Control Flow Tests

        [Test]
        public void BranchOp_CreatesCorrectly()
        {
            var targetBlock = new IRBlock("target");

            var branchOp = new BranchOp(targetBlock);

            Assert.IsNotNull(branchOp);
            Assert.AreEqual(targetBlock, branchOp.Target);
            Assert.AreEqual("branch", branchOp.Name);
            Assert.AreEqual(IROpcode.Branch, branchOp.Opcode);
        }

        [Test]
        public void ConditionalBranchOp_CreatesCorrectly()
        {
            var boolType = new ScalarType(DataType.Bool);
            var condition = new RegisterValue(boolType, "cond");
            var trueBlock = new IRBlock("true_block");
            var falseBlock = new IRBlock("false_block");

            var condBranchOp = new ConditionalBranchOp(condition, trueBlock, falseBlock);

            Assert.IsNotNull(condBranchOp);
            Assert.AreEqual(condition, condBranchOp.Condition);
            Assert.AreEqual(trueBlock, condBranchOp.TrueTarget);
            Assert.AreEqual(falseBlock, condBranchOp.FalseTarget);
        }

        [Test]
        public void ReturnOp_WithValue_CreatesCorrectly()
        {
            var scalarType = new ScalarType(DataType.Float32);
            var returnValue = new RegisterValue(scalarType, "result");

            var returnOp = new ReturnOp(returnValue);

            Assert.IsNotNull(returnOp);
            Assert.AreEqual(returnValue, returnOp.ReturnValue);
            Assert.AreEqual("return", returnOp.Name);
        }

        [Test]
        public void ReturnOp_Void_CreatesCorrectly()
        {
            var returnOp = new ReturnOp();

            Assert.IsNotNull(returnOp);
            Assert.IsNull(returnOp.ReturnValue);
            Assert.AreEqual("return", returnOp.Name);
        }

        #endregion

        #region Phi Node Tests

        [Test]
        public void PhiNode_CreatesCorrectly()
        {
            var scalarType = new ScalarType(DataType.Float32);
            var result = new RegisterValue(scalarType, "phi_result");

            var block1 = new IRBlock("block1");
            var value1 = new RegisterValue(scalarType, "value1");

            var block2 = new IRBlock("block2");
            var value2 = new RegisterValue(scalarType, "value2");

            var incomingValues = new System.Collections.Generic.List<(IRBlock, LLIRValue)>
            {
                (block1, value1),
                (block2, value2)
            };

            var phiNode = new PhiNode(result, incomingValues);

            Assert.IsNotNull(phiNode);
            Assert.AreEqual(result, phiNode.Result);
            Assert.AreEqual(2, phiNode.IncomingValues.Count);
        }

        [Test]
        public void PhiNode_AddIncoming_WorksCorrectly()
        {
            var scalarType = new ScalarType(DataType.Float32);
            var result = new RegisterValue(scalarType, "phi_result");

            var block1 = new IRBlock("block1");
            var value1 = new RegisterValue(scalarType, "value1");

            var incomingValues = new System.Collections.Generic.List<(IRBlock, LLIRValue)>
            {
                (block1, value1)
            };

            var phiNode = new PhiNode(result, incomingValues);

            var block2 = new IRBlock("block2");
            var value2 = new RegisterValue(scalarType, "value2");

            phiNode.AddIncoming(block2, value2);

            Assert.AreEqual(2, phiNode.IncomingValues.Count);
        }

        #endregion

        #region LLIR Function Tests

        [Test]
        public void LLIRFunction_CreatesCorrectly()
        {
            var function = new LLIRFunction("test_function", _context);

            Assert.IsNotNull(function);
            Assert.AreEqual("test_function", function.Name);
            Assert.AreEqual(_context, function.Context);
            Assert.IsFalse(function.IsKernel);
            Assert.IsNotNull(function.Registers);
            Assert.IsNotNull(function.Blocks);
        }

        [Test]
        public void LLIRFunction_Kernel_CreatesCorrectly()
        {
            var function = new LLIRFunction("kernel", _context, isKernel: true);

            Assert.IsTrue(function.IsKernel);
        }

        [Test]
        public void LLIRFunction_AllocateRegister_WorksCorrectly()
        {
            var function = new LLIRFunction("test_function", _context);
            var scalarType = new ScalarType(DataType.Float32);

            var register = function.AllocateRegister(scalarType, "r0");

            Assert.IsNotNull(register);
            Assert.IsInstanceOf<RegisterValue>(register);
            Assert.IsTrue(register.IsRegister);
            Assert.AreEqual("r0", register.Name);
            Assert.AreEqual(1, function.Registers.Count);
        }

        [Test]
        public void LLIRFunction_AllocateBuffer_WorksCorrectly()
        {
            var function = new LLIRFunction("test_function", _context);

            var buffer = function.AllocateBuffer(1024, alignment: 16);

            Assert.IsNotNull(buffer);
            Assert.IsInstanceOf<MemoryValue>(buffer);
            Assert.IsTrue(buffer.IsMemoryLocation);
            Assert.AreEqual(1, function.Registers.Count);
            Assert.AreEqual(1024, buffer.SizeInBytes);
        }

        [Test]
        public void LLIRFunction_AddBlock_WorksCorrectly()
        {
            var function = new LLIRFunction("test_function", _context);
            var block = new IRBlock("entry_block");

            function.AddBlock(block);

            Assert.AreEqual(1, function.Blocks.Count);
            Assert.AreEqual(block, function.Blocks[0]);
        }

        [Test]
        public void LLIRFunction_GetEntryBlock_ReturnsFirstBlock()
        {
            var function = new LLIRFunction("test_function", _context);
            var entryBlock = new IRBlock("entry_block");
            var otherBlock = new IRBlock("other_block");

            function.AddBlock(entryBlock);
            function.AddBlock(otherBlock);

            var result = function.GetEntryBlock();

            Assert.AreEqual(entryBlock, result);
        }

        [Test]
        public void LLIRFunction_AddParameter_WorksCorrectly()
        {
            var function = new LLIRFunction("test_function", _context);
            var scalarType = new ScalarType(DataType.Float32);
            var param = new RegisterValue(scalarType, "param1");

            function.AddParameter(param);

            Assert.AreEqual(1, function.Parameters.Count);
            Assert.AreEqual(param, function.Parameters[0]);
        }

        #endregion

        #region Value Tests

        [Test]
        public void LLIRValue_Register_CreatesCorrectly()
        {
            var scalarType = new ScalarType(DataType.Float32);
            var value = new LLIRValue(scalarType, "reg0", isRegister: true);

            Assert.IsTrue(value.IsRegister);
            Assert.IsFalse(value.IsMemoryLocation);
            Assert.AreEqual("reg0", value.Name);
        }

        [Test]
        public void LLIRValue_Memory_CreatesCorrectly()
        {
            var scalarType = new ScalarType(DataType.Float32);
            var value = new LLIRValue(scalarType, "mem0", isRegister: false);

            Assert.IsFalse(value.IsRegister);
            Assert.IsTrue(value.IsMemoryLocation);
            Assert.AreEqual("mem0", value.Name);
        }

        [Test]
        public void RegisterValue_CreatesCorrectly()
        {
            var scalarType = new ScalarType(DataType.Float32);
            var register = new RegisterValue(scalarType, "r0");

            Assert.IsInstanceOf<LLIRValue>(register);
            Assert.IsTrue(register.IsRegister);
            Assert.AreEqual("r0", register.Name);
        }

        [Test]
        public void MemoryValue_CreatesCorrectly()
        {
            var pointerType = new PointerType(new ScalarType(DataType.Float32));
            var memory = new MemoryValue(pointerType, "buf0", 0, 1024);

            Assert.IsInstanceOf<LLIRValue>(memory);
            Assert.IsTrue(memory.IsMemoryLocation);
            Assert.AreEqual("buf0", memory.Name);
            Assert.AreEqual(0, memory.MemoryOffset);
            Assert.AreEqual(1024, memory.SizeInBytes);
        }

        #endregion

        #region Type Tests

        [Test]
        public void ScalarType_CreatesCorrectly()
        {
            var scalarType = new ScalarType(DataType.Float32);

            Assert.IsNotNull(scalarType);
            Assert.AreEqual(DataType.Float32, scalarType.DataType);
            Assert.IsTrue(scalarType.IsFloat);
            Assert.IsFalse(scalarType.IsInteger);
        }

        [Test]
        public void PointerType_CreatesCorrectly()
        {
            var elementType = new ScalarType(DataType.Float32);
            var pointerType = new PointerType(elementType);

            Assert.IsNotNull(pointerType);
            Assert.AreEqual(elementType, pointerType.ElementType);
        }

        [Test]
        public void VectorType_CreatesCorrectly()
        {
            var elementType = new ScalarType(DataType.Float32);
            var vectorType = new VectorType(elementType, 4);

            Assert.IsNotNull(vectorType);
            Assert.AreEqual(elementType, vectorType.ElementType);
            Assert.AreEqual(4, vectorType.Width);
        }

        #endregion
    }
}
