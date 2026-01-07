using NUnit.Framework;
using MLFramework.IR;
using MLFramework.IR.MLIR.Type;
using MLFramework.IR.Types;
using MLFramework.IR.Values;

namespace MLFramework.Tests.IR.MLIR.Type
{
    [TestFixture]
    public class CastOpTests
    {
        private IRContext _context;

        [SetUp]
        public void Setup()
        {
            _context = new IRContext();
        }

        [Test]
        public void CastOp_CreatesCorrectOperation()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var resultType = new TensorType(DataType.Int32, new[] { 32, 64 });
            var input = _context.CreateValue(inputType, "input");
            var result = _context.CreateValue(resultType, "result");

            var castOp = new CastOp(input, result, DataType.Int32);

            Assert.AreEqual(input, castOp.Input);
            Assert.AreEqual(result, castOp.Result);
            Assert.AreEqual(DataType.Int32, castOp.TargetType);
        }

        [Test]
        public void CastOp_Validate_DoesNotThrowForValidCast()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var resultType = new TensorType(DataType.Int32, new[] { 32, 64 });
            var input = _context.CreateValue(inputType);
            var result = _context.CreateValue(resultType);

            var castOp = new CastOp(input, result, DataType.Int32);

            Assert.DoesNotThrow(() => castOp.Validate());
        }

        [Test]
        public void CastOp_Validate_ThrowsForShapeMismatch()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var resultType = new TensorType(DataType.Int32, new[] { 32, 128 });
            var input = _context.CreateValue(inputType);
            var result = _context.CreateValue(resultType);

            var castOp = new CastOp(input, result, DataType.Int32);

            Assert.Throws<System.InvalidOperationException>(() => castOp.Validate());
        }

        [Test]
        public void CastOp_Validate_ThrowsForMismatchedTargetType()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var resultType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var input = _context.CreateValue(inputType);
            var result = _context.CreateValue(resultType);

            var castOp = new CastOp(input, result, DataType.Int32);

            Assert.Throws<System.InvalidOperationException>(() => castOp.Validate());
        }

        [Test]
        public void CastOp_SupportsAllDataTypeConversions()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var input = _context.CreateValue(inputType);

            var targetTypes = new[] { DataType.Float16, DataType.Float32, DataType.Float64,
                                    DataType.Int8, DataType.Int16, DataType.Int32, DataType.Int64,
                                    DataType.UInt8, DataType.UInt16, DataType.UInt32, DataType.UInt64 };

            foreach (var targetType in targetTypes)
            {
                var resultType = new TensorType(targetType, new[] { 32, 64 });
                var result = _context.CreateValue(resultType);

                var castOp = new CastOp(input, result, targetType);
                Assert.AreEqual(targetType, castOp.TargetType);
                Assert.DoesNotThrow(() => castOp.Validate());
            }
        }

        [Test]
        public void CastOp_Construction_ThrowsForNullInput()
        {
            var resultType = new TensorType(DataType.Int32, new[] { 32, 64 });
            var result = _context.CreateValue(resultType);

            Assert.Throws<System.ArgumentNullException>(() =>
                new CastOp(null, result, DataType.Int32));
        }

        [Test]
        public void CastOp_Construction_ThrowsForNullResult()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var input = _context.CreateValue(inputType);

            Assert.Throws<System.ArgumentNullException>(() =>
                new CastOp(input, null, DataType.Int32));
        }

        [Test]
        public void CastOp_Clone_CreatesIndependentCopy()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var resultType = new TensorType(DataType.Int32, new[] { 32, 64 });
            var input = _context.CreateValue(inputType);
            var result = _context.CreateValue(resultType);

            var original = new CastOp(input, result, DataType.Int32);
            var cloned = (CastOp)original.Clone();

            Assert.AreEqual(original.Input, cloned.Input);
            Assert.AreEqual(original.Result, cloned.Result);
            Assert.AreEqual(original.TargetType, cloned.TargetType);
        }

        [Test]
        public void CastOp_Validate_ThrowsForNonTensorInput()
        {
            var input = _context.CreateValue(new Attributes.ScalarType(DataType.Float32));
            var resultType = new TensorType(DataType.Int32, new[] { 32, 64 });
            var result = _context.CreateValue(resultType);

            var castOp = new CastOp(input, result, DataType.Int32);

            Assert.Throws<System.InvalidOperationException>(() => castOp.Validate());
        }

        [Test]
        public void CastOp_Validate_ThrowsForNonTensorResult()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var input = _context.CreateValue(inputType);
            var result = _context.CreateValue(new Attributes.ScalarType(DataType.Int32));

            var castOp = new CastOp(input, result, DataType.Int32);

            Assert.Throws<System.InvalidOperationException>(() => castOp.Validate());
        }

        [Test]
        public void CastOp_CanCastBetweenFloatingPointTypes()
        {
            var inputTypes = new[] { DataType.Float16, DataType.Float32, DataType.Float64 };
            var targetTypes = new[] { DataType.Float16, DataType.Float32, DataType.Float64 };

            foreach (var inputType in inputTypes)
            {
                foreach (var targetType in targetTypes)
                {
                    var inputT = new TensorType(inputType, new[] { 32, 64 });
                    var resultT = new TensorType(targetType, new[] { 32, 64 });
                    var input = _context.CreateValue(inputT);
                    var result = _context.CreateValue(resultT);

                    var castOp = new CastOp(input, result, targetType);
                    Assert.DoesNotThrow(() => castOp.Validate());
                }
            }
        }

        [Test]
        public void CastOp_CanCastBetweenIntegerTypes()
        {
            var inputTypes = new[] { DataType.Int8, DataType.Int16, DataType.Int32, DataType.Int64 };
            var targetTypes = new[] { DataType.Int8, DataType.Int16, DataType.Int32, DataType.Int64 };

            foreach (var inputType in inputTypes)
            {
                foreach (var targetType in targetTypes)
                {
                    var inputT = new TensorType(inputType, new[] { 32, 64 });
                    var resultT = new TensorType(targetType, new[] { 32, 64 });
                    var input = _context.CreateValue(inputT);
                    var result = _context.CreateValue(resultT);

                    var castOp = new CastOp(input, result, targetType);
                    Assert.DoesNotThrow(() => castOp.Validate());
                }
            }
        }

        [Test]
        public void CastOp_CanCastFromFloatingToInteger()
        {
            var floatTypes = new[] { DataType.Float16, DataType.Float32, DataType.Float64 };
            var intTypes = new[] { DataType.Int8, DataType.Int16, DataType.Int32, DataType.Int64 };

            foreach (var floatType in floatTypes)
            {
                foreach (var intType in intTypes)
                {
                    var inputT = new TensorType(floatType, new[] { 32, 64 });
                    var resultT = new TensorType(intType, new[] { 32, 64 });
                    var input = _context.CreateValue(inputT);
                    var result = _context.CreateValue(resultT);

                    var castOp = new CastOp(input, result, intType);
                    Assert.DoesNotThrow(() => castOp.Validate());
                }
            }
        }
    }
}
