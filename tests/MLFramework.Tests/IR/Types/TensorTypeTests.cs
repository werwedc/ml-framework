using NUnit.Framework;
using MLFramework.IR.Types;

namespace MLFramework.Tests.IR.Types
{
    [TestFixture]
    public class TensorTypeTests
    {
        [Test]
        public void TensorType_Construction_CreatesCorrectType()
        {
            var type = new TensorType(DataType.Float32, new[] { 32, 784 });

            Assert.AreEqual(DataType.Float32, type.ElementType);
            Assert.AreEqual(new[] { 32, 784 }, type.Shape);
            Assert.AreEqual(2, type.Rank);
        }

        [Test]
        public void TensorType_WithDynamicShape_CreatesDynamicType()
        {
            var type = new TensorType(DataType.Float32, new[] { -1, 784 });

            Assert.IsTrue(type.IsDynamic);
            Assert.IsFalse(type.HasKnownShape());
        }

        [Test]
        public void TensorType_WithNewShape_CreatesNewType()
        {
            var original = new TensorType(DataType.Float32, new[] { 32, 784 });
            var reshaped = original.WithNewShape(new[] { 32 * 784 });

            Assert.AreEqual(new[] { 32 * 784 }, reshaped.Shape);
        }

        [Test]
        public void TensorType_Equals_ReturnsTrueForSameShape()
        {
            var type1 = new TensorType(DataType.Float32, new[] { 32, 784 });
            var type2 = new TensorType(DataType.Float32, new[] { 32, 784 });

            Assert.IsTrue(type1.Equals(type2));
        }

        [Test]
        public void TensorType_Equals_ReturnsFalseForDifferentShape()
        {
            var type1 = new TensorType(DataType.Float32, new[] { 32, 784 });
            var type2 = new TensorType(DataType.Float32, new[] { 64, 784 });

            Assert.IsFalse(type1.Equals(type2));
        }

        [Test]
        public void TensorType_Equals_ReturnsFalseForDifferentElementType()
        {
            var type1 = new TensorType(DataType.Float32, new[] { 32, 784 });
            var type2 = new TensorType(DataType.Int32, new[] { 32, 784 });

            Assert.IsFalse(type1.Equals(type2));
        }

        [Test]
        public void TensorType_Equals_ReturnsFalseForDifferentRank()
        {
            var type1 = new TensorType(DataType.Float32, new[] { 32, 784 });
            var type2 = new TensorType(DataType.Float32, new[] { 32, 784, 1 });

            Assert.IsFalse(type1.Equals(type2));
        }

        [Test]
        public void TensorType_Equals_ReturnsTrueForSameTypeInstance()
        {
            var type = new TensorType(DataType.Float32, new[] { 32, 784 });

            Assert.IsTrue(type.Equals(type));
        }

        [Test]
        public void TensorType_Equals_ReturnsFalseForNull()
        {
            var type = new TensorType(DataType.Float32, new[] { 32, 784 });

            Assert.IsFalse(type.Equals(null));
        }

        [Test]
        public void TensorType_Equals_ReturnsFalseForDifferentType()
        {
            var type = new TensorType(DataType.Float32, new[] { 32, 784 });
            IIRType otherType = null; // Not a TensorType

            Assert.IsFalse(type.Equals(otherType));
        }

        [Test]
        public void TensorType_HasKnownShape_ReturnsTrueForStaticShape()
        {
            var type = new TensorType(DataType.Float32, new[] { 32, 784 });

            Assert.IsTrue(type.HasKnownShape());
        }

        [Test]
        public void TensorType_HasKnownShape_ReturnsFalseForDynamicShape()
        {
            var type = new TensorType(DataType.Float32, new[] { -1, 784 });

            Assert.IsFalse(type.HasKnownShape());
        }

        [Test]
        public void TensorType_IsDynamic_ReturnsTrueForDynamicDimension()
        {
            var type = new TensorType(DataType.Float32, new[] { -1, 784 });

            Assert.IsTrue(type.IsDynamic);
        }

        [Test]
        public void TensorType_IsDynamic_ReturnsFalseForStaticShape()
        {
            var type = new TensorType(DataType.Float32, new[] { 32, 784 });

            Assert.IsFalse(type.IsDynamic);
        }

        [Test]
        public void TensorType_Rank_ReturnsCorrectDimensionCount()
        {
            var type1 = new TensorType(DataType.Float32, new[] { 32, 784 });
            var type2 = new TensorType(DataType.Float32, new[] { 1, 3, 28, 28 });
            var type3 = new TensorType(DataType.Float32, new[] { 512 });

            Assert.AreEqual(2, type1.Rank);
            Assert.AreEqual(4, type2.Rank);
            Assert.AreEqual(1, type3.Rank);
        }

        [Test]
        public void TensorType_Name_ReturnsCorrectString()
        {
            var type = new TensorType(DataType.Float32, new[] { 32, 784 });

            Assert.AreEqual("tensor<Float32>[32, 784]", type.Name);
        }

        [Test]
        public void TensorType_ToString_ReturnsCorrectRepresentation()
        {
            var type = new TensorType(DataType.Float32, new[] { 32, 784 });

            Assert.AreEqual("tensor<Float32>[32, 784]", type.ToString());
        }

        [Test]
        public void TensorType_GetHashCode_ReturnsSameForEqualTypes()
        {
            var type1 = new TensorType(DataType.Float32, new[] { 32, 784 });
            var type2 = new TensorType(DataType.Float32, new[] { 32, 784 });

            Assert.AreEqual(type1.GetHashCode(), type2.GetHashCode());
        }

        [Test]
        public void TensorType_GetHashCode_ReturnsDifferentForDifferentTypes()
        {
            var type1 = new TensorType(DataType.Float32, new[] { 32, 784 });
            var type2 = new TensorType(DataType.Float32, new[] { 64, 784 });

            Assert.AreNotEqual(type1.GetHashCode(), type2.GetHashCode());
        }

        [Test]
        public void TensorType_Canonicalize_ReturnsSameInstance()
        {
            var type = new TensorType(DataType.Float32, new[] { 32, 784 });

            var canonicalized = type.Canonicalize();

            Assert.AreSame(type, canonicalized);
        }

        [Test]
        public void TensorType_Construction_ThrowsForNullShape()
        {
            Assert.Throws<System.ArgumentException>(() =>
                new TensorType(DataType.Float32, null));
        }

        [Test]
        public void TensorType_Construction_ThrowsForEmptyShape()
        {
            Assert.Throws<System.ArgumentException>(() =>
                new TensorType(DataType.Float32, new int[0]));
        }

        [Test]
        public void TensorType_WithNewShape_PreservesElementType()
        {
            var original = new TensorType(DataType.Float32, new[] { 32, 784 });
            var reshaped = original.WithNewShape(new[] { 32 * 784 });

            Assert.AreEqual(DataType.Float32, reshaped.ElementType);
        }
    }
}
