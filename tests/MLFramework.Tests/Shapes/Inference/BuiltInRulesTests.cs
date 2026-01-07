using MLFramework.Shapes;
using MLFramework.Shapes.Inference;
using MLFramework.Shapes.Inference.Rules;

namespace MLFramework.Tests.Shapes.Inference.Rules
{
    /// <summary>
    /// Unit tests for built-in shape inference rules.
    /// </summary>
    [TestClass]
    public class BuiltInRulesTests
    {
        #region MatMulRule Tests

        [TestMethod]
        public void MatMulRule_2DMatrixMultiplication_ShouldReturnCorrectShape()
        {
            // Arrange
            var rule = new MatMulRule();
            var shapeA = SymbolicShapeFactory.FromConcrete(3, 4); // [M=3, K=4]
            var shapeB = SymbolicShapeFactory.FromConcrete(4, 5); // [K=4, N=5]
            var inputs = new List<SymbolicShape> { shapeA, shapeB };

            // Act
            var outputs = rule.Infer("MatMul", inputs);

            // Assert
            Assert.AreEqual(1, outputs.Count);
            CollectionAssert.AreEqual(new int[] { 3, 5 }, outputs[0].ToConcrete());
        }

        [TestMethod]
        public void MatMulRule_BatchedMatrixMultiplication_ShouldReturnCorrectShape()
        {
            // Arrange
            var rule = new MatMulRule();
            var shapeA = SymbolicShapeFactory.FromConcrete(2, 3, 4); // [B=2, M=3, K=4]
            var shapeB = SymbolicShapeFactory.FromConcrete(2, 4, 5); // [B=2, K=4, N=5]
            var inputs = new List<SymbolicShape> { shapeA, shapeB };

            // Act
            var outputs = rule.Infer("BatchMatMul", inputs);

            // Assert
            Assert.AreEqual(1, outputs.Count);
            CollectionAssert.AreEqual(new int[] { 2, 3, 5 }, outputs[0].ToConcrete());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void MatMulRule_IncompatibleInnerDimensions_ShouldThrow()
        {
            // Arrange
            var rule = new MatMulRule();
            var shapeA = SymbolicShapeFactory.FromConcrete(3, 4); // [M=3, K=4]
            var shapeB = SymbolicShapeFactory.FromConcrete(5, 6); // [K=5, N=6] - incompatible K
            var inputs = new List<SymbolicShape> { shapeA, shapeB };

            // Act
            rule.Infer("MatMul", inputs);
        }

        [TestMethod]
        public void MatMulRule_SymbolicDimensions_ShouldPropagateSymbolicDims()
        {
            // Arrange
            var rule = new MatMulRule();
            var batch = SymbolicDimension.Named("batch");
            var dimM = SymbolicDimension.Named("M");
            var dimK = SymbolicDimension.Named("K");
            var dimN = SymbolicDimension.Named("N");

            var shapeA = new SymbolicShape(batch, dimM, dimK);
            var shapeB = new SymbolicShape(batch, dimK, dimN);
            var inputs = new List<SymbolicShape> { shapeA, shapeB };

            // Act
            var outputs = rule.Infer("MatMul", inputs);

            // Assert
            Assert.AreEqual(1, outputs.Count);
            Assert.AreEqual(3, outputs[0].Rank);
            Assert.AreSame(batch, outputs[0].GetDimension(0));
            Assert.AreSame(dimM, outputs[0].GetDimension(1));
            Assert.AreSame(dimN, outputs[0].GetDimension(2));
        }

        [TestMethod]
        public void MatMulRule_CanInfer_ShouldValidateCorrectly()
        {
            // Arrange
            var rule = new MatMulRule();
            var validInputs = new List<SymbolicShape>
            {
                SymbolicShapeFactory.FromConcrete(3, 4),
                SymbolicShapeFactory.FromConcrete(4, 5)
            };
            var invalidInputs = new List<SymbolicShape>
            {
                SymbolicShapeFactory.FromConcrete(3, 4)
            };

            // Act & Assert
            Assert.IsTrue(rule.CanInfer("MatMul", validInputs));
            Assert.IsFalse(rule.CanInfer("MatMul", invalidInputs));
            Assert.IsFalse(rule.CanInfer("NonExistent", validInputs));
        }

        #endregion

        #region ElementWiseRules Tests

        [TestMethod]
        public void AddRule_SameShapes_ShouldReturnSameShape()
        {
            // Arrange
            var rule = new AddRule();
            var shapeA = SymbolicShapeFactory.FromConcrete(3, 4);
            var shapeB = SymbolicShapeFactory.FromConcrete(3, 4);
            var inputs = new List<SymbolicShape> { shapeA, shapeB };

            // Act
            var outputs = rule.Infer("Add", inputs);

            // Assert
            Assert.AreEqual(1, outputs.Count);
            CollectionAssert.AreEqual(new int[] { 3, 4 }, outputs[0].ToConcrete());
        }

        [TestMethod]
        public void AddRule_BroadcastableShapes_ShouldReturnBroadcastedShape()
        {
            // Arrange
            var rule = new AddRule();
            var shapeA = SymbolicShapeFactory.FromConcrete(3, 4);
            var shapeB = SymbolicShapeFactory.FromConcrete(1, 4);
            var inputs = new List<SymbolicShape> { shapeA, shapeB };

            // Act
            var outputs = rule.Infer("Add", inputs);

            // Assert
            Assert.AreEqual(1, outputs.Count);
            CollectionAssert.AreEqual(new int[] { 3, 4 }, outputs[0].ToConcrete());
        }

        [TestMethod]
        public void AddRule_ScalarBroadcast_ShouldReturnLargerShape()
        {
            // Arrange
            var rule = new AddRule();
            var shapeA = SymbolicShapeFactory.FromConcrete(3, 4, 5);
            var shapeB = SymbolicShapeFactory.Scalar();
            var inputs = new List<SymbolicShape> { shapeA, shapeB };

            // Act
            var outputs = rule.Infer("Add", inputs);

            // Assert
            Assert.AreEqual(1, outputs.Count);
            CollectionAssert.AreEqual(new int[] { 3, 4, 5 }, outputs[0].ToConcrete());
        }

        [TestMethod]
        public void MulRule_SameShapes_ShouldReturnSameShape()
        {
            // Arrange
            var rule = new MulRule();
            var shapeA = SymbolicShapeFactory.FromConcrete(2, 3);
            var shapeB = SymbolicShapeFactory.FromConcrete(2, 3);
            var inputs = new List<SymbolicShape> { shapeA, shapeB };

            // Act
            var outputs = rule.Infer("Mul", inputs);

            // Assert
            Assert.AreEqual(1, outputs.Count);
            CollectionAssert.AreEqual(new int[] { 2, 3 }, outputs[0].ToConcrete());
        }

        [TestMethod]
        public void SubRule_SameShapes_ShouldReturnSameShape()
        {
            // Arrange
            var rule = new SubRule();
            var shapeA = SymbolicShapeFactory.FromConcrete(2, 3);
            var shapeB = SymbolicShapeFactory.FromConcrete(2, 3);
            var inputs = new List<SymbolicShape> { shapeA, shapeB };

            // Act
            var outputs = rule.Infer("Sub", inputs);

            // Assert
            Assert.AreEqual(1, outputs.Count);
            CollectionAssert.AreEqual(new int[] { 2, 3 }, outputs[0].ToConcrete());
        }

        [TestMethod]
        public void DivRule_SameShapes_ShouldReturnSameShape()
        {
            // Arrange
            var rule = new DivRule();
            var shapeA = SymbolicShapeFactory.FromConcrete(2, 3);
            var shapeB = SymbolicShapeFactory.FromConcrete(2, 3);
            var inputs = new List<SymbolicShape> { shapeA, shapeB };

            // Act
            var outputs = rule.Infer("Div", inputs);

            // Assert
            Assert.AreEqual(1, outputs.Count);
            CollectionAssert.AreEqual(new int[] { 2, 3 }, outputs[0].ToConcrete());
        }

        #endregion

        #region Conv2DRule Tests

        [TestMethod]
        public void Conv2DRule_BasicConvolution_ShouldReturnCorrectShape()
        {
            // Arrange
            var rule = new Conv2DRule();
            var inputShape = SymbolicShapeFactory.FromConcrete(2, 3, 32, 32); // [N=2, C_in=3, H=32, W=32]
            var weightShape = SymbolicShapeFactory.FromConcrete(16, 3, 3, 3); // [C_out=16, C_in=3, K_h=3, K_w=3]
            var inputs = new List<SymbolicShape> { inputShape, weightShape };

            // Act
            var outputs = rule.Infer("Conv2D", inputs);

            // Assert
            Assert.AreEqual(1, outputs.Count);
            // Output: [N=2, C_out=16, H_out=(32+0-3)/1+1=30, W_out=(32+0-3)/1+1=30]
            CollectionAssert.AreEqual(new int[] { 2, 16, 30, 30 }, outputs[0].ToConcrete());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Conv2DRule_InvalidInputRank_ShouldThrow()
        {
            // Arrange
            var rule = new Conv2DRule();
            var inputShape = SymbolicShapeFactory.FromConcrete(32, 32); // Wrong rank
            var weightShape = SymbolicShapeFactory.FromConcrete(16, 3, 3, 3);
            var inputs = new List<SymbolicShape> { inputShape, weightShape };

            // Act
            rule.Infer("Conv2D", inputs);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Conv2DRule_InvalidWeightRank_ShouldThrow()
        {
            // Arrange
            var rule = new Conv2DRule();
            var inputShape = SymbolicShapeFactory.FromConcrete(2, 3, 32, 32);
            var weightShape = SymbolicShapeFactory.FromConcrete(16, 3); // Wrong rank
            var inputs = new List<SymbolicShape> { inputShape, weightShape };

            // Act
            rule.Infer("Conv2D", inputs);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Conv2DRule_ChannelMismatch_ShouldThrow()
        {
            // Arrange
            var rule = new Conv2DRule();
            var inputShape = SymbolicShapeFactory.FromConcrete(2, 3, 32, 32); // C_in=3
            var weightShape = SymbolicShapeFactory.FromConcrete(16, 4, 3, 3); // C_in=4 (mismatch)
            var inputs = new List<SymbolicShape> { inputShape, weightShape };

            // Act
            rule.Infer("Conv2D", inputs);
        }

        #endregion

        #region TransposeRule Tests

        [TestMethod]
        public void TransposeRule_2DMatrix_ShouldSwapDimensions()
        {
            // Arrange
            var rule = new TransposeRule();
            var inputShape = SymbolicShapeFactory.FromConcrete(3, 4);
            var inputs = new List<SymbolicShape> { inputShape };

            // Act
            var outputs = rule.Infer("Transpose", inputs);

            // Assert
            Assert.AreEqual(1, outputs.Count);
            CollectionAssert.AreEqual(new int[] { 4, 3 }, outputs[0].ToConcrete());
        }

        [TestMethod]
        public void TransposeRule_InferWithPermutation_ShouldApplyPermutation()
        {
            // Arrange
            var rule = new TransposeRule();
            var inputShape = SymbolicShapeFactory.FromConcrete(2, 3, 4);
            var permutation = new[] { 2, 0, 1 };

            // Act
            var outputShape = rule.InferWithPermutation("Transpose", inputShape, permutation);

            // Assert
            CollectionAssert.AreEqual(new int[] { 4, 2, 3 }, outputShape.ToConcrete());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void TransposeRule_InvalidPermutationLength_ShouldThrow()
        {
            // Arrange
            var rule = new TransposeRule();
            var inputShape = SymbolicShapeFactory.FromConcrete(2, 3, 4);
            var permutation = new[] { 0, 1 }; // Wrong length

            // Act
            rule.InferWithPermutation("Transpose", inputShape, permutation);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void TransposeRule_InvalidPermutationValues_ShouldThrow()
        {
            // Arrange
            var rule = new TransposeRule();
            var inputShape = SymbolicShapeFactory.FromConcrete(2, 3, 4);
            var permutation = new[] { 0, 1, 1 }; // Duplicate

            // Act
            rule.InferWithPermutation("Transpose", inputShape, permutation);
        }

        #endregion

        #region ReshapeRule Tests

        [TestMethod]
        public void ReshapeRule_SimpleReshape_ShouldReturnTargetShape()
        {
            // Arrange
            var rule = new ReshapeRule();
            var inputShape = SymbolicShapeFactory.FromConcrete(12); // [12]
            var targetShape = SymbolicShapeFactory.FromConcrete(3, 4); // [3, 4]
            var inputs = new List<SymbolicShape> { inputShape, targetShape };

            // Act
            var outputs = rule.Infer("Reshape", inputs);

            // Assert
            Assert.AreEqual(1, outputs.Count);
            CollectionAssert.AreEqual(new int[] { 3, 4 }, outputs[0].ToConcrete());
        }

        [TestMethod]
        public void ReshapeRule_WithNegOneDimension_ShouldInferDimension()
        {
            // Arrange
            var rule = new ReshapeRule();
            var inputShape = SymbolicShapeFactory.FromConcrete(12); // [12]
            var targetShape = new SymbolicShape(
                SymbolicDimension.FromConcrete(3),
                SymbolicDimension.FromConcrete(-1)
            ); // [3, -1]
            var inputs = new List<SymbolicShape> { inputShape, targetShape };

            // Act
            var outputs = rule.Infer("Reshape", inputs);

            // Assert
            Assert.AreEqual(1, outputs.Count);
            CollectionAssert.AreEqual(new int[] { 3, 4 }, outputs[0].ToConcrete());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ReshapeRule_ElementCountMismatch_ShouldThrow()
        {
            // Arrange
            var rule = new ReshapeRule();
            var inputShape = SymbolicShapeFactory.FromConcrete(12); // [12]
            var targetShape = SymbolicShapeFactory.FromConcrete(5, 5); // [25] - mismatch
            var inputs = new List<SymbolicShape> { inputShape, targetShape };

            // Act
            rule.Infer("Reshape", inputs);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ReshapeRule_MultipleNegOneDimensions_ShouldThrow()
        {
            // Arrange
            var rule = new ReshapeRule();
            var inputShape = SymbolicShapeFactory.FromConcrete(12);
            var targetShape = new SymbolicShape(
                SymbolicDimension.FromConcrete(-1),
                SymbolicDimension.FromConcrete(-1)
            );
            var inputs = new List<SymbolicShape> { inputShape, targetShape };

            // Act
            rule.Infer("Reshape", inputs);
        }

        [TestMethod]
        public void ReshapeRule_SymbolicInput_ShouldReturnTargetShapeAsIs()
        {
            // Arrange
            var rule = new ReshapeRule();
            var dimN = SymbolicDimension.Named("N");
            var inputShape = new SymbolicShape(dimN); // [N] - not fully known
            var targetShape = SymbolicShapeFactory.FromConcrete(3, 4);
            var inputs = new List<SymbolicShape> { inputShape, targetShape };

            // Act
            var outputs = rule.Infer("Reshape", inputs);

            // Assert
            Assert.AreEqual(1, outputs.Count);
            CollectionAssert.AreEqual(new int[] { 3, 4 }, outputs[0].ToConcrete());
        }

        #endregion
    }
}
