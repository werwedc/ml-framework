using MLFramework.Shapes;
using MLFramework.Shapes.Inference;

namespace MLFramework.Tests.Shapes.Inference
{
    /// <summary>
    /// Unit tests for the InferenceContext class.
    /// </summary>
    [TestClass]
    public class InferenceContextTests
    {
        private InferenceContext _context;

        [TestInitialize]
        public void Setup()
        {
            _context = new InferenceContext();
        }

        [TestCleanup]
        public void Cleanup()
        {
            _context = null;
        }

        #region Shape Management Tests

        [TestMethod]
        public void SetShape_ShouldAddShape()
        {
            // Arrange
            var shape = SymbolicShapeFactory.FromConcrete(3, 4);

            // Act
            _context.SetShape("tensor1", shape);

            // Assert
            Assert.IsTrue(_context.HasShape("tensor1"));
            Assert.IsNotNull(_context.GetShape("tensor1"));
            Assert.AreSame(shape, _context.GetShape("tensor1"));
        }

        [TestMethod]
        public void SetShape_ShouldReplaceExistingShape()
        {
            // Arrange
            var shape1 = SymbolicShapeFactory.FromConcrete(3, 4);
            var shape2 = SymbolicShapeFactory.FromConcrete(5, 6);
            _context.SetShape("tensor1", shape1);

            // Act
            _context.SetShape("tensor1", shape2);

            // Assert
            Assert.AreSame(shape2, _context.GetShape("tensor1"));
            Assert.AreNotSame(shape1, _context.GetShape("tensor1"));
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void SetShape_WithNullName_ShouldThrow()
        {
            // Act
            _context.SetShape(null, SymbolicShapeFactory.FromConcrete(3, 4));
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void SetShape_WithNullShape_ShouldThrow()
        {
            // Act
            _context.SetShape("tensor1", null);
        }

        [TestMethod]
        public void GetShape_WithNonExistentTensor_ShouldReturnNull()
        {
            // Act
            var shape = _context.GetShape("nonexistent");

            // Assert
            Assert.IsNull(shape);
        }

        [TestMethod]
        public void GetShape_WithEmptyName_ShouldReturnNull()
        {
            // Act
            var shape = _context.GetShape("");

            // Assert
            Assert.IsNull(shape);
        }

        [TestMethod]
        public void HasShape_WithExistingTensor_ShouldReturnTrue()
        {
            // Arrange
            _context.SetShape("tensor1", SymbolicShapeFactory.FromConcrete(3, 4));

            // Act
            bool result = _context.HasShape("tensor1");

            // Assert
            Assert.IsTrue(result);
        }

        [TestMethod]
        public void HasShape_WithNonExistentTensor_ShouldReturnFalse()
        {
            // Act
            bool result = _context.HasShape("nonexistent");

            // Assert
            Assert.IsFalse(result);
        }

        #endregion

        #region Operation Results Tests

        [TestMethod]
        public void RecordInference_ShouldAddResults()
        {
            // Arrange
            var results = new List<SymbolicShape>
            {
                SymbolicShapeFactory.FromConcrete(3, 4)
            };

            // Act
            _context.RecordInference("op1", results);

            // Assert
            Assert.IsTrue(_context.HasOperationResults("op1"));
            var retrievedResults = _context.GetOperationResults("op1");
            Assert.IsNotNull(retrievedResults);
            Assert.AreEqual(1, retrievedResults.Count);
        }

        [TestMethod]
        public void RecordInference_ShouldReplaceExistingResults()
        {
            // Arrange
            var results1 = new List<SymbolicShape>
            {
                SymbolicShapeFactory.FromConcrete(3, 4)
            };
            var results2 = new List<SymbolicShape>
            {
                SymbolicShapeFactory.FromConcrete(5, 6)
            };
            _context.RecordInference("op1", results1);

            // Act
            _context.RecordInference("op1", results2);

            // Assert
            var retrievedResults = _context.GetOperationResults("op1");
            Assert.AreEqual(1, retrievedResults.Count);
            CollectionAssert.AreEqual(results2[0].ToConcrete(), retrievedResults[0].ToConcrete());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void RecordInference_WithNullOpId_ShouldThrow()
        {
            // Act
            _context.RecordInference(null, new List<SymbolicShape>());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void RecordInference_WithNullResults_ShouldThrow()
        {
            // Act
            _context.RecordInference("op1", null);
        }

        [TestMethod]
        public void GetOperationResults_WithNonExistentOp_ShouldReturnNull()
        {
            // Act
            var results = _context.GetOperationResults("nonexistent");

            // Assert
            Assert.IsNull(results);
        }

        [TestMethod]
        public void HasOperationResults_WithExistingOp_ShouldReturnTrue()
        {
            // Arrange
            var results = new List<SymbolicShape>
            {
                SymbolicShapeFactory.FromConcrete(3, 4)
            };
            _context.RecordInference("op1", results);

            // Act
            bool result = _context.HasOperationResults("op1");

            // Assert
            Assert.IsTrue(result);
        }

        [TestMethod]
        public void HasOperationResults_WithNonExistentOp_ShouldReturnFalse()
        {
            // Act
            bool result = _context.HasOperationResults("nonexistent");

            // Assert
            Assert.IsFalse(result);
        }

        #endregion

        #region Clear Tests

        [TestMethod]
        public void ClearTensorShapes_ShouldRemoveAllShapes()
        {
            // Arrange
            _context.SetShape("tensor1", SymbolicShapeFactory.FromConcrete(3, 4));
            _context.SetShape("tensor2", SymbolicShapeFactory.FromConcrete(5, 6));
            _context.RecordInference("op1", new List<SymbolicShape> { SymbolicShapeFactory.FromConcrete(3, 4) });

            // Act
            _context.ClearTensorShapes();

            // Assert
            Assert.IsFalse(_context.HasShape("tensor1"));
            Assert.IsFalse(_context.HasShape("tensor2"));
            Assert.IsTrue(_context.HasOperationResults("op1")); // Operation results should remain
        }

        [TestMethod]
        public void ClearOperationResults_ShouldRemoveAllResults()
        {
            // Arrange
            _context.SetShape("tensor1", SymbolicShapeFactory.FromConcrete(3, 4));
            _context.RecordInference("op1", new List<SymbolicShape> { SymbolicShapeFactory.FromConcrete(3, 4) });
            _context.RecordInference("op2", new List<SymbolicShape> { SymbolicShapeFactory.FromConcrete(5, 6) });

            // Act
            _context.ClearOperationResults();

            // Assert
            Assert.IsFalse(_context.HasOperationResults("op1"));
            Assert.IsFalse(_context.HasOperationResults("op2"));
            Assert.IsTrue(_context.HasShape("tensor1")); // Tensor shapes should remain
        }

        [TestMethod]
        public void Clear_ShouldRemoveAllData()
        {
            // Arrange
            _context.SetShape("tensor1", SymbolicShapeFactory.FromConcrete(3, 4));
            _context.RecordInference("op1", new List<SymbolicShape> { SymbolicShapeFactory.FromConcrete(3, 4) });

            // Act
            _context.Clear();

            // Assert
            Assert.IsFalse(_context.HasShape("tensor1"));
            Assert.IsFalse(_context.HasOperationResults("op1"));
            Assert.AreEqual(0, _context.TensorShapes.Count);
            Assert.AreEqual(0, _context.OperationResults.Count);
        }

        #endregion

        #region Constructor Tests

        [TestMethod]
        public void Constructor_WithInitialShapes_ShouldPopulateContext()
        {
            // Arrange
            var initialShapes = new Dictionary<string, SymbolicShape>
            {
                { "tensor1", SymbolicShapeFactory.FromConcrete(3, 4) },
                { "tensor2", SymbolicShapeFactory.FromConcrete(5, 6) }
            };

            // Act
            var context = new InferenceContext(initialShapes);

            // Assert
            Assert.IsTrue(context.HasShape("tensor1"));
            Assert.IsTrue(context.HasShape("tensor2"));
            CollectionAssert.AreEqual(new int[] { 3, 4 }, context.GetShape("tensor1").ToConcrete());
            CollectionAssert.AreEqual(new int[] { 5, 6 }, context.GetShape("tensor2").ToConcrete());
        }

        [TestMethod]
        public void Constructor_WithNullInitialShapes_ShouldThrow()
        {
            // Act & Assert
            Assert.ThrowsException<ArgumentNullException>(() => new InferenceContext(null));
        }

        #endregion

        #region Clone Tests

        [TestMethod]
        public void Clone_ShouldCreateIndependentCopy()
        {
            // Arrange
            var shape1 = SymbolicShapeFactory.FromConcrete(3, 4);
            _context.SetShape("tensor1", shape1);
            var results = new List<SymbolicShape> { SymbolicShapeFactory.FromConcrete(5, 6) };
            _context.RecordInference("op1", results);

            // Act
            var cloned = _context.Clone();

            // Assert
            Assert.IsTrue(cloned.HasShape("tensor1"));
            Assert.IsTrue(cloned.HasOperationResults("op1"));

            // Modify original
            _context.SetShape("tensor2", SymbolicShapeFactory.FromConcrete(7, 8));

            // Cloned should not be affected
            Assert.IsFalse(cloned.HasShape("tensor2"));
            Assert.IsTrue(_context.HasShape("tensor2"));
        }

        [TestMethod]
        public void Clone_ShouldDeepCopyShapes()
        {
            // Arrange
            var shape1 = SymbolicShapeFactory.FromConcrete(3, 4);
            _context.SetShape("tensor1", shape1);

            // Act
            var cloned = _context.Clone();
            var shape1InClone = cloned.GetShape("tensor1");

            // Assert
            Assert.IsNotNull(shape1InClone);
            Assert.AreEqual(shape1.Rank, shape1InClone.Rank);
            CollectionAssert.AreEqual(shape1.ToConcrete(), shape1InClone.ToConcrete());
            Assert.AreNotSame(shape1, shape1InClone);
        }

        #endregion
    }
}
