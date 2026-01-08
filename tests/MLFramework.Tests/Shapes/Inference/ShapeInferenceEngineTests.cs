using MLFramework.Shapes;
using MLFramework.Shapes.Inference;
using MLFramework.Shapes.Inference.Rules;

namespace MLFramework.Tests.Shapes.Inference
{
    /// <summary>
    /// Unit tests for the ShapeInferenceEngine.
    /// </summary>
    [TestClass]
    public class ShapeInferenceEngineTests
    {
        private ShapeInferenceEngine _engine;

        [TestInitialize]
        public void Setup()
        {
            _engine = new ShapeInferenceEngine();
        }

        [TestCleanup]
        public void Cleanup()
        {
            _engine = null;
        }

        #region Rule Registration Tests

        [TestMethod]
        public void RegisterRule_ShouldAddRule()
        {
            // Arrange
            var rule = new AddRule();

            // Act
            _engine.RegisterRule("Add", rule);

            // Assert
            Assert.IsTrue(_engine.HasRule("Add"));
            Assert.IsNotNull(_engine.GetRule("Add"));
            Assert.AreSame(rule, _engine.GetRule("Add"));
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void RegisterRule_WithNullOpName_ShouldThrow()
        {
            // Arrange
            var rule = new AddRule();

            // Act
            _engine.RegisterRule(null, rule);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void RegisterRule_WithNullRule_ShouldThrow()
        {
            // Act
            _engine.RegisterRule("Add", null);
        }

        [TestMethod]
        public void UnregisterRule_ShouldRemoveRule()
        {
            // Arrange
            var rule = new AddRule();
            _engine.RegisterRule("Add", rule);

            // Act
            bool result = _engine.UnregisterRule("Add");

            // Assert
            Assert.IsTrue(result);
            Assert.IsFalse(_engine.HasRule("Add"));
        }

        [TestMethod]
        public void UnregisterRule_WithNonExistentRule_ShouldReturnFalse()
        {
            // Act
            bool result = _engine.UnregisterRule("NonExistent");

            // Assert
            Assert.IsFalse(result);
        }

        [TestMethod]
        public void RegisterRules_ShouldAddMultipleRules()
        {
            // Arrange
            var rules = new Dictionary<string, IShapeInferenceRule>
            {
                { "Add", new AddRule() },
                { "Mul", new MulRule() },
                { "MatMul", new MatMulRule() }
            };

            // Act
            _engine.RegisterRules(rules);

            // Assert
            Assert.IsTrue(_engine.HasRule("Add"));
            Assert.IsTrue(_engine.HasRule("Mul"));
            Assert.IsTrue(_engine.HasRule("MatMul"));
        }

        [TestMethod]
        public void GetRegisteredOperations_ShouldReturnAllRegisteredOps()
        {
            // Arrange
            _engine.RegisterRule("Add", new AddRule());
            _engine.RegisterRule("Mul", new MulRule());

            // Act
            var ops = _engine.GetRegisteredOperations().ToList();

            // Assert
            Assert.AreEqual(2, ops.Count);
            CollectionAssert.Contains(ops, "Add");
            CollectionAssert.Contains(ops, "Mul");
        }

        [TestMethod]
        public void ClearRules_ShouldRemoveAllRules()
        {
            // Arrange
            _engine.RegisterRule("Add", new AddRule());
            _engine.RegisterRule("Mul", new MulRule());

            // Act
            _engine.ClearRules();

            // Assert
            Assert.IsFalse(_engine.HasRule("Add"));
            Assert.IsFalse(_engine.HasRule("Mul"));
            Assert.AreEqual(0, _engine.GetRegisteredOperations().Count());
        }

        #endregion

        #region CanInfer Tests

        [TestMethod]
        public void CanInfer_WithRegisteredRule_ShouldReturnTrue()
        {
            // Arrange
            _engine.RegisterRule("Add", new AddRule());
            var inputs = new List<SymbolicShape>
            {
                SymbolicShapeFactory.FromConcrete(3, 4),
                SymbolicShapeFactory.FromConcrete(3, 4)
            };

            // Act
            bool result = _engine.CanInfer("Add", inputs);

            // Assert
            Assert.IsTrue(result);
        }

        [TestMethod]
        public void CanInfer_WithUnregisteredOperation_ShouldReturnFalse()
        {
            // Arrange
            var inputs = new List<SymbolicShape>
            {
                SymbolicShapeFactory.FromConcrete(3, 4)
            };

            // Act
            bool result = _engine.CanInfer("NonExistent", inputs);

            // Assert
            Assert.IsFalse(result);
        }

        [TestMethod]
        public void CanInfer_WithNullOpName_ShouldReturnFalse()
        {
            // Act
            bool result = _engine.CanInfer(null, new List<SymbolicShape>());

            // Assert
            Assert.IsFalse(result);
        }

        #endregion

        #region Infer Tests

        [TestMethod]
        public void Infer_WithValidInputs_ShouldReturnOutputShape()
        {
            // Arrange
            _engine.RegisterRule("Add", new AddRule());
            var inputs = new List<SymbolicShape>
            {
                SymbolicShapeFactory.FromConcrete(3, 4),
                SymbolicShapeFactory.FromConcrete(3, 4)
            };

            // Act
            var outputs = _engine.Infer("Add", inputs);

            // Assert
            Assert.AreEqual(1, outputs.Count);
            Assert.AreEqual(2, outputs[0].Rank);
            CollectionAssert.AreEqual(new int[] { 3, 4 }, outputs[0].ToConcrete());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Infer_WithUnregisteredOperation_ShouldThrow()
        {
            // Arrange
            var inputs = new List<SymbolicShape>
            {
                SymbolicShapeFactory.FromConcrete(3, 4)
            };

            // Act
            _engine.Infer("NonExistent", inputs);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void Infer_WithNullOpName_ShouldThrow()
        {
            // Act
            _engine.Infer(null, new List<SymbolicShape>());
        }

        [TestMethod]
        public void Infer_ShouldValidateInputs()
        {
            // Arrange
            _engine.RegisterRule("Add", new AddRule());
            var inputs = new List<SymbolicShape>
            {
                null,
                SymbolicShapeFactory.FromConcrete(3, 4)
            };

            // Act & Assert
            Assert.ThrowsException<ArgumentException>(() => _engine.Infer("Add", inputs));
        }

        #endregion

        #region Validate Tests

        [TestMethod]
        public void Validate_WithValidInputs_ShouldReturnTrue()
        {
            // Arrange
            _engine.RegisterRule("Add", new AddRule());
            var inputs = new List<SymbolicShape>
            {
                SymbolicShapeFactory.FromConcrete(3, 4),
                SymbolicShapeFactory.FromConcrete(3, 4)
            };

            // Act
            bool result = _engine.Validate("Add", inputs);

            // Assert
            Assert.IsTrue(result);
        }

        [TestMethod]
        public void Validate_WithInvalidInputs_ShouldReturnFalse()
        {
            // Arrange
            _engine.RegisterRule("Add", new AddRule());
            var inputs = new List<SymbolicShape>
            {
                SymbolicShapeFactory.FromConcrete(3, 4)
            };

            // Act
            bool result = _engine.Validate("Add", inputs);

            // Assert
            Assert.IsFalse(result);
        }

        #endregion
    }
}
