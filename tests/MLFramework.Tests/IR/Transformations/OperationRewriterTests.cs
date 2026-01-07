using NUnit.Framework;
using MLFramework.IR.Transformations;

namespace MLFramework.Tests.IR.Transformations
{
    [TestFixture]
    public class OperationRewriterTests
    {
        private IRContext _sourceContext;
        private IRContext _targetContext;
        private OperationRewriter _rewriter;
        private HLIRModule _module;

        [SetUp]
        public void Setup()
        {
            _sourceContext = new IRContext();
            _targetContext = new IRContext();
            _rewriter = new OperationRewriter(_sourceContext, _targetContext);
            _module = new HLIRModule();
        }

        [Test]
        public void Constructor_InitializesContexts()
        {
            Assert.IsNotNull(_rewriter.SourceContext);
            Assert.IsNotNull(_rewriter.TargetContext);
            Assert.AreEqual(_sourceContext, _rewriter.SourceContext);
            Assert.AreEqual(_targetContext, _rewriter.TargetContext);
        }

        [Test]
        public void Constructor_WithNullSource_ThrowsArgumentNullException()
        {
            Assert.Throws<System.ArgumentNullException>(() => new OperationRewriter(null, _targetContext));
        }

        [Test]
        public void Constructor_WithNullTarget_ThrowsArgumentNullException()
        {
            Assert.Throws<System.ArgumentNullException>(() => new OperationRewriter(_sourceContext, null));
        }

        [Test]
        public void RemapValue_WithNullValue_ThrowsArgumentNullException()
        {
            Assert.Throws<System.ArgumentNullException>(() => _rewriter.RemapValue(null));
        }

        [Test]
        public void RemapValue_WithSourceValue_CreatesNewValue()
        {
            var sourceValue = _sourceContext.CreateValue(new TensorType(DataType.Float32, new[] { 32, 784 }), "input");

            var remapped = _rewriter.RemapValue(sourceValue);

            Assert.IsNotNull(remapped);
            Assert.AreNotSame(sourceValue, remapped);
            Assert.AreEqual(_targetContext, remapped.Context);
            Assert.AreEqual("input", remapped.Name);
            Assert.AreEqual(sourceValue.Type, remapped.Type);
        }

        [Test]
        public void RemapValue_WithSameValue_ReturnsSameMappedValue()
        {
            var sourceValue = _sourceContext.CreateValue(new TensorType(DataType.Float32, new[] { 1 }), "val");

            var firstRemap = _rewriter.RemapValue(sourceValue);
            var secondRemap = _rewriter.RemapValue(sourceValue);

            Assert.AreSame(firstRemap, secondRemap);
        }

        [Test]
        public void RemapValue_WithTargetContextValue_ReturnsAsIs()
        {
            var targetValue = _targetContext.CreateValue(new TensorType(DataType.Float32, new[] { 1 }), "val");

            var remapped = _rewriter.RemapValue(targetValue);

            Assert.AreSame(targetValue, remapped);
        }

        [Test]
        public void SetMapping_WithNullSource_ThrowsArgumentNullException()
        {
            var targetValue = _targetContext.CreateValue(new TensorType(DataType.Float32, new[] { 1 }), "val");
            Assert.Throws<System.ArgumentNullException>(() => _rewriter.SetMapping(null, targetValue));
        }

        [Test]
        public void SetMapping_WithNullTarget_ThrowsArgumentNullException()
        {
            var sourceValue = _sourceContext.CreateValue(new TensorType(DataType.Float32, new[] { 1 }), "val");
            Assert.Throws<System.ArgumentNullException>(() => _rewriter.SetMapping(sourceValue, null));
        }

        [Test]
        public void SetMapping_ThenRemapValue_ReturnsMappedValue()
        {
            var sourceValue = _sourceContext.CreateValue(new TensorType(DataType.Float32, new[] { 1 }), "src");
            var targetValue = _targetContext.CreateValue(new TensorType(DataType.Float32, new[] { 1 }), "tgt");

            _rewriter.SetMapping(sourceValue, targetValue);
            var remapped = _rewriter.RemapValue(sourceValue);

            Assert.AreSame(targetValue, remapped);
        }

        [Test]
        public void HasMapping_WithUnmappedValue_ReturnsFalse()
        {
            var sourceValue = _sourceContext.CreateValue(new TensorType(DataType.Float32, new[] { 1 }), "val");

            Assert.IsFalse(_rewriter.HasMapping(sourceValue));
        }

        [Test]
        public void HasMapping_WithMappedValue_ReturnsTrue()
        {
            var sourceValue = _sourceContext.CreateValue(new TensorType(DataType.Float32, new[] { 1 }), "src");
            var targetValue = _targetContext.CreateValue(new TensorType(DataType.Float32, new[] { 1 }), "tgt");

            _rewriter.SetMapping(sourceValue, targetValue);

            Assert.IsTrue(_rewriter.HasMapping(sourceValue));
        }

        [Test]
        public void HasMapping_WithNullValue_ReturnsFalse()
        {
            Assert.IsFalse(_rewriter.HasMapping(null));
        }

        [Test]
        public void MappingCount_InitiallyZero()
        {
            Assert.AreEqual(0, _rewriter.MappingCount);
        }

        [Test]
        public void MappingCount_AfterMapping_ReturnsCorrectCount()
        {
            var source1 = _sourceContext.CreateValue(new TensorType(DataType.Float32, new[] { 1 }), "s1");
            var source2 = _sourceContext.CreateValue(new TensorType(DataType.Float32, new[] { 1 }), "s2");
            var target1 = _targetContext.CreateValue(new TensorType(DataType.Float32, new[] { 1 }), "t1");
            var target2 = _targetContext.CreateValue(new TensorType(DataType.Float32, new[] { 1 }), "t2");

            _rewriter.SetMapping(source1, target1);
            _rewriter.SetMapping(source2, target2);

            Assert.AreEqual(2, _rewriter.MappingCount);
        }

        [Test]
        public void ClearMappings_ThenMappingCount_ReturnsZero()
        {
            var source1 = _sourceContext.CreateValue(new TensorType(DataType.Float32, new[] { 1 }), "s1");
            var target1 = _targetContext.CreateValue(new TensorType(DataType.Float32, new[] { 1 }), "t1");

            _rewriter.SetMapping(source1, target1);
            _rewriter.ClearMappings();

            Assert.AreEqual(0, _rewriter.MappingCount);
            Assert.IsFalse(_rewriter.HasMapping(source1));
        }

        [Test]
        public void RemapBlock_WithNullBlock_ThrowsArgumentNullException()
        {
            Assert.Throws<System.ArgumentNullException>(() => _rewriter.RemapBlock(null, _targetContext));
        }

        [Test]
        public void RemapBlock_WithNullTargetContext_ThrowsArgumentNullException()
        {
            var block = new IRBlock("test");
            Assert.Throws<System.ArgumentNullException>(() => _rewriter.RemapBlock(block, null));
        }

        [Test]
        public void RemapBlock_WithEmptyBlock_CreatesNewBlock()
        {
            var sourceBlock = new IRBlock("test");

            var remapped = _rewriter.RemapBlock(sourceBlock, _targetContext);

            Assert.IsNotNull(remapped);
            Assert.AreNotSame(sourceBlock, remapped);
            Assert.AreEqual("test", remapped.Name);
        }

        [Test]
        public void RemapFunction_WithNullFunction_ThrowsArgumentNullException()
        {
            Assert.Throws<System.ArgumentNullException>(() => _rewriter.RemapFunction(null, _targetContext));
        }

        [Test]
        public void RemapFunction_WithNullTargetContext_ThrowsArgumentNullException()
        {
            var func = _module.CreateFunction("test");
            Assert.Throws<System.ArgumentNullException>(() => _rewriter.RemapFunction(func, null));
        }
    }
}
