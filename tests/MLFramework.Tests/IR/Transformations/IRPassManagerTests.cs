using NUnit.Framework;
using MLFramework.IR.Transformations;

namespace MLFramework.Tests.IR.Transformations
{
    [TestFixture]
    public class IRPassManagerTests
    {
        private IRPassManager _passManager;
        private HLIRModule _module;

        [SetUp]
        public void Setup()
        {
            _passManager = new IRPassManager();
            _module = new HLIRModule();
        }

        [Test]
        public void Constructor_InitializesEmptyPasses()
        {
            Assert.AreEqual(0, _passManager.PassCount);
        }

        [Test]
        public void AddPass_AddsPassToManager()
        {
            var pass = new MockTransformation("TestPass");
            _passManager.AddPass(pass, IRPassManager.PassType.Optimization);

            Assert.AreEqual(1, _passManager.PassCount);
            Assert.AreEqual(1, _passManager.GetPasses(IRPassManager.PassType.Optimization).Count);
        }

        [Test]
        public void AddPass_ByType_AddsPassToCorrectCategory()
        {
            var optPass = new MockTransformation("OptPass");
            var analysisPass = new MockTransformation("AnalysisPass", isAnalysisOnly: true);

            _passManager.AddPass(optPass, IRPassManager.PassType.Optimization);
            _passManager.AddPass(analysisPass, IRPassManager.PassType.Analysis);

            Assert.AreEqual(2, _passManager.PassCount);
            Assert.AreEqual(1, _passManager.GetPasses(IRPassManager.PassType.Optimization).Count);
            Assert.AreEqual(1, _passManager.GetPasses(IRPassManager.PassType.Analysis).Count);
        }

        [Test]
        public void AddPass_WithNullPass_ThrowsArgumentNullException()
        {
            Assert.Throws<System.ArgumentNullException>(() => _passManager.AddPass(null, IRPassManager.PassType.Optimization));
        }

        [Test]
        public void RunAll_RunsAllRegisteredPasses()
        {
            var pass1 = new MockTransformation("Pass1");
            var pass2 = new MockTransformation("Pass2");

            _passManager.AddPass(pass1, IRPassManager.PassType.Optimization);
            _passManager.AddPass(pass2, IRPassManager.PassType.Optimization);

            var changed = _passManager.RunAll(_module);

            Assert.IsTrue(pass1.RunCalled);
            Assert.IsTrue(pass2.RunCalled);
            Assert.IsFalse(changed); // Mock pass returns false
        }

        [Test]
        public void RunAnalysisPasses_OnlyRunsAnalysisPasses()
        {
            var analysisPass = new MockTransformation("AnalysisPass", isAnalysisOnly: true);
            var optPass = new MockTransformation("OptPass");

            _passManager.AddPass(analysisPass, IRPassManager.PassType.Analysis);
            _passManager.AddPass(optPass, IRPassManager.PassType.Optimization);

            _passManager.RunAnalysisPasses(_module);

            Assert.IsTrue(analysisPass.RunCalled);
            Assert.IsFalse(optPass.RunCalled);
        }

        [Test]
        public void RunOptimizationPasses_OnlyRunsOptimizationPasses()
        {
            var analysisPass = new MockTransformation("AnalysisPass", isAnalysisOnly: true);
            var optPass = new MockTransformation("OptPass");

            _passManager.AddPass(analysisPass, IRPassManager.PassType.Analysis);
            _passManager.AddPass(optPass, IRPassManager.PassType.Optimization);

            _passManager.RunOptimizationPasses(_module);

            Assert.IsFalse(analysisPass.RunCalled);
            Assert.IsTrue(optPass.RunCalled);
        }

        [Test]
        public void RunValidationPasses_OnlyRunsValidationPasses()
        {
            var validationPass = new MockTransformation("ValidationPass", isAnalysisOnly: true);
            var optPass = new MockTransformation("OptPass");

            _passManager.AddPass(validationPass, IRPassManager.PassType.Validation);
            _passManager.AddPass(optPass, IRPassManager.PassType.Optimization);

            var valid = _passManager.RunValidationPasses(_module);

            Assert.IsTrue(validationPass.RunCalled);
            Assert.IsFalse(optPass.RunCalled);
            Assert.IsTrue(valid);
        }

        [Test]
        public void Clear_RemovesAllPasses()
        {
            _passManager.AddPass(new MockTransformation("Pass1"), IRPassManager.PassType.Optimization);
            _passManager.AddPass(new MockTransformation("Pass2"), IRPassManager.PassType.Analysis);

            _passManager.Clear();

            Assert.AreEqual(0, _passManager.PassCount);
            Assert.AreEqual(0, _passManager.GetPasses(IRPassManager.PassType.Optimization).Count);
            Assert.AreEqual(0, _passManager.GetPasses(IRPassManager.PassType.Analysis).Count);
        }

        // Mock transformation for testing
        private class MockTransformation : IRTransformation
        {
            public bool RunCalled { get; private set; }
            public bool InitializeCalled { get; private set; }
            public bool CleanupCalled { get; private set; }
            private readonly bool _returnValue;

            public MockTransformation(string name, bool isAnalysisOnly = false, bool returnValue = false)
                : base(name, isAnalysisOnly)
            {
                _returnValue = returnValue;
            }

            public override bool Run(HLIRModule module)
            {
                RunCalled = true;
                return _returnValue;
            }

            public override void Initialize(HLIRModule module)
            {
                base.Initialize(module);
                InitializeCalled = true;
            }

            public override void Cleanup()
            {
                base.Cleanup();
                CleanupCalled = true;
            }
        }
    }
}
