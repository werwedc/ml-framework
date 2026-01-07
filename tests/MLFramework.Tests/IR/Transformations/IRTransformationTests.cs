using NUnit.Framework;
using MLFramework.IR.Transformations;

namespace MLFramework.Tests.IR.Transformations
{
    [TestFixture]
    public class IRTransformationTests
    {
        private class TestTransformation : IRTransformation
        {
            public bool InitializeCalled { get; private set; }
            public bool CleanupCalled { get; private set; }
            public bool RunCalled { get; private set; }
            private readonly bool _returnValue;

            public TestTransformation(string name, bool isAnalysisOnly = false, bool returnValue = false)
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

        [Test]
        public void Constructor_SetsNameProperty()
        {
            var transform = new TestTransformation("TestTransform");

            Assert.AreEqual("TestTransform", transform.Name);
        }

        [Test]
        public void Constructor_WithNullName_ThrowsArgumentNullException()
        {
            Assert.Throws<System.ArgumentNullException>(() => new TestTransformation(null));
        }

        [Test]
        public void Constructor_DefaultIsAnalysisOnly_ReturnsFalse()
        {
            var transform = new TestTransformation("TestTransform");

            Assert.IsFalse(transform.IsAnalysisOnly);
        }

        [Test]
        public void Constructor_WithIsAnalysisOnlyTrue_SetsProperty()
        {
            var transform = new TestTransformation("TestTransform", isAnalysisOnly: true);

            Assert.IsTrue(transform.IsAnalysisOnly);
        }

        [Test]
        public void Initialize_CallsBaseInitialize()
        {
            var transform = new TestTransformation("TestTransform");
            var module = new HLIRModule();

            transform.Initialize(module);

            Assert.IsTrue(transform.InitializeCalled);
        }

        [Test]
        public void Run_CallsDerivedRunMethod()
        {
            var transform = new TestTransformation("TestTransform", returnValue: true);
            var module = new HLIRModule();

            var result = transform.Run(module);

            Assert.IsTrue(transform.RunCalled);
            Assert.IsTrue(result);
        }

        [Test]
        public void Cleanup_CallsBaseCleanup()
        {
            var transform = new TestTransformation("TestTransform");

            transform.Cleanup();

            Assert.IsTrue(transform.CleanupCalled);
        }

        [Test]
        public void InitializeThenRunThenCleanup_AllMethodsCalled()
        {
            var transform = new TestTransformation("TestTransform");
            var module = new HLIRModule();

            transform.Initialize(module);
            transform.Run(module);
            transform.Cleanup();

            Assert.IsTrue(transform.InitializeCalled);
            Assert.IsTrue(transform.RunCalled);
            Assert.IsTrue(transform.CleanupCalled);
        }

        [Test]
        public void NameProperty_IsReadOnly()
        {
            var transform = new TestTransformation("TestTransform");
            var name = transform.Name;

            // Name should not be modifiable
            Assert.AreEqual("TestTransform", name);
        }

        [Test]
        public void IsAnalysisOnlyProperty_IsReadOnly()
        {
            var transform = new TestTransformation("TestTransform", isAnalysisOnly: true);
            var isAnalysisOnly = transform.IsAnalysisOnly;

            Assert.IsTrue(isAnalysisOnly);
        }
    }
}
