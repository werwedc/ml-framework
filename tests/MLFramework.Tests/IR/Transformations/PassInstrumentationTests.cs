using NUnit.Framework;
using MLFramework.IR.Transformations;

namespace MLFramework.Tests.IR.Transformations
{
    [TestFixture]
    public class PassInstrumentationTests
    {
        private PassInstrumentation _instrumentation;
        private MockTransformation _mockPass;
        private HLIRModule _module;

        [SetUp]
        public void Setup()
        {
            _instrumentation = new PassInstrumentation();
            _mockPass = new MockTransformation("TestPass");
            _module = new HLIRModule();
        }

        [Test]
        public void Constructor_InitializesEmptyEvents()
        {
            Assert.IsFalse(_instrumentation.HasListeners);
        }

        [Test]
        public void HasListeners_WithNoListeners_ReturnsFalse()
        {
            Assert.IsFalse(_instrumentation.HasListeners);
        }

        [Test]
        public void HasListeners_WithBeforePassListener_ReturnsTrue()
        {
            _instrumentation.BeforePass += (pass, module) => { };

            Assert.IsTrue(_instrumentation.HasListeners);
        }

        [Test]
        public void HasListeners_WithAfterPassListener_ReturnsTrue()
        {
            _instrumentation.AfterPass += (pass, module) => { };

            Assert.IsTrue(_instrumentation.HasListeners);
        }

        [Test]
        public void HasListeners_WithOnPassErrorListener_ReturnsTrue()
        {
            _instrumentation.OnPassError += (pass, module, ex) => { };

            Assert.IsTrue(_instrumentation.HasListeners);
        }

        [Test]
        public void NotifyBeforePass_WithListener_InvokesListener()
        {
            bool invoked = false;
            _instrumentation.BeforePass += (pass, module) => invoked = true;

            _instrumentation.NotifyBeforePass(_mockPass, _module);

            Assert.IsTrue(invoked);
        }

        [Test]
        public void NotifyBeforePass_PassesCorrectArguments()
        {
            IRTransformation receivedPass = null;
            HLIRModule receivedModule = null;

            _instrumentation.BeforePass += (pass, module) =>
            {
                receivedPass = pass;
                receivedModule = module;
            };

            _instrumentation.NotifyBeforePass(_mockPass, _module);

            Assert.AreSame(_mockPass, receivedPass);
            Assert.AreSame(_module, receivedModule);
        }

        [Test]
        public void NotifyBeforePass_WithNullPass_ThrowsArgumentNullException()
        {
            Assert.Throws<System.ArgumentNullException>(() => _instrumentation.NotifyBeforePass(null, _module));
        }

        [Test]
        public void NotifyBeforePass_WithNullModule_ThrowsArgumentNullException()
        {
            Assert.Throws<System.ArgumentNullException>(() => _instrumentation.NotifyBeforePass(_mockPass, null));
        }

        [Test]
        public void NotifyAfterPass_WithListener_InvokesListener()
        {
            bool invoked = false;
            _instrumentation.AfterPass += (pass, module) => invoked = true;

            _instrumentation.NotifyAfterPass(_mockPass, _module);

            Assert.IsTrue(invoked);
        }

        [Test]
        public void NotifyAfterPass_PassesCorrectArguments()
        {
            IRTransformation receivedPass = null;
            HLIRModule receivedModule = null;

            _instrumentation.AfterPass += (pass, module) =>
            {
                receivedPass = pass;
                receivedModule = module;
            };

            _instrumentation.NotifyAfterPass(_mockPass, _module);

            Assert.AreSame(_mockPass, receivedPass);
            Assert.AreSame(_module, receivedModule);
        }

        [Test]
        public void NotifyAfterPass_WithNullPass_ThrowsArgumentNullException()
        {
            Assert.Throws<System.ArgumentNullException>(() => _instrumentation.NotifyAfterPass(null, _module));
        }

        [Test]
        public void NotifyAfterPass_WithNullModule_ThrowsArgumentNullException()
        {
            Assert.Throws<System.ArgumentNullException>(() => _instrumentation.NotifyAfterPass(_mockPass, null));
        }

        [Test]
        public void NotifyPassError_WithListener_InvokesListener()
        {
            bool invoked = false;
            var exception = new System.Exception("Test error");
            _instrumentation.OnPassError += (pass, module, ex) => invoked = true;

            _instrumentation.NotifyPassError(_mockPass, _module, exception);

            Assert.IsTrue(invoked);
        }

        [Test]
        public void NotifyPassError_PassesCorrectArguments()
        {
            IRTransformation receivedPass = null;
            HLIRModule receivedModule = null;
            System.Exception receivedEx = null;
            var exception = new System.Exception("Test error");

            _instrumentation.OnPassError += (pass, module, ex) =>
            {
                receivedPass = pass;
                receivedModule = module;
                receivedEx = ex;
            };

            _instrumentation.NotifyPassError(_mockPass, _module, exception);

            Assert.AreSame(_mockPass, receivedPass);
            Assert.AreSame(_module, receivedModule);
            Assert.AreSame(exception, receivedEx);
        }

        [Test]
        public void NotifyPassError_WithNullPass_ThrowsArgumentNullException()
        {
            var exception = new System.Exception("Test error");
            Assert.Throws<System.ArgumentNullException>(() => _instrumentation.NotifyPassError(null, _module, exception));
        }

        [Test]
        public void NotifyPassError_WithNullModule_ThrowsArgumentNullException()
        {
            var exception = new System.Exception("Test error");
            Assert.Throws<System.ArgumentNullException>(() => _instrumentation.NotifyPassError(_mockPass, null, exception));
        }

        [Test]
        public void NotifyPassError_WithNullException_ThrowsArgumentNullException()
        {
            Assert.Throws<System.ArgumentNullException>(() => _instrumentation.NotifyPassError(_mockPass, _module, null));
        }

        [Test]
        public void ClearListeners_RemovesAllListeners()
        {
            _instrumentation.BeforePass += (pass, module) => { };
            _instrumentation.AfterPass += (pass, module) => { };
            _instrumentation.OnPassError += (pass, module, ex) => { };

            _instrumentation.ClearListeners();

            Assert.IsFalse(_instrumentation.HasListeners);
        }

        [Test]
        public void MultipleListeners_AllInvoked()
        {
            int beforeCount = 0;
            int afterCount = 0;

            _instrumentation.BeforePass += (pass, module) => beforeCount++;
            _instrumentation.AfterPass += (pass, module) => afterCount++;

            _instrumentation.NotifyBeforePass(_mockPass, _module);
            _instrumentation.NotifyAfterPass(_mockPass, _module);

            Assert.AreEqual(1, beforeCount);
            Assert.AreEqual(1, afterCount);
        }

        // Mock transformation for testing
        private class MockTransformation : IRTransformation
        {
            public MockTransformation(string name) : base(name, false) { }
            public override bool Run(HLIRModule module) => true;
        }
    }
}
