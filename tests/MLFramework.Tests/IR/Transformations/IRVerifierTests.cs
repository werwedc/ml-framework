using NUnit.Framework;
using MLFramework.IR.Transformations;

namespace MLFramework.Tests.IR.Transformations
{
    [TestFixture]
    public class IRVerifierTests
    {
        private IRVerifier _verifier;
        private HLIRModule _module;

        [SetUp]
        public void Setup()
        {
            _verifier = new IRVerifier();
            _module = new HLIRModule();
        }

        [Test]
        public void Constructor_InitializesEmptyErrorsAndWarnings()
        {
            Assert.AreEqual(0, _verifier.ErrorCount);
            Assert.AreEqual(0, _verifier.WarningCount);
        }

        [Test]
        public void Run_WithNullModule_AddsError()
        {
            var result = _verifier.Run(null);

            Assert.IsFalse(result);
            Assert.AreEqual(1, _verifier.ErrorCount);
            Assert.That(_verifier.Errors, Contains.Item("Module is null"));
        }

        [Test]
        public void Run_WithEmptyModule_ReturnsTrue()
        {
            var result = _verifier.Run(_module);

            Assert.IsTrue(result);
            Assert.AreEqual(0, _verifier.ErrorCount);
            Assert.AreEqual(0, _verifier.WarningCount);
        }

        [Test]
        public void Run_WithFunctionWithoutName_AddsError()
        {
            // Note: This test assumes we can create a function without a name
            // In practice, the constructor might enforce this
            var func = _module.CreateFunction("");  // Empty name

            var result = _verifier.Run(_module);

            Assert.IsFalse(result);
            Assert.IsTrue(_verifier.ErrorCount > 0);
        }

        [Test]
        public void GetErrorMessage_WithNoErrors_ReturnsMessage()
        {
            var message = _verifier.GetErrorMessage();

            Assert.AreEqual("No errors found", message);
        }

        [Test]
        public void GetErrorMessage_WithErrors_ReturnsFormattedMessage()
        {
            _verifier.Run(null);  // Will add an error

            var message = _verifier.GetErrorMessage();

            Assert.That(message, Does.StartWith("Found 1 error(s):"));
            Assert.That(message, Does.Contain("Module is null"));
        }

        [Test]
        public void GetWarningMessage_WithNoWarnings_ReturnsMessage()
        {
            var message = _verifier.GetWarningMessage();

            Assert.AreEqual("No warnings found", message);
        }

        [Test]
        public void GetFullReport_WithNoErrorsOrWarnings_ReturnsSuccessMessage()
        {
            _verifier.Run(_module);

            var report = _verifier.GetFullReport();

            Assert.AreEqual("Verification passed: No errors or warnings found", report);
        }

        [Test]
        public void GetFullReport_WithErrors_ReturnsErrorMessage()
        {
            _verifier.Run(null);

            var report = _verifier.GetFullReport();

            Assert.That(report, Does.Contain("Found 1 error(s):"));
            Assert.That(report, Does.Contain("Module is null"));
        }

        [Test]
        public void Verify_NameProperty_ReturnsVerifier()
        {
            Assert.AreEqual("Verifier", _verifier.Name);
        }

        [Test]
        public void Verify_IsAnalysisOnly_ReturnsTrue()
        {
            Assert.IsTrue(_verifier.IsAnalysisOnly);
        }

        [Test]
        public void ErrorsAndWarnings_AreAccessible()
        {
            _verifier.Run(null);

            Assert.IsNotNull(_verifier.Errors);
            Assert.IsNotNull(_verifier.Warnings);
            Assert.AreEqual(1, _verifier.Errors.Count);
        }
    }
}
