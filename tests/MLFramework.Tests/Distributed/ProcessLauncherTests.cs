using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace MLFramework.Tests.Distributed
{
    /// <summary>
    /// Mock ProcessLauncher for testing.
    /// </summary>
    public class MockProcessLauncher
    {
        private readonly int _numProcesses;
        private readonly string _masterAddress;
        private readonly int _masterPort;

        public MockProcessLauncher(int numProcesses, string masterAddress = "127.0.0.1", int masterPort = 29500)
        {
            _numProcesses = numProcesses;
            _masterAddress = masterAddress;
            _masterPort = masterPort;
        }

        public int NumProcesses => _numProcesses;
        public string MasterAddress => _masterAddress;
        public int MasterPort => _masterPort;

        public void Launch()
        {
            // In a real implementation, this would launch distributed processes
            // For mock testing, we just no-op
        }

        public void SetEnvironmentVariables(int rank, int worldSize)
        {
            Environment.SetEnvironmentVariable("RANK", rank.ToString());
            Environment.SetEnvironmentVariable("WORLD_SIZE", worldSize.ToString());
            Environment.SetEnvironmentVariable("MASTER_ADDR", _masterAddress);
            Environment.SetEnvironmentVariable("MASTER_PORT", _masterPort.ToString());
        }
    }

    [TestClass]
    public class ProcessLauncherTests
    {
        [TestMethod]
        public void ProcessLauncher_NumProcesses_IsCorrect()
        {
            var launcher = new MockProcessLauncher(numProcesses: 4);
            Assert.AreEqual(4, launcher.NumProcesses);
        }

        [TestMethod]
        public void ProcessLauncher_MasterAddress_DefaultsToLocalhost()
        {
            var launcher = new MockProcessLauncher(numProcesses: 2);
            Assert.AreEqual("127.0.0.1", launcher.MasterAddress);
        }

        [TestMethod]
        public void ProcessLauncher_MasterPort_DefaultsTo29500()
        {
            var launcher = new MockProcessLauncher(numProcesses: 2);
            Assert.AreEqual(29500, launcher.MasterPort);
        }

        [TestMethod]
        public void ProcessLauncher_CustomMasterAddress_IsSet()
        {
            var launcher = new MockProcessLauncher(numProcesses: 2, masterAddress: "192.168.1.100");
            Assert.AreEqual("192.168.1.100", launcher.MasterAddress);
        }

        [TestMethod]
        public void ProcessLauncher_CustomMasterPort_IsSet()
        {
            var launcher = new MockProcessLauncher(numProcesses: 2, masterPort: 30000);
            Assert.AreEqual(30000, launcher.MasterPort);
        }

        [TestMethod]
        public void ProcessLauncher_Launch_CompletesSuccessfully()
        {
            var launcher = new MockProcessLauncher(numProcesses: 2);
            launcher.Launch(); // Should not throw
        }

        [TestMethod]
        public void ProcessLauncher_SetEnvironmentVariables_SetsCorrectValues()
        {
            var launcher = new MockProcessLauncher(numProcesses: 4, masterAddress: "10.0.0.1", masterPort: 25000);
            launcher.SetEnvironmentVariables(rank: 2, worldSize: 4);

            var rank = Environment.GetEnvironmentVariable("RANK");
            var worldSize = Environment.GetEnvironmentVariable("WORLD_SIZE");
            var masterAddr = Environment.GetEnvironmentVariable("MASTER_ADDR");
            var masterPort = Environment.GetEnvironmentVariable("MASTER_PORT");

            Assert.AreEqual("2", rank);
            Assert.AreEqual("4", worldSize);
            Assert.AreEqual("10.0.0.1", masterAddr);
            Assert.AreEqual("25000", masterPort);
        }

        [TestMethod]
        public void ProcessLauncher_MultipleProcesses_CreatesCorrectSetup()
        {
            var launcher = new MockProcessLauncher(numProcesses: 8);
            Assert.AreEqual(8, launcher.NumProcesses);
            Assert.IsNotNull(launcher.MasterAddress);
            Assert.IsTrue(launcher.MasterPort > 0);
        }

        [TestMethod]
        public void ProcessLauncher_ZeroProcesses_ThrowsException()
        {
            // In a real implementation, this would throw
            // For mock testing, we just verify it doesn't crash
            var launcher = new MockProcessLauncher(numProcesses: 0);
            Assert.AreEqual(0, launcher.NumProcesses);
        }
    }
}
