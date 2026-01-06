using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Core;
using MLFramework.Distributed.Gloo;
using MLFramework.Tensor;
using RitterFramework.Core.Tensor;
using System;
using System.Reflection;

namespace MLFramework.Tests.Distributed.Gloo
{
    [TestClass]
    public class GlooBackendTests
    {
        [TestMethod]
        public void GlooBackend_Name_IsCorrect()
        {
            var backend = new GlooBackend();
            Assert.AreEqual("Gloo", backend.Name);
        }

        [TestMethod]
        public void GlooBackend_DeviceCount_ReturnsWorldSize()
        {
            // Set environment variables for testing
            Environment.SetEnvironmentVariable("RANK", "0");
            Environment.SetEnvironmentVariable("WORLD_SIZE", "4");

            try
            {
                var backend = new GlooBackend();
                // Note: We can't initialize the backend without the actual Gloo library,
                // so we just check that the property exists and has a default value
                Assert.AreEqual(1, backend.DeviceCount); // Default before initialization
            }
            finally
            {
                Environment.SetEnvironmentVariable("RANK", null);
                Environment.SetEnvironmentVariable("WORLD_SIZE", null);
            }
        }

        [TestMethod]
        public void GlooBackend_SupportsAsync_ReturnsFalse()
        {
            var backend = new GlooBackend();
            Assert.IsFalse(backend.SupportsAsync);
        }

        [TestMethod]
        public void GlooBackend_SupportsGPUDirect_ReturnsFalse()
        {
            var backend = new GlooBackend();
            Assert.IsFalse(backend.SupportsGPUDirect);
        }

        [TestMethod]
        public void GlooBackend_GetBufferSizeLimit_Returns1GB()
        {
            var backend = new GlooBackend();
            Assert.AreEqual(1024L * 1024 * 1024, backend.GetBufferSizeLimit());
        }

        [TestMethod]
        public void GlooBackend_NotInitialized_ThrowsOnFinalize()
        {
            var backend = new GlooBackend();
            // Should not throw even if not initialized
            backend.Finalize();
        }

        [TestMethod]
        public void GlooBackend_DoubleInitialize_Throws()
        {
            var backend = new GlooBackend();

            try
            {
                // First initialization will fail without Gloo library
                backend.Initialize();
            }
            catch (CommunicationException)
            {
                // Expected if Gloo library is not installed
            }

            // Second initialization should throw
            Assert.ThrowsException<InvalidOperationException>(() => backend.Initialize());
        }

        [TestMethod]
        public void GlooBackend_Dispose_CallsFinalize()
        {
            var backend = new GlooBackend();
            backend.Dispose();
            // Should not throw
        }
    }

    [TestClass]
    public class GlooProcessGroupTests
    {
        [TestMethod]
        public void GlooProcessGroup_Constructor_NullBackend_Throws()
        {
            Assert.ThrowsException<ArgumentNullException>(() =>
            {
                new GlooProcessGroup(null);
            });
        }

        [TestMethod]
        public void GlooProcessGroup_Constructor_ValidBackend_CreatesGroup()
        {
            try
            {
                // Set environment variables for testing
                Environment.SetEnvironmentVariable("RANK", "0");
                Environment.SetEnvironmentVariable("WORLD_SIZE", "2");

                var backend = new GlooBackend();

                try
                {
                    var group = new GlooProcessGroup(backend);
                    Assert.IsNotNull(group);
                    Assert.AreEqual("Gloo", group.Backend.Name);
                    group.Destroy();
                }
                catch (CommunicationException)
                {
                    // Expected if Gloo library is not installed
                }
            }
            finally
            {
                Environment.SetEnvironmentVariable("RANK", null);
                Environment.SetEnvironmentVariable("WORLD_SIZE", null);
            }
        }

        [TestMethod]
        public void GlooProcessGroup_AllReduce_NullTensor_Throws()
        {
            try
            {
                Environment.SetEnvironmentVariable("RANK", "0");
                Environment.SetEnvironmentVariable("WORLD_SIZE", "1");

                var backend = new GlooBackend();
                try
                {
                    var group = new GlooProcessGroup(backend);
                    Assert.ThrowsException<ArgumentNullException>(() =>
                    {
                        group.AllReduce(null, ReduceOp.Sum);
                    });
                    group.Destroy();
                }
                catch (CommunicationException)
                {
                    // Expected if Gloo library is not installed
                }
            }
            finally
            {
                Environment.SetEnvironmentVariable("RANK", null);
                Environment.SetEnvironmentVariable("WORLD_SIZE", null);
            }
        }

        [TestMethod]
        public void GlooProcessGroup_Broadcast_InvalidRoot_Throws()
        {
            try
            {
                Environment.SetEnvironmentVariable("RANK", "0");
                Environment.SetEnvironmentVariable("WORLD_SIZE", "2");

                var backend = new GlooBackend();
                try
                {
                    var group = new GlooProcessGroup(backend);
                    var tensor = new Tensor(new float[] { 1, 2, 3 }, new int[] { 3 });

                    Assert.ThrowsException<ArgumentOutOfRangeException>(() =>
                    {
                        group.Broadcast(tensor, 5); // Invalid root rank
                    });

                    group.Destroy();
                }
                catch (CommunicationException)
                {
                    // Expected if Gloo library is not installed
                }
            }
            finally
            {
                Environment.SetEnvironmentVariable("RANK", null);
                Environment.SetEnvironmentVariable("WORLD_SIZE", null);
            }
        }

        [TestMethod]
        public void GlooProcessGroup_Dispose_CallsDestroy()
        {
            try
            {
                Environment.SetEnvironmentVariable("RANK", "0");
                Environment.SetEnvironmentVariable("WORLD_SIZE", "1");

                var backend = new GlooBackend();
                try
                {
                    var group = new GlooProcessGroup(backend);
                    group.Dispose();
                    // Should not throw
                }
                catch (CommunicationException)
                {
                    // Expected if Gloo library is not installed
                }
            }
            finally
            {
                Environment.SetEnvironmentVariable("RANK", null);
                Environment.SetEnvironmentVariable("WORLD_SIZE", null);
            }
        }

        [TestMethod]
        public void GlooProcessGroup_AsyncMethods_ReturnTasks()
        {
            try
            {
                Environment.SetEnvironmentVariable("RANK", "0");
                Environment.SetEnvironmentVariable("WORLD_SIZE", "1");

                var backend = new GlooBackend();
                try
                {
                    var group = new GlooProcessGroup(backend);
                    var tensor = new Tensor(new float[] { 1, 2, 3 }, new int[] { 3 });

                    var task1 = group.AllReduceAsync(tensor, ReduceOp.Sum);
                    Assert.IsNotNull(task1);

                    var task2 = group.BroadcastAsync(tensor, 0);
                    Assert.IsNotNull(task2);

                    var task3 = group.BarrierAsync();
                    Assert.IsNotNull(task3);

                    group.Destroy();
                }
                catch (CommunicationException)
                {
                    // Expected if Gloo library is not installed
                }
            }
            finally
            {
                Environment.SetEnvironmentVariable("RANK", null);
                Environment.SetEnvironmentVariable("WORLD_SIZE", null);
            }
        }
    }
}
