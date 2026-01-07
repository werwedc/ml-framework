using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Distributed.FSDP;
using System;

namespace MLFramework.Tests.Distributed.FSDP
{
    [TestClass]
    public class FSDPConfigTests
    {
        [TestMethod]
        public void FSDPConfig_DefaultValues_AreCorrect()
        {
            var config = new FSDPConfig();

            Assert.AreEqual(ShardingStrategy.Full, config.ShardingStrategy);
            Assert.IsTrue(config.MixedPrecision);
            Assert.IsFalse(config.OffloadToCPU);
            Assert.IsFalse(config.ActivationCheckpointing);
            Assert.AreEqual(25, config.BucketSizeMB);
            Assert.AreEqual(2, config.NumCommunicationWorkers);
        }

        [TestMethod]
        public void FSDPConfig_ShardingStrategy_CanBeSet()
        {
            var config = new FSDPConfig();

            config.ShardingStrategy = ShardingStrategy.LayerWise;
            Assert.AreEqual(ShardingStrategy.LayerWise, config.ShardingStrategy);

            config.ShardingStrategy = ShardingStrategy.Hybrid;
            Assert.AreEqual(ShardingStrategy.Hybrid, config.ShardingStrategy);
        }

        [TestMethod]
        public void FSDPConfig_MixedPrecision_CanBeSet()
        {
            var config = new FSDPConfig
            {
                MixedPrecision = false
            };

            Assert.IsFalse(config.MixedPrecision);
        }

        [TestMethod]
        public void FSDPConfig_OffloadToCPU_CanBeSet()
        {
            var config = new FSDPConfig
            {
                OffloadToCPU = true
            };

            Assert.IsTrue(config.OffloadToCPU);
        }

        [TestMethod]
        public void FSDPConfig_ActivationCheckpointing_CanBeSet()
        {
            var config = new FSDPConfig
            {
                ActivationCheckpointing = true
            };

            Assert.IsTrue(config.ActivationCheckpointing);
        }

        [TestMethod]
        public void FSDPConfig_BucketSizeMB_CanBeSet()
        {
            var config = new FSDPConfig
            {
                BucketSizeMB = 50
            };

            Assert.AreEqual(50, config.BucketSizeMB);
        }

        [TestMethod]
        public void FSDPConfig_NumCommunicationWorkers_CanBeSet()
        {
            var config = new FSDPConfig
            {
                NumCommunicationWorkers = 4
            };

            Assert.AreEqual(4, config.NumCommunicationWorkers);
        }

        [TestMethod]
        public void FSDPConfig_Validate_ValidBucketSize_DoesNotThrow()
        {
            var config = new FSDPConfig { BucketSizeMB = 100 };

            config.Validate(); // Should not throw
        }

        [TestMethod]
        public void FSDPConfig_Validate_ValidCommunicationWorkers_DoesNotThrow()
        {
            var config = new FSDPConfig { NumCommunicationWorkers = 8 };

            config.Validate(); // Should not throw
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void FSDPConfig_Validate_BucketSizeZero_ThrowsArgumentException()
        {
            var config = new FSDPConfig { BucketSizeMB = 0 };

            config.Validate();
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void FSDPConfig_Validate_BucketSizeNegative_ThrowsArgumentException()
        {
            var config = new FSDPConfig { BucketSizeMB = -10 };

            config.Validate();
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void FSDPConfig_Validate_BucketSizeTooLarge_ThrowsArgumentException()
        {
            var config = new FSDPConfig { BucketSizeMB = 1001 };

            config.Validate();
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void FSDPConfig_Validate_BucketSizeExactlyMax_DoesNotThrow()
        {
            var config = new FSDPConfig { BucketSizeMB = 1000 };

            config.Validate(); // Should not throw as 1000 is allowed
        }

        [TestMethod]
        public void FSDPConfig_Validate_BucketSizeExactlyMin_DoesNotThrow()
        {
            var config = new FSDPConfig { BucketSizeMB = 1 };

            config.Validate(); // Should not throw as 1 is allowed
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void FSDPConfig_Validate_NumCommunicationWorkersZero_ThrowsArgumentException()
        {
            var config = new FSDPConfig { NumCommunicationWorkers = 0 };

            config.Validate();
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void FSDPConfig_Validate_NumCommunicationWorkersNegative_ThrowsArgumentException()
        {
            var config = new FSDPConfig { NumCommunicationWorkers = -5 };

            config.Validate();
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void FSDPConfig_Validate_NumCommunicationWorkersTooLarge_ThrowsArgumentException()
        {
            var config = new FSDPConfig { NumCommunicationWorkers = 17 };

            config.Validate();
        }

        [TestMethod]
        public void FSDPConfig_Validate_NumCommunicationWorkersExactlyMax_DoesNotThrow()
        {
            var config = new FSDPConfig { NumCommunicationWorkers = 16 };

            config.Validate(); // Should not throw as 16 is allowed
        }

        [TestMethod]
        public void FSDPConfig_Validate_NumCommunicationWorkersExactlyMin_DoesNotThrow()
        {
            var config = new FSDPConfig { NumCommunicationWorkers = 1 };

            config.Validate(); // Should not throw as 1 is allowed
        }

        [TestMethod]
        public void FSDPConfig_Validate_MultipleInvalidProperties_ThrowsOnFirst()
        {
            var config = new FSDPConfig
            {
                BucketSizeMB = 0,
                NumCommunicationWorkers = 0
            };

            try
            {
                config.Validate();
                Assert.Fail("Expected ArgumentException to be thrown");
            }
            catch (ArgumentException ex)
            {
                // Should throw because of BucketSizeMB
                Assert.IsTrue(ex.Message.Contains("BucketSizeMB"));
            }
        }

        [TestMethod]
        public void FSDPConfig_FullConfiguration_ValidatesSuccessfully()
        {
            var config = new FSDPConfig
            {
                ShardingStrategy = ShardingStrategy.Hybrid,
                MixedPrecision = false,
                OffloadToCPU = true,
                ActivationCheckpointing = true,
                BucketSizeMB = 500,
                NumCommunicationWorkers = 4
            };

            config.Validate(); // Should not throw

            Assert.AreEqual(ShardingStrategy.Hybrid, config.ShardingStrategy);
            Assert.IsFalse(config.MixedPrecision);
            Assert.IsTrue(config.OffloadToCPU);
            Assert.IsTrue(config.ActivationCheckpointing);
            Assert.AreEqual(500, config.BucketSizeMB);
            Assert.AreEqual(4, config.NumCommunicationWorkers);
        }
    }

    [TestClass]
    public class FSDPStateTests
    {
        [TestMethod]
        public void FSDPState_DefaultValues_AreZero()
        {
            var state = new FSDPState();

            Assert.AreEqual(0, state.OwnerRank);
            Assert.AreEqual(0, state.NumShards);
            Assert.AreEqual(0, state.ShardIndex);
            Assert.IsFalse(state.IsGathered);
            Assert.IsFalse(state.IsOffloaded);
        }

        [TestMethod]
        public void FSDPState_OwnerRank_CanBeSet()
        {
            var state = new FSDPState
            {
                OwnerRank = 5
            };

            Assert.AreEqual(5, state.OwnerRank);
        }

        [TestMethod]
        public void FSDPState_NumShards_CanBeSet()
        {
            var state = new FSDPState
            {
                NumShards = 8
            };

            Assert.AreEqual(8, state.NumShards);
        }

        [TestMethod]
        public void FSDPState_ShardIndex_CanBeSet()
        {
            var state = new FSDPState
            {
                ShardIndex = 3
            };

            Assert.AreEqual(3, state.ShardIndex);
        }

        [TestMethod]
        public void FSDPState_IsGathered_CanBeSet()
        {
            var state = new FSDPState
            {
                IsGathered = true
            };

            Assert.IsTrue(state.IsGathered);
        }

        [TestMethod]
        public void FSDPState_IsOffloaded_CanBeSet()
        {
            var state = new FSDPState
            {
                IsOffloaded = true
            };

            Assert.IsTrue(state.IsOffloaded);
        }

        [TestMethod]
        public void FSDPState_FullConfiguration_CanBeSet()
        {
            var state = new FSDPState
            {
                OwnerRank = 2,
                NumShards = 4,
                ShardIndex = 1,
                IsGathered = true,
                IsOffloaded = false
            };

            Assert.AreEqual(2, state.OwnerRank);
            Assert.AreEqual(4, state.NumShards);
            Assert.AreEqual(1, state.ShardIndex);
            Assert.IsTrue(state.IsGathered);
            Assert.IsFalse(state.IsOffloaded);
        }

        [TestMethod]
        public void FSDPState_CanTrackGatheredAndOffloadedStates()
        {
            var state = new FSDPState();

            // Initially neither gathered nor offloaded
            Assert.IsFalse(state.IsGathered);
            Assert.IsFalse(state.IsOffloaded);

            // Mark as gathered
            state.IsGathered = true;
            Assert.IsTrue(state.IsGathered);
            Assert.IsFalse(state.IsOffloaded);

            // Mark as offloaded (still gathered)
            state.IsOffloaded = true;
            Assert.IsTrue(state.IsGathered);
            Assert.IsTrue(state.IsOffloaded);

            // Ungather
            state.IsGathered = false;
            Assert.IsFalse(state.IsGathered);
            Assert.IsTrue(state.IsOffloaded);
        }
    }
}
