using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Data;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.Tests.Distributed
{
    [TestClass]
    public class DistributedSamplerTests
    {
        [TestMethod]
        public void DistributedSampler_Partitions_CorrectlyAcrossRanks()
        {
            int datasetSize = 100;
            int worldSize = 4;

            var allIndices = new List<int>[worldSize];
            for (int rank = 0; rank < worldSize; rank++)
            {
                var sampler = new DistributedSampler(
                    datasetSize: datasetSize,
                    numReplicas: worldSize,
                    rank: rank,
                    shuffle: false
                );
                allIndices[rank] = sampler.Iterate().ToList();
            }

            // Check that all indices are covered without overlap
            var allCombined = allIndices.SelectMany(i => i).OrderBy(i => i).ToList();
            var expected = Enumerable.Range(0, 100).ToList();

            CollectionAssert.AreEqual(expected, allCombined);

            // Check no overlaps
            int distinctCount = allCombined.Distinct().Count();
            Assert.AreEqual(100, distinctCount);
        }

        [TestMethod]
        public void DistributedSampler_SetEpoch_ChangesShuffleOrder()
        {
            int datasetSize = 100;

            var sampler1 = new DistributedSampler(
                datasetSize: datasetSize,
                numReplicas: 2,
                rank: 0,
                shuffle: true,
                seed: 42
            );
            sampler1.SetEpoch(0);
            var indices1 = sampler1.Iterate().ToList();

            var sampler2 = new DistributedSampler(
                datasetSize: datasetSize,
                numReplicas: 2,
                rank: 0,
                shuffle: true,
                seed: 42
            );
            sampler2.SetEpoch(1);
            var indices2 = sampler2.Iterate().ToList();

            // Should be different due to different epochs
            CollectionAssert.AreNotEqual(indices1, indices2);
        }

        [TestMethod]
        public void DistributedSampler_SameEpochSameSeed_SameOrder()
        {
            int datasetSize = 100;

            var sampler1 = new DistributedSampler(
                datasetSize: datasetSize,
                numReplicas: 2,
                rank: 0,
                shuffle: true,
                seed: 42
            );
            sampler1.SetEpoch(0);
            var indices1 = sampler1.Iterate().ToList();

            var sampler2 = new DistributedSampler(
                datasetSize: datasetSize,
                numReplicas: 2,
                rank: 0,
                shuffle: true,
                seed: 42
            );
            sampler2.SetEpoch(0);
            var indices2 = sampler2.Iterate().ToList();

            // Should be identical
            CollectionAssert.AreEqual(indices1, indices2);
        }

        [TestMethod]
        public void DistributedSampler_NoShuffle_KeepsOrder()
        {
            int datasetSize = 50;

            var sampler = new DistributedSampler(
                datasetSize: datasetSize,
                numReplicas: 2,
                rank: 0,
                shuffle: false
            );

            var indices = sampler.Iterate().ToList();

            // First half (0-24) for rank 0
            var expected = Enumerable.Range(0, 25).ToList();
            CollectionAssert.AreEqual(expected, indices);
        }

        [TestMethod]
        public void DistributedSampler_DropLast_RemovesUnevenSamples()
        {
            int datasetSize = 103; // Not evenly divisible by 4
            int worldSize = 4;

            var sampler = new DistributedSampler(
                datasetSize: datasetSize,
                numReplicas: worldSize,
                rank: 0,
                dropLast: true
            );

            var indices = sampler.Iterate().ToList();

            // Should have 100/4 = 25 samples (103 - 3 to make it divisible)
            Assert.AreEqual(25, indices.Count);
        }

        [TestMethod]
        public void DistributedSampler_RankAndWorldSize_AreCorrect()
        {
            int datasetSize = 100;
            int worldSize = 8;
            int rank = 3;

            var sampler = new DistributedSampler(
                datasetSize: datasetSize,
                numReplicas: worldSize,
                rank: rank
            );

            Assert.AreEqual(worldSize, sampler.NumReplicas);
            Assert.AreEqual(rank, sampler.Rank);
        }

        [TestMethod]
        public void DistributedSampler_Episode_DefaultsToZero()
        {
            int datasetSize = 100;

            var sampler = new DistributedSampler(
                datasetSize: datasetSize,
                numReplicas: 2,
                rank: 0
            );

            Assert.AreEqual(0, sampler.Epoch);
        }

        [TestMethod]
        public void DistributedSampler_SetEpoch_UpdatesEpisodeProperty()
        {
            int datasetSize = 100;

            var sampler = new DistributedSampler(
                datasetSize: datasetSize,
                numReplicas: 2,
                rank: 0
            );

            sampler.SetEpoch(5);
            Assert.AreEqual(5, sampler.Epoch);
        }

        [TestMethod]
        public void DistributedSampler_NegativeEpoch_ThrowsException()
        {
            int datasetSize = 100;

            var sampler = new DistributedSampler(
                datasetSize: datasetSize,
                numReplicas: 2,
                rank: 0
            );

            Assert.ThrowsException<System.ArgumentOutOfRangeException>(() =>
            {
                sampler.SetEpoch(-1);
            });
        }

        [TestMethod]
        public void DistributedSampler_MultipleRanks_CoverAllData()
        {
            int datasetSize = 200;
            int worldSize = 5;

            var allIndices = new HashSet<int>();

            for (int rank = 0; rank < worldSize; rank++)
            {
                var sampler = new DistributedSampler(
                    datasetSize: datasetSize,
                    numReplicas: worldSize,
                    rank: rank,
                    shuffle: false
                );

                var indices = sampler.Iterate();
                foreach (var idx in indices)
                {
                    Assert.IsFalse(allIndices.Contains(idx), $"Index {idx} appears in multiple ranks");
                    allIndices.Add(idx);
                }
            }

            // Should cover all indices
            Assert.AreEqual(datasetSize, allIndices.Count);
            for (int i = 0; i < datasetSize; i++)
            {
                Assert.IsTrue(allIndices.Contains(i), $"Index {i} not covered by any rank");
            }
        }

        [TestMethod]
        public void DistributedSampler_Length_ReturnsCorrectCount()
        {
            int datasetSize = 100;
            int worldSize = 4;

            var sampler = new DistributedSampler(
                datasetSize: datasetSize,
                numReplicas: worldSize,
                rank: 0
            );

            // 100 / 4 = 25
            Assert.AreEqual(25, sampler.Length);
        }

        [TestMethod]
        public void DistributedSampler_NegativeDatasetSize_ThrowsException()
        {
            Assert.ThrowsException<System.ArgumentOutOfRangeException>(() =>
            {
                var sampler = new DistributedSampler(
                    datasetSize: -10,
                    numReplicas: 2,
                    rank: 0
                );
            });
        }

        [TestMethod]
        public void DistributedSampler_InvalidRank_ThrowsException()
        {
            int datasetSize = 100;

            Assert.ThrowsException<System.ArgumentOutOfRangeException>(() =>
            {
                var sampler = new DistributedSampler(
                    datasetSize: datasetSize,
                    numReplicas: 4,
                    rank: 5  // Invalid rank (>= numReplicas)
                );
            });
        }
    }
}
