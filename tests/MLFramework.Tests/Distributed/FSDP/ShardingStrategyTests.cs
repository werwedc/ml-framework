using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Distributed.FSDP;
using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.Tests.Distributed.FSDP
{
    /// <summary>
    /// Unit tests for sharding strategies.
    /// </summary>
    [TestClass]
    public class ShardingStrategyTests
    {
        [TestMethod]
        public void TestFullShardingStrategy()
        {
            var strategy = new FullShardingStrategy();

            var parameters = new List<ParameterInfo>
            {
                new ParameterInfo
                {
                    Name = "param1",
                    Shape = new[] { 1000L },
                    SizeBytes = 4000,
                    LayerName = "layer1",
                    AlwaysGather = false
                },
                new ParameterInfo
                {
                    Name = "param2",
                    Shape = new[] { 500L },
                    SizeBytes = 2000,
                    LayerName = "layer1",
                    AlwaysGather = false
                }
            };

            var plan = strategy.CalculateShardingPlan(parameters, 4);

            Assert.AreEqual(4, plan.TotalShards);
            Assert.IsTrue(plan.Assignments.Count > 0);
            Assert.AreEqual(0, plan.AlwaysGathered.Count);
        }

        [TestMethod]
        public void TestFullShardingStrategyProperSharding()
        {
            var strategy = new FullShardingStrategy();

            var parameters = new List<ParameterInfo>
            {
                new ParameterInfo
                {
                    Name = "test_param",
                    Shape = new[] { 1000L },
                    SizeBytes = 4000,
                    LayerName = "test",
                    AlwaysGather = false
                }
            };

            var plan = strategy.CalculateShardingPlan(parameters, 4);

            // Should create 4 shards
            Assert.AreEqual(4, plan.Assignments.Count);

            // Each shard should have correct properties
            foreach (var kvp in plan.Assignments)
            {
                var assignment = kvp.Value;
                Assert.IsTrue(assignment.OwnerRank >= 0 && assignment.OwnerRank < 4);
                Assert.IsTrue(assignment.ShardIndex >= 0 && assignment.ShardIndex < 4);
                Assert.IsTrue(assignment.StartOffset >= 0);
                Assert.IsTrue(assignment.ShardSize > 0);
            }
        }

        [TestMethod]
        public void TestLayerWiseShardingStrategy()
        {
            var strategy = new LayerWiseShardingStrategy();

            var parameters = new List<ParameterInfo>
            {
                new ParameterInfo
                {
                    Name = "layer1.weight",
                    Shape = new[] { 1000L },
                    SizeBytes = 4000,
                    LayerName = "layer1",
                    AlwaysGather = false
                },
                new ParameterInfo
                {
                    Name = "layer2.weight",
                    Shape = new[] { 500L },
                    SizeBytes = 2000,
                    LayerName = "layer2",
                    AlwaysGather = false
                }
            };

            var plan = strategy.CalculateShardingPlan(parameters, 4);

            Assert.AreEqual(4, plan.TotalShards);
            Assert.IsTrue(plan.Assignments.Count > 0);
        }

        [TestMethod]
        public void TestHybridShardingStrategy()
        {
            var fullShardedLayers = new List<string> { "transformer" };
            var layerWiseShardedLayers = new List<string> { "classifier" };

            var strategy = new HybridShardingStrategy(fullShardedLayers, layerWiseShardedLayers);

            var parameters = new List<ParameterInfo>
            {
                new ParameterInfo
                {
                    Name = "transformer.weight",
                    Shape = new[] { 1000L },
                    SizeBytes = 4000,
                    LayerName = "transformer",
                    AlwaysGather = false
                },
                new ParameterInfo
                {
                    Name = "classifier.weight",
                    Shape = new[] { 500L },
                    SizeBytes = 2000,
                    LayerName = "classifier",
                    AlwaysGather = false
                }
            };

            var plan = strategy.CalculateShardingPlan(parameters, 4);

            Assert.AreEqual(4, plan.TotalShards);
            Assert.IsTrue(plan.Assignments.Count > 0);
        }

        [TestMethod]
        public void TestAlwaysGatheredParameters()
        {
            var strategy = new FullShardingStrategy();

            var parameters = new List<ParameterInfo>
            {
                new ParameterInfo
                {
                    Name = "embedding.weight",
                    Shape = new[] { 10000L },
                    SizeBytes = 40000,
                    LayerName = "embedding",
                    AlwaysGather = true
                }
            };

            var plan = strategy.CalculateShardingPlan(parameters, 4);

            Assert.AreEqual(1, plan.AlwaysGathered.Count);
            Assert.IsTrue(plan.AlwaysGathered.Contains("embedding.weight"));
            Assert.IsFalse(plan.Assignments.ContainsKey("embedding.weight_rank0"));
            Assert.IsFalse(plan.Assignments.ContainsKey("embedding.weight"));
        }

        [TestMethod]
        public void TestFullShardingStrategyWithMixedParameters()
        {
            var strategy = new FullShardingStrategy();

            var parameters = new List<ParameterInfo>
            {
                new ParameterInfo
                {
                    Name = "param1",
                    Shape = new[] { 1000L },
                    SizeBytes = 4000,
                    LayerName = "layer1",
                    AlwaysGather = false
                },
                new ParameterInfo
                {
                    Name = "always_gather_param",
                    Shape = new[] { 500L },
                    SizeBytes = 2000,
                    LayerName = "layer1",
                    AlwaysGather = true
                },
                new ParameterInfo
                {
                    Name = "param2",
                    Shape = new[] { 2000L },
                    SizeBytes = 8000,
                    LayerName = "layer2",
                    AlwaysGather = false
                }
            };

            var plan = strategy.CalculateShardingPlan(parameters, 4);

            Assert.AreEqual(1, plan.AlwaysGathered.Count);
            Assert.IsTrue(plan.AlwaysGathered.Contains("always_gather_param"));

            // Should have shards for param1 and param2, but not always_gather_param
            Assert.IsTrue(plan.Assignments.Any(kvp => kvp.Key.StartsWith("param1")));
            Assert.IsTrue(plan.Assignments.Any(kvp => kvp.Key.StartsWith("param2")));
            Assert.IsFalse(plan.Assignments.Any(kvp => kvp.Key.StartsWith("always_gather_param")));
        }

        [TestMethod]
        public void TestShardingAssignmentCorrectness()
        {
            var strategy = new FullShardingStrategy();

            var parameters = new List<ParameterInfo>
            {
                new ParameterInfo
                {
                    Name = "test_param",
                    Shape = new[] { 1000L },
                    SizeBytes = 4000,
                    LayerName = "test",
                    AlwaysGather = false
                }
            };

            var worldSize = 4;
            var plan = strategy.CalculateShardingPlan(parameters, worldSize);

            // Check that each shard has correct properties
            for (int rank = 0; rank < worldSize; rank++)
            {
                var key = $"test_param_rank{rank}";
                Assert.IsTrue(plan.Assignments.ContainsKey(key), $"Missing assignment for rank {rank}");

                var assignment = plan.Assignments[key];
                Assert.AreEqual(rank, assignment.OwnerRank);
                Assert.AreEqual(rank, assignment.ShardIndex);
                Assert.IsTrue(assignment.ShardSize > 0);
                Assert.IsTrue(assignment.StartOffset >= 0);
            }
        }

        [TestMethod]
        public void TestLayerWiseShardingMultipleLayers()
        {
            var strategy = new LayerWiseShardingStrategy();

            var parameters = new List<ParameterInfo>
            {
                new ParameterInfo
                {
                    Name = "layer1.weight",
                    Shape = new[] { 1000L },
                    SizeBytes = 4000,
                    LayerName = "layer1",
                    AlwaysGather = false
                },
                new ParameterInfo
                {
                    Name = "layer1.bias",
                    Shape = new[] { 100L },
                    SizeBytes = 400,
                    LayerName = "layer1",
                    AlwaysGather = false
                },
                new ParameterInfo
                {
                    Name = "layer2.weight",
                    Shape = new[] { 500L },
                    SizeBytes = 2000,
                    LayerName = "layer2",
                    AlwaysGather = false
                }
            };

            var plan = strategy.CalculateShardingPlan(parameters, 4);

            Assert.AreEqual(3, plan.Assignments.Count);
        }

        [TestMethod]
        public void TestHybridStrategyWithEmptyLists()
        {
            var strategy = new HybridShardingStrategy(new List<string>(), new List<string>());

            var parameters = new List<ParameterInfo>
            {
                new ParameterInfo
                {
                    Name = "test_param",
                    Shape = new[] { 1000L },
                    SizeBytes = 4000,
                    LayerName = "test",
                    AlwaysGather = false
                }
            };

            var plan = strategy.CalculateShardingPlan(parameters, 2);

            // Should default to full sharding
            Assert.IsTrue(plan.Assignments.Any(kvp => kvp.Key.StartsWith("test_param")));
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void TestEmptyParameterList()
        {
            var strategy = new FullShardingStrategy();

            var plan = strategy.CalculateShardingPlan(new List<ParameterInfo>(), 4);

            Assert.AreEqual(4, plan.TotalShards);
            Assert.AreEqual(0, plan.Assignments.Count);
            Assert.AreEqual(0, plan.AlwaysGathered.Count);
        }

        [TestMethod]
        public void TestWorldSizeOne()
        {
            var strategy = new FullShardingStrategy();

            var parameters = new List<ParameterInfo>
            {
                new ParameterInfo
                {
                    Name = "param1",
                    Shape = new[] { 1000L },
                    SizeBytes = 4000,
                    LayerName = "layer1",
                    AlwaysGather = false
                }
            };

            var plan = strategy.CalculateShardingPlan(parameters, 1);

            Assert.AreEqual(1, plan.TotalShards);
            Assert.IsTrue(plan.Assignments.ContainsKey("param1_rank0"));
        }

        [TestMethod]
        public void TestLayerWiseShardingWithMoreDevicesThanLayers()
        {
            var strategy = new LayerWiseShardingStrategy();

            var parameters = new List<ParameterInfo>
            {
                new ParameterInfo
                {
                    Name = "layer1.weight",
                    Shape = new[] { 1000L },
                    SizeBytes = 4000,
                    LayerName = "layer1",
                    AlwaysGather = false
                },
                new ParameterInfo
                {
                    Name = "layer2.weight",
                    Shape = new[] { 500L },
                    SizeBytes = 2000,
                    LayerName = "layer2",
                    AlwaysGather = false
                }
            };

            // 8 devices but only 2 layers
            var plan = strategy.CalculateShardingPlan(parameters, 8);

            Assert.AreEqual(8, plan.TotalShards);
            Assert.AreEqual(2, plan.Assignments.Count);
        }

        [TestMethod]
        public void TestStrategyNames()
        {
            var fullStrategy = new FullShardingStrategy();
            var layerWiseStrategy = new LayerWiseShardingStrategy();
            var hybridStrategy = new HybridShardingStrategy(new List<string>(), new List<string>());

            Assert.AreEqual("Full", fullStrategy.Name);
            Assert.AreEqual("LayerWise", layerWiseStrategy.Name);
            Assert.AreEqual("Hybrid", hybridStrategy.Name);
        }

        [TestMethod]
        public void TestShardingStrategyFactory()
        {
            var fullStrategy = ShardingStrategyFactory.Create(ShardingStrategy.Full);
            var layerWiseStrategy = ShardingStrategyFactory.Create(ShardingStrategy.LayerWise);
            var hybridStrategy = ShardingStrategyFactory.Create(ShardingStrategy.Hybrid);

            Assert.IsInstanceOfType(fullStrategy, typeof(FullShardingStrategy));
            Assert.IsInstanceOfType(layerWiseStrategy, typeof(LayerWiseShardingStrategy));
            Assert.IsInstanceOfType(hybridStrategy, typeof(HybridShardingStrategy));
        }

        [TestMethod]
        public void TestShardingStrategyFactoryWithHybridConfig()
        {
            var config = new HybridConfig
            {
                FullShardedLayers = new List<string> { "transformer" },
                LayerWiseShardedLayers = new List<string> { "classifier" }
            };

            var hybridStrategy = ShardingStrategyFactory.Create(ShardingStrategy.Hybrid, config);

            Assert.IsInstanceOfType(hybridStrategy, typeof(HybridShardingStrategy));

            var parameters = new List<ParameterInfo>
            {
                new ParameterInfo
                {
                    Name = "transformer.weight",
                    Shape = new[] { 1000L },
                    SizeBytes = 4000,
                    LayerName = "transformer",
                    AlwaysGather = false
                }
            };

            var plan = hybridStrategy.CalculateShardingPlan(parameters, 4);
            Assert.IsTrue(plan.Assignments.Any(kvp => kvp.Key.StartsWith("transformer.weight")));
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void TestNullParametersThrowsException()
        {
            var strategy = new FullShardingStrategy();
            strategy.CalculateShardingPlan(null, 4);
        }

        [TestMethod]
        public void TestUnevenShardSizes()
        {
            var strategy = new FullShardingStrategy();

            // Parameter with size that doesn't divide evenly
            var parameters = new List<ParameterInfo>
            {
                new ParameterInfo
                {
                    Name = "test_param",
                    Shape = new[] { 1003L }, // Not evenly divisible by 4
                    SizeBytes = 4012,
                    LayerName = "test",
                    AlwaysGather = false
                }
            };

            var plan = strategy.CalculateShardingPlan(parameters, 4);

            // Should create 4 shards with different sizes
            Assert.AreEqual(4, plan.Assignments.Count);

            // Last shard should be smaller if total doesn't divide evenly
            var lastShard = plan.Assignments[$"test_param_rank3"];
            Assert.IsTrue(lastShard.ShardSize > 0);
        }

        [TestMethod]
        public void TestMultipleParametersWithDifferentLayers()
        {
            var strategy = new LayerWiseShardingStrategy();

            var parameters = new List<ParameterInfo>
            {
                new ParameterInfo
                {
                    Name = "transformer.qkv.weight",
                    Shape = new[] { 768L, 768L },
                    SizeBytes = 2359296,
                    LayerName = "transformer.qkv",
                    AlwaysGather = false
                },
                new ParameterInfo
                {
                    Name = "transformer.qkv.bias",
                    Shape = new[] { 768L },
                    SizeBytes = 3072,
                    LayerName = "transformer.qkv",
                    AlwaysGather = false
                },
                new ParameterInfo
                {
                    Name = "transformer.attention.weight",
                    Shape = new[] { 768L, 768L },
                    SizeBytes = 2359296,
                    LayerName = "transformer.attention",
                    AlwaysGather = false
                },
                new ParameterInfo
                {
                    Name = "classifier.weight",
                    Shape = new[] { 768L, 10L },
                    SizeBytes = 30720,
                    LayerName = "classifier",
                    AlwaysGather = false
                }
            };

            var plan = strategy.CalculateShardingPlan(parameters, 4);

            // Should have assignments for all 4 parameters
            Assert.AreEqual(4, plan.Assignments.Count);
            Assert.IsTrue(plan.Assignments.ContainsKey("transformer.qkv.weight"));
            Assert.IsTrue(plan.Assignments.ContainsKey("transformer.qkv.bias"));
            Assert.IsTrue(plan.Assignments.ContainsKey("transformer.attention.weight"));
            Assert.IsTrue(plan.Assignments.ContainsKey("classifier.weight"));

            // Each assignment should have valid properties
            foreach (var kvp in plan.Assignments)
            {
                var assignment = kvp.Value;
                Assert.IsTrue(assignment.OwnerRank >= 0 && assignment.OwnerRank < 4);
                Assert.IsTrue(assignment.ShardIndex >= 0 && assignment.ShardIndex < 4);
                Assert.AreEqual(0, assignment.StartOffset); // Layer-wise doesn't offset
                Assert.IsTrue(assignment.ShardSize > 0);
            }
        }
    }
}
