using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Distributed.FSDP;
using System.Collections.Generic;

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
            Assert.AreEqual(2, plan.Assignments.Count);
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

            var plan = strategy.CalculateShardingPlan(parameters, 2);

            Assert.IsTrue(plan.Assignments.ContainsKey("test_param"));
            var assignment = plan.Assignments["test_param"];
            Assert.AreEqual("test_param", assignment.ParameterName);
            Assert.AreEqual(4000, assignment.Size);
            Assert.AreEqual(0, assignment.Offset);
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
            Assert.IsTrue(plan.Assignments.ContainsKey("test_param"));
        }

        [TestMethod]
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
            Assert.IsTrue(plan.Assignments.ContainsKey("param1"));
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
    }
}
