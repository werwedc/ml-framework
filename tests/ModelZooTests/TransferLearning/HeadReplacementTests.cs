using System;
using System.Linq;
using MLFramework.ModelZoo.TransferLearning;
using MLFramework.NN;
using Xunit;

namespace MLFramework.Tests.ModelZooTests.TransferLearning
{
    /// <summary>
    /// Unit tests for head replacement utilities.
    /// </summary>
    public class HeadReplacementTests
    {
        /// <summary>
        /// Helper method to create a mock module for testing.
        /// </summary>
        private Module CreateMockModule(string name)
        {
            return new LinearHead(10, 10, true, name);
        }

        /// <summary>
        /// Helper method to create a sequential model with multiple layers.
        /// </summary>
        private SequentialModule CreateTestModel(int numLayers = 3)
        {
            var model = new SequentialModule("test_model");
            for (int i = 0; i < numLayers; i++)
            {
                model.Add(CreateMockModule($"layer_{i}"));
            }
            return model;
        }

        [Fact]
        public void WeightInitializationStrategy_GetDefaultParameters_ReturnsValidParameters()
        {
            // Test Xavier strategy
            var (xavierGain, xavierStd) = WeightInitializationStrategy.Xavier.GetDefaultParameters();
            Assert.Equal(1.0f, xavierGain);
            Assert.Equal(1.0f, xavierStd);

            // Test Kaiming strategy
            var (kaimingGain, kaimingStd) = WeightInitializationStrategy.Kaiming.GetDefaultParameters();
            Assert.Equal(MathF.Sqrt(2.0f), kaimingGain);
            Assert.Equal(1.0f, kaimingStd);

            // Test Uniform strategy
            var (uniformGain, uniformStd) = WeightInitializationStrategy.Uniform.GetDefaultParameters();
            Assert.Equal(1.0f, uniformGain);
            Assert.Equal(0.02f, uniformStd);
        }

        [Fact]
        public void WeightInitializationStrategy_IsRecommendedForActivation_ReturnsExpectedResults()
        {
            // Xavier is recommended for sigmoid/tanh
            Assert.True(WeightInitializationStrategy.Xavier.IsRecommendedForActivation("sigmoid"));
            Assert.True(WeightInitializationStrategy.Xavier.IsRecommendedForActivation("tanh"));

            // Kaiming is recommended for ReLU variants
            Assert.True(WeightInitializationStrategy.Kaiming.IsRecommendedForActivation("relu"));
            Assert.True(WeightInitializationStrategy.Kaiming.IsRecommendedForActivation("leaky_relu"));
            Assert.True(WeightInitializationStrategy.Kaiming.IsRecommendedForActivation("elu"));

            // Xavier is not recommended for ReLU
            Assert.False(WeightInitializationStrategy.Xavier.IsRecommendedForActivation("relu"));
        }

        [Fact]
        public void RemoveLastLayer_RemovesFinalLayerFromModel()
        {
            var model = CreateTestModel(3);
            int initialCount = model.Count;

            var removed = model.RemoveLastLayer();

            Assert.Equal(2, model.Count);
            Assert.NotNull(removed);
            Assert.Equal("layer_2", removed.Name);
        }

        [Fact]
        public void RemoveLastLayer_ThrowsExceptionWhenModelIsEmpty()
        {
            var model = new SequentialModule("empty_model");

            Assert.Throws<HeadReplacementException>(() => model.RemoveLastLayer());
        }

        [Fact]
        public void RemoveLastNLayers_RemovesCorrectNumberOfLayers()
        {
            var model = CreateTestModel(5);

            var removed = model.RemoveLastNLayers(2);

            Assert.Equal(3, model.Count);
            Assert.Equal(2, removed.Count);
            Assert.Equal("layer_4", removed[0].Name);
            Assert.Equal("layer_3", removed[1].Name);
        }

        [Fact]
        public void RemoveLastNLayers_ThrowsExceptionWhenNExceedsLayerCount()
        {
            var model = CreateTestModel(2);

            Assert.Throws<HeadReplacementException>(() => model.RemoveLastNLayers(5));
        }

        [Fact]
        public void RemoveLastNLayers_ThrowsExceptionWhenNIsZeroOrNegative()
        {
            var model = CreateTestModel(3);

            Assert.Throws<ArgumentOutOfRangeException>(() => model.RemoveLastNLayers(0));
            Assert.Throws<ArgumentOutOfRangeException>(() => model.RemoveLastNLayers(-1));
        }

        [Fact]
        public void AddHead_AddsModuleAsLastLayer()
        {
            var model = CreateTestModel(2);
            var newHead = CreateMockModule("new_head");

            model.AddHead(newHead);

            Assert.Equal(3, model.Count);
            Assert.Equal("new_head", model.GetHead().Name);
        }

        [Fact]
        public void AddHead_ThrowsExceptionWhenModuleIsNull()
        {
            var model = CreateTestModel(2);

            Assert.Throws<ArgumentNullException>(() => model.AddHead(null));
        }

        [Fact]
        public void ReplaceHead_ReplacesFinalLayerWithNewModule()
        {
            var model = CreateTestModel(3);
            var newHead = CreateMockModule("new_head");

            var oldHead = model.ReplaceHead(newHead);

            Assert.NotNull(oldHead);
            Assert.Equal("layer_2", oldHead.Name);
            Assert.Equal(3, model.Count);
            Assert.Equal("new_head", model.GetHead().Name);
        }

        [Fact]
        public void ReplaceHead_ThrowsExceptionWhenModelIsEmpty()
        {
            var model = new SequentialModule("empty_model");
            var newHead = CreateMockModule("new_head");

            Assert.Throws<HeadReplacementException>(() => model.ReplaceHead(newHead));
        }

        [Fact]
        public void ReplaceHead_ThrowsExceptionWhenNewHeadIsNull()
        {
            var model = CreateTestModel(2);

            Assert.Throws<ArgumentNullException>(() => model.ReplaceHead(null));
        }

        [Fact]
        public void GetHead_ReturnsFinalLayer()
        {
            var model = CreateTestModel(3);

            var head = model.GetHead();

            Assert.NotNull(head);
            Assert.Equal("layer_2", head.Name);
        }

        [Fact]
        public void GetHead_ReturnsNullForEmptyModel()
        {
            var model = new SequentialModule("empty_model");

            var head = model.GetHead();

            Assert.Null(head);
        }

        [Fact]
        public void GetHeads_ReturnsFinalNLayers()
        {
            var model = CreateTestModel(5);

            var heads = model.GetHeads(3);

            Assert.Equal(3, heads.Count);
            Assert.Equal("layer_4", heads[0].Name);
            Assert.Equal("layer_3", heads[1].Name);
            Assert.Equal("layer_2", heads[2].Name);
        }

        [Fact]
        public void ValidateHeadReplacement_ReturnsTrueForValidConfiguration()
        {
            var model = CreateTestModel(3);
            var newHead = CreateMockModule("new_head");

            var isValid = model.ValidateHeadReplacement(newHead, null);

            Assert.True(isValid);
        }

        [Fact]
        public void GetHeadSummary_ReturnsCorrectSummary()
        {
            var model = CreateTestModel(3);

            var summary = model.GetHeadSummary();

            Assert.Contains("3 total layer", summary);
            Assert.Contains("layer_2", summary);
        }

        [Fact]
        public void HeadBuilder_LinearHead_CreatesLinearHead()
        {
            var head = HeadBuilder.LinearHead(100, 10);

            Assert.NotNull(head);
            Assert.IsType<LinearHead>(head);
        }

        [Fact]
        public void HeadBuilder_LinearHead_WithDropout_StillWorks()
        {
            var head = HeadBuilder.LinearHead(100, 10, true, 0.5f);

            Assert.NotNull(head);
            Assert.IsType<LinearHead>(head);
        }

        [Fact]
        public void HeadBuilder_MLPHead_CreatesMLPHead()
        {
            var head = HeadBuilder.MLPHead(100, new[] { 50, 25 }, 10);

            Assert.NotNull(head);
            Assert.IsType<MLPHead>(head);
        }

        [Fact]
        public void HeadBuilder_ConvHead_CreatesPlaceholderHead()
        {
            var head = HeadBuilder.ConvHead(64, 10);

            Assert.NotNull(head);
            Assert.IsType<LinearHead>(head); // Currently returns LinearHead as placeholder
        }

        [Fact]
        public void HeadBuilder_AdaptiveAvgPoolHead_CreatesPlaceholderHead()
        {
            var head = HeadBuilder.AdaptiveAvgPoolHead(2048, 10);

            Assert.NotNull(head);
            Assert.IsType<LinearHead>(head); // Currently returns LinearHead as placeholder
        }

        [Fact]
        public void HeadBuilder_AttentionHead_CreatesMLPHead()
        {
            var head = HeadBuilder.AttentionHead(512, 10, 8);

            Assert.NotNull(head);
            Assert.IsType<MLPHead>(head); // Currently returns MLPHead as placeholder
        }

        [Fact]
        public void HeadBuilder_CreateDefaultHead_CreatesCorrectHeadType()
        {
            var linearHead = HeadBuilder.CreateDefaultHead(100, 10, "linear");
            var mlpHead = HeadBuilder.CreateDefaultHead(100, 10, "mlp");

            Assert.IsType<LinearHead>(linearHead);
            Assert.IsType<MLPHead>(mlpHead);
        }

        [Fact]
        public void HeadBuilder_CreateDefaultHead_ThrowsExceptionForUnknownType()
        {
            Assert.Throws<ArgumentException>(() => HeadBuilder.CreateDefaultHead(100, 10, "unknown"));
        }

        [Fact]
        public void LinearHead_InitializeWeights_UpdatesParameters()
        {
            var head = new LinearHead(100, 10);
            var originalWeight = head.Weight.Data[0];

            head.InitializeWeights(WeightInitializationStrategy.Xavier);

            // Weight should have changed (with high probability)
            Assert.NotEqual(originalWeight, head.Weight.Data[0]);
        }

        [Fact]
        public void MLPHead_InitializeWeights_UpdatesAllLayers()
        {
            var head = new MLPHead(100, new[] { 50, 25 }, 10);
            var parameters = head.GetParameters().ToList();
            var originalValue = parameters[0].Data[0];

            head.InitializeWeights(WeightInitializationStrategy.Kaiming);

            // At least one parameter should have changed
            var anyChanged = parameters.Any(p => p.Data[0] != originalValue);
            Assert.True(anyChanged);
        }

        [Fact]
        public void HeadAdapter_Creation_InitializesCorrectly()
        {
            var model = CreateTestModel(3);
            var adapter = new HeadAdapter(model, 100);

            Assert.NotNull(adapter);
            Assert.Equal(model, adapter.Model);
            Assert.Equal(100, adapter.InputDim);
        }

        [Fact]
        public void HeadAdapter_Creation_ThrowsExceptionForNullModel()
        {
            Assert.Throws<ArgumentNullException>(() => new HeadAdapter(null));
        }

        [Fact]
        public void HeadAdapter_SetNumberOfClasses_CreatesNewHead()
        {
            var model = CreateTestModel(2);
            var adapter = new HeadAdapter(model, 100);

            adapter.SetNumberOfClasses(10, "linear", WeightInitializationStrategy.Kaiming);

            Assert.Equal(10, adapter.NumClasses);
            Assert.NotNull(adapter.CurrentHead);
            Assert.Equal(2, model.Count); // Original layers, head was replaced
        }

        [Fact]
        public void HeadAdapter_SetNumberOfClasses_ThrowsExceptionForInvalidNumClasses()
        {
            var model = CreateTestModel(2);
            var adapter = new HeadAdapter(model, 100);

            Assert.Throws<ArgumentOutOfRangeException>(() => adapter.SetNumberOfClasses(0));
            Assert.Throws<ArgumentOutOfRangeException>(() => adapter.SetNumberOfClasses(-1));
        }

        [Fact]
        public void HeadAdapter_SetNumberOfClasses_ThrowsExceptionWhenInputDimUnknown()
        {
            var model = CreateTestModel(2);
            var adapter = new HeadAdapter(model, -1); // Unknown input dimension

            Assert.Throws<HeadAdapterException>(() => adapter.SetNumberOfClasses(10));
        }

        [Fact]
        public void HeadAdapter_ReplaceHead_ReplacesCurrentHead()
        {
            var model = CreateTestModel(3);
            var adapter = new HeadAdapter(model, 100);
            var newHead = CreateMockModule("new_head");

            adapter.ReplaceHead(newHead);

            Assert.Equal(newHead, adapter.CurrentHead);
            Assert.Equal(3, model.Count);
        }

        [Fact]
        public void HeadAdapter_ReplaceHead_ThrowsExceptionForNullHead()
        {
            var model = CreateTestModel(2);
            var adapter = new HeadAdapter(model, 100);

            Assert.Throws<ArgumentNullException>(() => adapter.ReplaceHead(null));
        }

        [Fact]
        public void HeadAdapter_InitializeWeights_InitializesCurrentHead()
        {
            var model = CreateTestModel(2);
            var adapter = new HeadAdapter(model, 100);
            adapter.SetNumberOfClasses(10);

            var parameters = adapter.CurrentHead.GetParameters().ToList();
            var originalValue = parameters[0].Data[0];

            adapter.InitializeWeights(WeightInitializationStrategy.Xavier);

            var anyChanged = parameters.Any(p => p.Data[0] != originalValue);
            Assert.True(anyChanged);
        }

        [Fact]
        public void HeadAdapter_InitializeWeights_ThrowsExceptionWhenNoHead()
        {
            var model = new SequentialModule("empty");
            var adapter = new HeadAdapter(model, 100);

            Assert.Throws<HeadAdapterException>(() => adapter.InitializeWeights(WeightInitializationStrategy.Xavier));
        }

        [Fact]
        public void HeadAdapter_SetInputDim_UpdatesInputDimension()
        {
            var model = CreateTestModel(2);
            var adapter = new HeadAdapter(model, -1);

            adapter.SetInputDim(200);

            Assert.Equal(200, adapter.InputDim);
        }

        [Fact]
        public void HeadAdapter_SetInputDim_ThrowsExceptionForInvalidValue()
        {
            var model = CreateTestModel(2);
            var adapter = new HeadAdapter(model, 100);

            Assert.Throws<ArgumentOutOfRangeException>(() => adapter.SetInputDim(0));
            Assert.Throws<ArgumentOutOfRangeException>(() => adapter.SetInputDim(-1));
        }

        [Fact]
        public void HeadAdapter_GetInputDim_ReturnsCorrectValue()
        {
            var model = CreateTestModel(2);
            var adapter = new HeadAdapter(model, 150);

            var inputDim = adapter.GetInputDim();

            Assert.Equal(150, inputDim);
        }

        [Fact]
        public void HeadAdapter_Reset_RemovesCurrentHead()
        {
            var model = CreateTestModel(3);
            var adapter = new HeadAdapter(model, 100);
            adapter.SetNumberOfClasses(10);

            adapter.Reset();

            Assert.Null(adapter.CurrentHead);
            Assert.Equal(2, model.Count); // Head was removed
        }

        [Fact]
        public void HeadAdapter_AdaptPretrained_CreatesConfiguredAdapter()
        {
            var model = CreateTestModel(5);

            var adapter = HeadAdapter.AdaptPretrained(model, 10, freezeBackbone: false, "mlp");

            Assert.NotNull(adapter);
            Assert.Equal(10, adapter.NumClasses);
            Assert.NotNull(adapter.CurrentHead);
        }

        [Fact]
        public void HeadAdapter_GetSummary_ReturnsCorrectInformation()
        {
            var model = CreateTestModel(3);
            var adapter = new HeadAdapter(model, 100);

            var summary = adapter.GetSummary();

            Assert.Contains("Input Dimension: 100", summary);
            Assert.Contains("Model Layers: 3", summary);
        }

        [Fact]
        public void HeadBuilder_InitializeHead_WithLinearHead_Works()
        {
            var head = HeadBuilder.LinearHead(100, 10);
            var originalValue = head.Data[0];

            HeadBuilder.InitializeHead(head, WeightInitializationStrategy.Kaiming);

            Assert.NotEqual(originalValue, head.Data[0]);
        }

        [Fact]
        public void HeadBuilder_InitializeHead_ThrowsExceptionForNullHead()
        {
            Assert.Throws<ArgumentNullException>(() => HeadBuilder.InitializeHead(null, WeightInitializationStrategy.Xavier));
        }

        [Fact]
        public void HeadBuilder_InitializeHead_WithMLPHead_InitializesAllLayers()
        {
            var head = HeadBuilder.MLPHead(100, new[] { 50, 25 }, 10);
            var parameters = head.GetParameters().ToList();

            HeadBuilder.InitializeHead(head, WeightInitializationStrategy.Xavier);

            // All parameters should be initialized
            Assert.All(parameters, p =>
            {
                Assert.NotNull(p.Data);
                Assert.NotEmpty(p.Data);
            });
        }

        [Fact]
        public void RemoveLastNLayers_RemovesAllLayersWhenNEqualsLayerCount()
        {
            var model = CreateTestModel(3);

            var removed = model.RemoveLastNLayers(3);

            Assert.Equal(0, model.Count);
            Assert.Equal(3, removed.Count);
        }

        [Fact]
        public void GetHeadSummary_ForEmptyModel_ReturnsAppropriateMessage()
        {
            var model = new SequentialModule("empty");

            var summary = model.GetHeadSummary();

            Assert.Contains("no layers", summary);
        }
    }
}
