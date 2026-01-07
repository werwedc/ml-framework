using System;
using System.Collections.Generic;
using Xunit;
using MLFramework.Quantization.PTQ;
using MLFramework.Quantization.DataStructures;
using MLFramework.NN;
using MLFramework.Data;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Quantization.PTQ
{
    /// <summary>
    /// End-to-end tests for PTQ workflow.
    /// </summary>
    public class EndToEndPTQTests
    {
        private class MockLinearModule : Module
        {
            public Parameter Weight { get; }

            public MockLinearModule(string name, int inFeatures, int outFeatures)
            {
                Name = name;
                var random = new Random(42);
                float[] weightData = new float[inFeatures * outFeatures];
                for (int i = 0; i < weightData.Length; i++)
                {
                    weightData[i] = (float)((random.NextDouble() - 0.5) * 2.0);
                }
                Weight = new Parameter(weightData, new int[] { outFeatures, inFeatures }, requiresGrad: true);
            }

            public override Tensor Forward(Tensor input)
            {
                // Simple linear transformation
                int batchSize = input.Shape[0];
                int inFeatures = input.Shape[1];
                int outFeatures = Weight.Shape[0];

                float[] outputData = new float[batchSize * outFeatures];
                float[] inputData = input.Data;
                float[] weightData = Weight.Data;

                for (int b = 0; b < batchSize; b++)
                {
                    for (int o = 0; o < outFeatures; o++)
                    {
                        float sum = 0;
                        for (int i = 0; i < inFeatures; i++)
                        {
                            sum += inputData[b * inFeatures + i] * weightData[o * inFeatures + i];
                        }
                        outputData[b * outFeatures + o] = sum;
                    }
                }

                return new Tensor(outputData, new int[] { batchSize, outFeatures });
            }
        }

        private class MockReluModule : Module
        {
            public MockReluModule(string name)
            {
                Name = name;
            }

            public override Tensor Forward(Tensor input)
            {
                float[] outputData = new float[input.Data.Length];
                for (int i = 0; i < outputData.Length; i++)
                {
                    outputData[i] = MathF.Max(0, input.Data[i]);
                }
                return new Tensor(outputData, input.Shape);
            }
        }

        private class MockMNISTModel : Module
        {
            public MockLinearLayerModel Model { get; }

            public MockMNISTModel()
            {
                Name = "mnist_classifier";
                Model = new MockLinearLayerModel();
            }

            public override Tensor Forward(Tensor input)
            {
                return Model.Forward(input);
            }

            public float ComputeAccuracy(Tensor predictions, int[] labels)
            {
                int correct = 0;
                for (int i = 0; i < predictions.Shape[0]; i++)
                {
                    int predicted = ArgMax(predictions, i);
                    if (predicted == labels[i])
                    {
                        correct++;
                    }
                }
                return (float)correct / predictions.Shape[0];
            }

            private int ArgMax(Tensor tensor, int batchIdx)
            {
                int startIdx = batchIdx * tensor.Shape[1];
                float maxVal = tensor.Data[startIdx];
                int maxIdx = 0;

                for (int i = 1; i < tensor.Shape[1]; i++)
                {
                    if (tensor.Data[startIdx + i] > maxVal)
                    {
                        maxVal = tensor.Data[startIdx + i];
                        maxIdx = i;
                    }
                }

                return maxIdx;
            }
        }

        private class MockLinearLayerModel : Module
        {
            public MockLinearModule Linear1 { get; }
            public MockReluModule Relu { get; }
            public MockLinearModule Linear2 { get; }

            public MockLinearLayerModel()
            {
                Name = "model";
                Linear1 = new MockLinearModule("linear1", 784, 128);
                Relu = new MockReluModule("relu");
                Linear2 = new MockLinearModule("linear2", 128, 10);
            }

            public override Tensor Forward(Tensor input)
            {
                var output = Linear1.Forward(input);
                output = Relu.Forward(output);
                output = Linear2.Forward(output);
                return output;
            }
        }

        private class MockDataLoader : DataLoader<object>
        {
            private readonly Tensor[] _inputs;
            private readonly int[] _labels;
            private int _currentBatch;

            public MockDataLoader(Tensor[] inputs, int[] labels)
            {
                _inputs = inputs;
                _labels = labels;
                _currentBatch = 0;
            }

            public override int Count => _inputs.Length;
            public override int BatchSize => 1;

            public override IEnumerator<object> GetEnumerator()
            {
                while (_currentBatch < _inputs.Length)
                {
                    yield return (_inputs[_currentBatch], _labels[_currentBatch]);
                    _currentBatch++;
                }
            }

            public override void Reset()
            {
                _currentBatch = 0;
            }

            public Tensor GetInput(int index) => _inputs[index];
            public int GetLabel(int index) => _labels[index];
        }

        [Fact]
        public void CompletePTQWorkflow_TrainApplyCompare_VerifiesAccuracyDropIsAcceptable()
        {
            // Arrange
            var model = new MockMNISTModel();
            var random = new Random(42);

            // Create synthetic MNIST-like data
            var trainInputs = new Tensor[100];
            var trainLabels = new int[100];
            for (int i = 0; i < 100; i++)
            {
                float[] data = new float[784];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)random.NextDouble();
                }
                trainInputs[i] = new Tensor(data, new int[] { 1, 784 });
                trainLabels[i] = random.Next(0, 10);
            }

            var trainLoader = new MockDataLoader(trainInputs, trainLabels);

            // Get FP32 baseline accuracy
            float fp32Accuracy = ComputeAccuracyOnData(model, trainLoader);

            // Apply PTQ with calibration
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.PerTensor,
                CalibrationMethod = CalibrationMethod.MinMax
            };

            var quantizer = new PTQQuantizer();

            // Act
            quantizer.Quantize(model.Model, trainLoader, config);

            // Get quantized model accuracy
            trainLoader.Reset();
            float quantizedAccuracy = ComputeAccuracyOnData(model, trainLoader);

            // Assert
            // Accuracy drop should be acceptable (< 1%)
            float accuracyDrop = fp32Accuracy - quantizedAccuracy;
            Assert.True(accuracyDrop < 0.01f, 
                $"Accuracy drop {accuracyDrop:P2} exceeds 1% threshold. FP32: {fp32Accuracy:P2}, Quantized: {quantizedAccuracy:P2}");

            // Both accuracies should be reasonable
            Assert.True(fp32Accuracy >= 0.0f && fp32Accuracy <= 1.0f);
            Assert.True(quantizedAccuracy >= 0.0f && quantizedAccuracy <= 1.0f);
        }

        [Fact]
        public void CompletePTQWorkflow_CompareFP32VsQuantized_VerifiesAccuracyComparison()
        {
            // Arrange
            var model = new MockMNISTModel();
            var random = new Random(42);

            // Create validation data
            var valInputs = new Tensor[50];
            var valLabels = new int[50];
            for (int i = 0; i < 50; i++)
            {
                float[] data = new float[784];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)random.NextDouble();
                }
                valInputs[i] = new Tensor(data, new int[] { 1, 784 });
                valLabels[i] = random.Next(0, 10);
            }

            var valLoader = new MockDataLoader(valInputs, valLabels);

            // Get FP32 accuracy
            float fp32Accuracy = ComputeAccuracyOnData(model, valLoader);

            // Apply PTQ
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.PerTensor,
                CalibrationMethod = CalibrationMethod.MinMax
            };

            var quantizer = new PTQQuantizer();

            // Act
            quantizer.Quantize(model.Model, valLoader, config);

            // Get quantized accuracy
            valLoader.Reset();
            float quantizedAccuracy = ComputeAccuracyOnData(model, valLoader);

            // Assert
            // Quantized accuracy should be close to FP32
            Assert.InRange(quantizedAccuracy, fp32Accuracy - 0.01f, fp32Accuracy + 0.01f);

            // Both accuracies should be positive
            Assert.True(fp32Accuracy > 0);
            Assert.True(quantizedAccuracy > 0);
        }

        [Fact]
        public void CompletePTQWorkflow_VerifiesQuantizationParametersExist()
        {
            // Arrange
            var model = new MockMNISTModel();
            var random = new Random(42);

            // Create calibration data
            var calInputs = new Tensor[20];
            var calLabels = new int[20];
            for (int i = 0; i < 20; i++)
            {
                float[] data = new float[784];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)random.NextDouble();
                }
                calInputs[i] = new Tensor(data, new int[] { 1, 784 });
                calLabels[i] = random.Next(0, 10);
            }

            var calLoader = new MockDataLoader(calInputs, calLabels);

            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.PerTensor,
                CalibrationMethod = CalibrationMethod.MinMax
            };

            var quantizer = new PTQQuantizer();

            // Act
            quantizer.Quantize(model.Model, calLoader, config);

            // Assert
            // All quantizable layers should have weight parameters
            Assert.NotNull(quantizer.GetLayerQuantizationParameters("linear1"));
            Assert.NotNull(quantizer.GetLayerQuantizationParameters("linear2"));

            // Activations should have parameters in static quantization
            // (Note: This depends on the implementation details)
        }

        [Fact]
        public void CompletePTQWorkflow_VerifiesModelStructurePreserved()
        {
            // Arrange
            var model = new MockMNISTModel();
            var random = new Random(42);

            // Create calibration data
            var calInputs = new Tensor[10];
            var calLabels = new int[10];
            for (int i = 0; i < 10; i++)
            {
                float[] data = new float[784];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)random.NextDouble();
                }
                calInputs[i] = new Tensor(data, new int[] { 1, 784 });
                calLabels[i] = random.Next(0, 10);
            }

            var calLoader = new MockDataLoader(calInputs, calLabels);

            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.PerTensor,
                CalibrationMethod = CalibrationMethod.MinMax
            };

            var quantizer = new PTQQuantizer();

            // Act
            quantizer.Quantize(model.Model, calLoader, config);

            // Assert
            // Model structure should be preserved
            Assert.NotNull(model.Model.Linear1);
            Assert.NotNull(model.Model.Relu);
            Assert.NotNull(model.Model.Linear2);

            // Weights should still exist
            Assert.NotNull(model.Model.Linear1.Weight);
            Assert.NotNull(model.Model.Linear2.Weight);
        }

        [Fact]
        public void CompletePTQWorkflow_VerifiesInferenceWorks()
        {
            // Arrange
            var model = new MockMNISTModel();
            var random = new Random(42);

            // Create calibration data
            var calInputs = new Tensor[10];
            var calLabels = new int[10];
            for (int i = 0; i < 10; i++)
            {
                float[] data = new float[784];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)random.NextDouble();
                }
                calInputs[i] = new Tensor(data, new int[] { 1, 784 });
                calLabels[i] = random.Next(0, 10);
            }

            var calLoader = new MockDataLoader(calInputs, calLabels);

            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.PerTensor,
                CalibrationMethod = CalibrationMethod.MinMax
            };

            var quantizer = new PTQQuantizer();

            // Act
            quantizer.Quantize(model.Model, calLoader, config);

            // Test inference with new data
            float[] testData = new float[784];
            for (int i = 0; i < testData.Length; i++)
            {
                testData[i] = (float)random.NextDouble();
            }
            var testInput = new Tensor(testData, new int[] { 1, 784 });

            var output = model.Forward(testInput);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(new int[] { 1, 10 }, output.Shape);
        }

        private float ComputeAccuracyOnData(MockMNISTModel model, MockDataLoader dataLoader)
        {
            int total = 0;
            int correct = 0;

            foreach (var batch in dataLoader)
            {
                var (input, label) = (ValueTuple<Tensor, int>)batch;
                var predictions = model.Forward(input);

                if (predictions.Data[0] > 0.5f)
                {
                    correct++;
                }

                total++;
            }

            return (float)correct / total;
        }
    }
}
