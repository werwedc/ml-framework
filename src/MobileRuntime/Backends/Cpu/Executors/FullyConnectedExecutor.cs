using MLFramework.MobileRuntime.Backends.Cpu.Interfaces;
using MLFramework.MobileRuntime.Backends.Cpu.Utils;
using MLFramework.MobileRuntime;

namespace MLFramework.MobileRuntime.Backends.Cpu.Executors
{
    using System;
    using System.Collections.Generic;
    using System.Threading.Tasks;

    /// <summary>
    /// Executor for Fully Connected (Dense) layers.
    /// </summary>
    public sealed class FullyConnectedExecutor : IOperatorExecutor
    {
        private readonly CpuBackend _backend;

        public OperatorType OperatorType => OperatorType.FullyConnected;

        public FullyConnectedExecutor(CpuBackend backend)
        {
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        }

        public ITensor Execute(ITensor[] inputs, Dictionary<string, object> parameters)
        {
            if (inputs == null || inputs.Length < 2)
                throw new ArgumentException("FullyConnected requires at least input and weight tensors.");

            var input = inputs[0];
            var weight = inputs[1];
            var bias = inputs.Length > 2 ? inputs[2] : null;

            if (input.DataType != DataType.Float32 || weight.DataType != DataType.Float32)
                throw new ArgumentException("Only Float32 data type is currently supported.");

            var inputData = input.ToArray<float>();
            var weightData = weight.ToArray<float>();
            float[] biasData = bias?.ToArray<float>();

            var inputShape = input.Shape;
            var weightShape = weight.Shape;

            // Input should be 2D (batch, in_features) or 4D (batch, channels, height, width)
            // Weight should be 2D (out_features, in_features)
            // Bias should be 1D (out_features)

            int batchSize = inputShape[0];
            int inFeatures = inputShape.Length == 2 ? inputShape[1] : inputShape[1] * inputShape[2] * inputShape[3];
            int outFeatures = weightShape[0];

            if (inFeatures != weightShape[1])
                throw new ArgumentException($"Weight shape mismatch: expected {weightShape[1]}, got {inFeatures}");

            var outputData = new float[batchSize * outFeatures];

            // Check if we should use multi-threading
            bool useMultiThreading = _backend.IsMultiThreadingEnabled() &&
                                    batchSize * outFeatures * inFeatures > 16384; // 16K operations threshold

            if (useMultiThreading)
            {
                Parallel.For(0, batchSize, b =>
                {
                    for (int o = 0; o < outFeatures; o++)
                    {
                        int outputIdx = b * outFeatures + o;
                        outputData[outputIdx] = 0;

                        for (int i = 0; i < inFeatures; i++)
                        {
                            int inputIdx = inputShape.Length == 2 ? b * inFeatures + i :
                                         b * inFeatures + i; // Flatten 4D input
                            int weightIdx = o * inFeatures + i;
                            outputData[outputIdx] += inputData[inputIdx] * weightData[weightIdx];
                        }

                        // Add bias if present
                        if (biasData != null)
                        {
                            outputData[outputIdx] += biasData[o];
                        }
                    }
                });
            }
            else
            {
                unsafe
                {
                    fixed (float* inputPtr = inputData)
                    fixed (float* weightPtr = weightData)
                    fixed (float* biasPtr = biasData)
                    fixed (float* outputPtr = outputData)
                    {
                        for (int b = 0; b < batchSize; b++)
                        {
                            for (int o = 0; o < outFeatures; o++)
                            {
                                int outputIdx = b * outFeatures + o;
                                outputPtr[outputIdx] = 0;

                                float sum = 0;
                                for (int i = 0; i < inFeatures; i++)
                                {
                                    int inputIdx = b * inFeatures + i;
                                    int weightIdx = o * inFeatures + i;
                                    sum += inputPtr[inputIdx] * weightPtr[weightIdx];
                                }

                                // Add bias if present
                                if (biasPtr != null)
                                {
                                    sum += biasPtr[o];
                                }

                                outputPtr[outputIdx] = sum;
                            }
                        }
                    }
                }
            }

            // Create output tensor with shape (batch, out_features)
            throw new NotImplementedException("Tensor factory integration needed");
        }

        public bool CanFuseWith(IOperatorExecutor other)
        {
            // Can fuse with Relu
            return other is ReluExecutor;
        }

        public ITensor ExecuteFused(IOperatorExecutor[] executors, ITensor[][] inputs, Dictionary<string, object>[] parameters)
        {
            // Execute fully connected first, then fuse with following operators
            var fcOutput = Execute(inputs[0], parameters[0]);

            // If fused with ReLU, apply it in-place
            if (executors.Length > 1 && executors[1] is ReluExecutor)
            {
                var outputData = fcOutput.ToArray<float>();
                unsafe
                {
                    fixed (float* dataPtr = outputData)
                    {
                        CpuVectorization.Relu(dataPtr, outputData.Length, _backend.IsVectorizationEnabled());
                    }
                }
                // Update tensor data (this would be done differently in real implementation)
            }

            return fcOutput;
        }
    }
}
