using MLFramework.MobileRuntime.Backends.Cpu.Interfaces;
using MLFramework.MobileRuntime.Backends.Cpu.Models;
using MLFramework.MobileRuntime.Backends.Cpu.Utils;
using MLFramework.MobileRuntime;

namespace MLFramework.MobileRuntime.Backends.Cpu.Executors
{
    using System;
    using System.Collections.Generic;
    using System.Threading.Tasks;

    /// <summary>
    /// Executor for 2D Convolution operations.
    /// </summary>
    public sealed class Conv2DExecutor : IOperatorExecutor
    {
        private readonly CpuBackend _backend;

        public OperatorType OperatorType => OperatorType.Conv2D;

        public Conv2DExecutor(CpuBackend backend)
        {
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        }

        public ITensor Execute(ITensor[] inputs, Dictionary<string, object> parameters)
        {
            if (inputs == null || inputs.Length < 2)
                throw new ArgumentException("Conv2D requires at least input and weight tensors.");

            var input = inputs[0];
            var weight = inputs[1];
            var bias = inputs.Length > 2 ? inputs[2] : null;

            var convParams = ParseConvParams(parameters);

            var inputData = input.ToArray<float>();
            var weightData = weight.ToArray<float>();
            float[] biasData = bias?.ToArray<float>();

            var inputShape = input.Shape;
            var weightShape = weight.Shape;

            // Input shape: (batch, in_channels, height, width)
            // Weight shape: (out_channels, in_channels/groups, kernel_height, kernel_width)
            int batchSize = inputShape[0];
            int inChannels = inputShape[1];
            int inputHeight = inputShape[2];
            int inputWidth = inputShape[3];
            int outChannels = weightShape[0];
            int kernelHeight = convParams.KernelSize[0];
            int kernelWidth = convParams.KernelSize[1];
            int groups = convParams.Groups;

            if (inChannels % groups != 0 || outChannels % groups != 0)
                throw new ArgumentException("Channels must be divisible by groups.");

            int paddingHeight = convParams.Padding[0];
            int paddingWidth = convParams.Padding[1];
            int strideHeight = convParams.Stride[0];
            int strideWidth = convParams.Stride[1];

            // Calculate output dimensions
            int outputHeight = (inputHeight + 2 * paddingHeight - kernelHeight) / strideHeight + 1;
            int outputWidth = (inputWidth + 2 * paddingWidth - kernelWidth) / strideWidth + 1;

            var outputData = new float[batchSize * outChannels * outputHeight * outputWidth];

            bool useMultiThreading = _backend.IsMultiThreadingEnabled() &&
                                    batchSize * outChannels * outputHeight * outputWidth * kernelHeight * kernelWidth > 65536;

            if (useMultiThreading)
            {
                Parallel.For(0, batchSize, b =>
                {
                    for (int oc = 0; oc < outChannels; oc++)
                    {
                        int group = oc / (outChannels / groups);
                        int inChannelsPerGroup = inChannels / groups;
                        int outChannelsPerGroup = outChannels / groups;
                        int ocInGroup = oc % outChannelsPerGroup;

                        for (int oh = 0; oh < outputHeight; oh++)
                        {
                            for (int ow = 0; ow < outputWidth; ow++)
                            {
                                int outputIdx = b * outChannels * outputHeight * outputWidth +
                                              oc * outputHeight * outputWidth +
                                              oh * outputWidth + ow;

                                float sum = 0;

                                for (int ic = 0; ic < inChannelsPerGroup; ic++)
                                {
                                    int actualIc = group * inChannelsPerGroup + ic;

                                    for (int kh = 0; kh < kernelHeight; kh++)
                                    {
                                        int ih = oh * strideHeight + kh - paddingHeight;
                                        if (ih < 0 || ih >= inputHeight) continue;

                                        for (int kw = 0; kw < kernelWidth; kw++)
                                        {
                                            int iw = ow * strideWidth + kw - paddingWidth;
                                            if (iw < 0 || iw >= inputWidth) continue;

                                            int inputIdx = b * inChannels * inputHeight * inputWidth +
                                                          actualIc * inputHeight * inputWidth +
                                                          ih * inputWidth + iw;
                                            int weightIdx = oc * inChannelsPerGroup * kernelHeight * kernelWidth +
                                                          ic * kernelHeight * kernelWidth +
                                                          kh * kernelWidth + kw;

                                            sum += inputData[inputIdx] * weightData[weightIdx];
                                        }
                                    }
                                }

                                // Add bias if present
                                if (biasData != null)
                                {
                                    sum += biasData[oc];
                                }

                                outputData[outputIdx] = sum;
                            }
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
                            for (int oc = 0; oc < outChannels; oc++)
                            {
                                int group = oc / (outChannels / groups);
                                int inChannelsPerGroup = inChannels / groups;
                                int outChannelsPerGroup = outChannels / groups;
                                int ocInGroup = oc % outChannelsPerGroup;

                                for (int oh = 0; oh < outputHeight; oh++)
                                {
                                    for (int ow = 0; ow < outputWidth; ow++)
                                    {
                                        int outputIdx = b * outChannels * outputHeight * outputWidth +
                                                      oc * outputHeight * outputWidth +
                                                      oh * outputWidth + ow;

                                        float sum = 0;

                                        for (int ic = 0; ic < inChannelsPerGroup; ic++)
                                        {
                                            int actualIc = group * inChannelsPerGroup + ic;

                                            for (int kh = 0; kh < kernelHeight; kh++)
                                            {
                                                int ih = oh * strideHeight + kh - paddingHeight;
                                                if (ih < 0 || ih >= inputHeight) continue;

                                                for (int kw = 0; kw < kernelWidth; kw++)
                                                {
                                                    int iw = ow * strideWidth + kw - paddingWidth;
                                                    if (iw < 0 || iw >= inputWidth) continue;

                                                    int inputIdx = b * inChannels * inputHeight * inputWidth +
                                                                  actualIc * inputHeight * inputWidth +
                                                                  ih * inputWidth + iw;
                                                    int weightIdx = oc * inChannelsPerGroup * kernelHeight * kernelWidth +
                                                                  ic * kernelHeight * kernelWidth +
                                                                  kh * kernelWidth + kw;

                                                    sum += inputPtr[inputIdx] * weightPtr[weightIdx];
                                                }
                                            }
                                        }

                                        // Add bias if present
                                        if (biasPtr != null)
                                        {
                                            sum += biasPtr[oc];
                                        }

                                        outputPtr[outputIdx] = sum;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Create output tensor with shape (batch, out_channels, output_height, output_width)
            throw new NotImplementedException("Tensor factory integration needed");
        }

        public bool CanFuseWith(IOperatorExecutor other)
        {
            // Can fuse with BatchNorm, Relu, Sigmoid
            return other is ReluExecutor; // or BatchNormExecutor, SigmoidExecutor
        }

        public ITensor ExecuteFused(IOperatorExecutor[] executors, ITensor[][] inputs, Dictionary<string, object>[] parameters)
        {
            // Execute convolution first
            var convOutput = Execute(inputs[0], parameters[0]);

            // If fused with ReLU, apply it in-place
            if (executors.Length > 1 && executors[1] is ReluExecutor)
            {
                var outputData = convOutput.ToArray<float>();
                unsafe
                {
                    fixed (float* dataPtr = outputData)
                    {
                        CpuVectorization.Relu(dataPtr, outputData.Length, _backend.IsVectorizationEnabled());
                    }
                }
            }

            return convOutput;
        }

        private Conv2DParams ParseConvParams(Dictionary<string, object> parameters)
        {
            var convParams = new Conv2DParams();

            if (parameters.TryGetValue("kernel_size", out var kernelSizeObj))
            {
                convParams.KernelSize = (int[])kernelSizeObj;
            }
            else
            {
                convParams.KernelSize = new int[] { 3, 3 };
            }

            if (parameters.TryGetValue("stride", out var strideObj))
            {
                convParams.Stride = (int[])strideObj;
            }
            else
            {
                convParams.Stride = new int[] { 1, 1 };
            }

            if (parameters.TryGetValue("padding", out var paddingObj))
            {
                convParams.Padding = (int[])paddingObj;
            }
            else
            {
                convParams.Padding = new int[] { 0, 0 };
            }

            if (parameters.TryGetValue("dilation", out var dilationObj))
            {
                convParams.Dilation = (int[])dilationObj;
            }

            if (parameters.TryGetValue("groups", out var groupsObj))
            {
                convParams.Groups = (int)groupsObj;
            }

            return convParams;
        }
    }
}
