using MLFramework.MobileRuntime.Backends.Cpu.Interfaces;
using MLFramework.MobileRuntime.Backends.Cpu.Models;
using MLFramework.MobileRuntime;

namespace MLFramework.MobileRuntime.Backends.Cpu.Executors
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// Executor for 2D Max Pooling operations.
    /// </summary>
    public sealed class MaxPool2DExecutor : IOperatorExecutor
    {
        private readonly CpuBackend _backend;

        public OperatorType OperatorType => OperatorType.MaxPool2D;

        public MaxPool2DExecutor(CpuBackend backend)
        {
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        }

        public ITensor Execute(ITensor[] inputs, Dictionary<string, object> parameters)
        {
            if (inputs == null || inputs.Length != 1)
                throw new ArgumentException("MaxPool2D requires exactly one input tensor.");

            var input = inputs[0];
            var data = input.ToArray<float>();
            var shape = input.Shape;

            if (shape.Length != 4)
                throw new ArgumentException("Input tensor must be 4D (batch, channels, height, width).");

            // Parse pooling parameters
            var poolParams = ParsePoolParams(parameters);

            int batchSize = shape[0];
            int channels = shape[1];
            int inputHeight = shape[2];
            int inputWidth = shape[3];
            int kernelHeight = poolParams.KernelSize[0];
            int kernelWidth = poolParams.KernelSize[1];
            int strideHeight = poolParams.Stride[0];
            int strideWidth = poolParams.Stride[1];
            int paddingHeight = poolParams.Padding[0];
            int paddingWidth = poolParams.Padding[1];

            // Calculate output dimensions
            int outputHeight = (inputHeight + 2 * paddingHeight - kernelHeight) / strideHeight + 1;
            int outputWidth = (inputWidth + 2 * paddingWidth - kernelWidth) / strideWidth + 1;

            var outputData = new float[batchSize * channels * outputHeight * outputWidth];

            ExecuteMaxPool2D(data, outputData, batchSize, channels, inputHeight, inputWidth,
                          outputHeight, outputWidth, kernelHeight, kernelWidth,
                          strideHeight, strideWidth, paddingHeight, paddingWidth);

            // Create output tensor
            throw new NotImplementedException("Tensor factory integration needed");
        }

        public bool CanFuseWith(IOperatorExecutor other)
        {
            return false;
        }

        public ITensor ExecuteFused(IOperatorExecutor[] executors, ITensor[][] inputs, Dictionary<string, object>[] parameters)
        {
            throw new NotSupportedException("MaxPool2DExecutor does not support operator fusion.");
        }

        private void ExecuteMaxPool2D(float[] inputData, float[] outputData,
                                     int batchSize, int channels,
                                     int inputHeight, int inputWidth,
                                     int outputHeight, int outputWidth,
                                     int kernelHeight, int kernelWidth,
                                     int strideHeight, int strideWidth,
                                     int paddingHeight, int paddingWidth)
        {
            for (int b = 0; b < batchSize; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int oh = 0; oh < outputHeight; oh++)
                    {
                        for (int ow = 0; ow < outputWidth; ow++)
                        {
                            int outputIdx = b * channels * outputHeight * outputWidth +
                                          c * outputHeight * outputWidth +
                                          oh * outputWidth + ow;

                            float maxValue = float.MinValue;

                            // Calculate input window bounds
                            int ihStart = oh * strideHeight - paddingHeight;
                            int ihEnd = Math.Min(ihStart + kernelHeight, inputHeight);
                            ihStart = Math.Max(ihStart, 0);

                            int iwStart = ow * strideWidth - paddingWidth;
                            int iwEnd = Math.Min(iwStart + kernelWidth, inputWidth);
                            iwStart = Math.Max(iwStart, 0);

                            // Find max in window
                            for (int ih = ihStart; ih < ihEnd; ih++)
                            {
                                for (int iw = iwStart; iw < iwEnd; iw++)
                                {
                                    int inputIdx = b * channels * inputHeight * inputWidth +
                                                  c * inputHeight * inputWidth +
                                                  ih * inputWidth + iw;
                                    maxValue = Math.Max(maxValue, inputData[inputIdx]);
                                }
                            }

                            outputData[outputIdx] = maxValue;
                        }
                    }
                }
            }
        }

        private Pool2DParams ParsePoolParams(Dictionary<string, object> parameters)
        {
            var poolParams = new Pool2DParams();

            if (parameters.TryGetValue("kernel_size", out var kernelSizeObj))
            {
                poolParams.KernelSize = (int[])kernelSizeObj;
            }
            else
            {
                poolParams.KernelSize = new int[] { 2, 2 };
            }

            if (parameters.TryGetValue("stride", out var strideObj))
            {
                poolParams.Stride = (int[])strideObj;
            }
            else
            {
                poolParams.Stride = poolParams.KernelSize;
            }

            if (parameters.TryGetValue("padding", out var paddingObj))
            {
                poolParams.Padding = (int[])paddingObj;
            }
            else
            {
                poolParams.Padding = new int[] { 0, 0 };
            }

            if (parameters.TryGetValue("count_include_pad", out var includePadObj))
            {
                poolParams.CountIncludePad = (bool)includePadObj;
            }

            return poolParams;
        }
    }
}
