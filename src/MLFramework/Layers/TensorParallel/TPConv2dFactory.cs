using RitterFramework.Core.Tensor;

namespace MLFramework.Layers.TensorParallel
{
    /// <summary>
    /// Factory methods for creating tensor-parallel Conv2d layers
    /// </summary>
    public static class TPConv2dFactory
    {
        /// <summary>
        /// Create output-channel parallel Conv2d (most common)
        /// </summary>
        public static Conv2dOutputParallel CreateOutputParallel(
            int inChannels,
            int outChannels,
            int kernelSize,
            bool gatherOutput = false,
            int stride = 1,
            int padding = 0,
            bool bias = true)
        {
            return new Conv2dOutputParallel(
                inChannels, outChannels, kernelSize,
                gatherOutput: gatherOutput,
                processGroup: null,
                stride: stride,
                padding: padding,
                bias: bias);
        }

        /// <summary>
        /// Create output-channel parallel Conv2d with custom process group
        /// </summary>
        public static Conv2dOutputParallel CreateOutputParallel(
            int inChannels,
            int outChannels,
            int kernelSize,
            TensorParallelGroup processGroup,
            bool gatherOutput = false,
            int stride = 1,
            int padding = 0,
            bool bias = true)
        {
            return new Conv2dOutputParallel(
                inChannels, outChannels, kernelSize,
                gatherOutput: gatherOutput,
                processGroup: processGroup,
                stride: stride,
                padding: padding,
                bias: bias);
        }

        /// <summary>
        /// Create input-channel parallel Conv2d
        /// </summary>
        public static Conv2dInputParallel CreateInputParallel(
            int inChannels,
            int outChannels,
            int kernelSize,
            int stride = 1,
            int padding = 0,
            bool bias = true)
        {
            return new Conv2dInputParallel(
                inChannels, outChannels, kernelSize,
                processGroup: null,
                stride: stride,
                padding: padding,
                bias: bias);
        }

        /// <summary>
        /// Create input-channel parallel Conv2d with custom process group
        /// </summary>
        public static Conv2dInputParallel CreateInputParallel(
            int inChannels,
            int outChannels,
            int kernelSize,
            TensorParallelGroup processGroup,
            int stride = 1,
            int padding = 0,
            bool bias = true)
        {
            return new Conv2dInputParallel(
                inChannels, outChannels, kernelSize,
                processGroup: processGroup,
                stride: stride,
                padding: padding,
                bias: bias);
        }

        /// <summary>
        /// Create standard pattern: output-parallel then input-parallel
        /// Useful for bottleneck blocks
        /// </summary>
        public static (Conv2dOutputParallel, Conv2dInputParallel) CreateBottleneckPair(
            int inChannels,
            int bottleneckChannels,
            int outChannels,
            int kernelSize = 1,
            int stride = 1)
        {
            var conv1 = new Conv2dOutputParallel(
                inChannels, bottleneckChannels, kernelSize,
                gatherOutput: false,
                processGroup: null,
                stride: 1,
                padding: 0,
                bias: false);

            var conv2 = new Conv2dInputParallel(
                bottleneckChannels, outChannels, kernelSize,
                processGroup: null,
                stride: stride,
                padding: 0,
                bias: false);

            return (conv1, conv2);
        }

        /// <summary>
        /// Create bottleneck pair with custom process group
        /// </summary>
        public static (Conv2dOutputParallel, Conv2dInputParallel) CreateBottleneckPair(
            int inChannels,
            int bottleneckChannels,
            int outChannels,
            TensorParallelGroup processGroup,
            int kernelSize = 1,
            int stride = 1)
        {
            var conv1 = new Conv2dOutputParallel(
                inChannels, bottleneckChannels, kernelSize,
                gatherOutput: false,
                processGroup: processGroup,
                stride: 1,
                padding: 0,
                bias: false);

            var conv2 = new Conv2dInputParallel(
                bottleneckChannels, outChannels, kernelSize,
                processGroup: processGroup,
                stride: stride,
                padding: 0,
                bias: false);

            return (conv1, conv2);
        }
    }
}
