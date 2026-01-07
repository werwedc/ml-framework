namespace MLFramework.Quantization.Backends.CPUBackend
{
    /// <summary>
    /// SIMD-optimized Int8 operations for CPU backend.
    /// </summary>
    internal class CPUInt8Operations
    {
        /// <summary>
        /// Performs int8 matrix multiplication.
        /// </summary>
        /// <param name="A">First matrix data (int8 values stored as float).</param>
        /// <param name="B">Second matrix data (int8 values stored as float).</param>
        /// <param name="C">Output matrix data.</param>
        /// <param name="m">Number of rows in A and C.</param>
        /// <param name="k">Number of columns in A and rows in B.</param>
        /// <param name="n">Number of columns in B and C.</param>
        /// <param name="outputScale">Scale factor for the output.</param>
        public void MatMulInt8(
            float[] A, float[] B, float[] C,
            int m, int k, int n, float outputScale)
        {
            Parallel.For(0, m, i =>
            {
                for (int j = 0; j < n; j++)
                {
                    float sum = MatMulInt8Scalar(A, B, i, j, m, k, n);
                    C[i * n + j] = sum * outputScale;
                }
            });
        }

        /// <summary>
        /// Scalar int8 matrix multiplication for a single element.
        /// </summary>
        private float MatMulInt8Scalar(float[] A, float[] B, int i, int j, int m, int k, int n)
        {
            int sum = 0;
            int aBase = i * k;
            int bBase = j;

            for (int kk = 0; kk < k; kk++)
            {
                sum += (int)(A[aBase + kk] * B[kk * n + bBase]);
            }

            return sum;
        }

        /// <summary>
        /// Performs int8 2D convolution.
        /// </summary>
        public void Conv2DInt8(
            float[] input, float[] weights, float[]? bias, float[] output,
            int batchSize, int inChannels, int inputHeight, int inputWidth,
            int outChannels, int kernelHeight, int kernelWidth,
            int outputHeight, int outputWidth,
            int strideHeight, int strideWidth,
            int paddingHeight, int paddingWidth,
            int dilationHeight, int dilationWidth,
            float outputScale)
        {
            Parallel.For(0, batchSize, b =>
            {
                for (int oc = 0; oc < outChannels; oc++)
                {
                    float biasVal = bias?[oc] ?? 0;

                    for (int oh = 0; oh < outputHeight; oh++)
                    {
                        for (int ow = 0; ow < outputWidth; ow++)
                        {
                            float sum = biasVal;

                            for (int ic = 0; ic < inChannels; ic++)
                            {
                                for (int kh = 0; kh < kernelHeight; kh++)
                                {
                                    for (int kw = 0; kw < kernelWidth; kw++)
                                    {
                                        int ih = oh * strideHeight - paddingHeight + kh * dilationHeight;
                                        int iw = ow * strideWidth - paddingWidth + kw * dilationWidth;

                                        if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                                        {
                                            int inputIdx = b * inChannels * inputHeight * inputWidth +
                                                           ic * inputHeight * inputWidth +
                                                           ih * inputWidth + iw;

                                            int weightIdx = oc * inChannels * kernelHeight * kernelWidth +
                                                            ic * kernelHeight * kernelWidth +
                                                            kh * kernelWidth + kw;

                                            sum += (int)(input[inputIdx] * weights[weightIdx]);
                                        }
                                    }
                                }
                            }

                            int outputIdx = b * outChannels * outputHeight * outputWidth +
                                            oc * outputHeight * outputWidth +
                                            oh * outputWidth + ow;

                            output[outputIdx] = sum * outputScale;
                        }
                    }
                }
            });
        }
    }
}
