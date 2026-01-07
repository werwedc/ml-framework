using MLFramework.Core;

namespace MLFramework.Fusion;

/// <summary>
/// Registry for CUDA kernel templates
/// </summary>
public static class CudaKernelTemplates
{
    /// <summary>
    /// Element-wise kernel template
    /// </summary>
    public const string ElementWiseTemplate = @"
template<typename T>
__global__ void {{kernel_name}}(
    {{#each parameters}}
    {{#if (eq Direction 'Input')}}const {{/if}}{{CudaType}}* {{#if (eq Direction 'Input')}}__restrict__ {{/if}}{{name}}{{#unless @last}}, {{/unless}}
    {{/each}}
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements)
        return;

    // Load input
    T val = {{input_var}}[idx];

    {{#each nodes}}
    // {{OriginalOpType}}
    {{#if (eq OriginalOpType 'Add')}}
    val = {{InputVars.[0]}} + {{InputVars.[1]}};
    {{/if}}
    {{#if (eq OriginalOpType 'ReLU')}}
    val = fmaxf(0.0f, val);
    {{/if}}
    {{#if (eq OriginalOpType 'Sigmoid')}}
    val = 1.0f / (1.0f + expf(-val));
    {{/if}}
    {{#if (eq OriginalOpType 'Exp')}}
    val = expf(val);
    {{/if}}
    {{/each}}

    // Store output
    {{output_var}}[idx] = val;
}";

    /// <summary>
    /// Convolution + activation kernel template
    /// </summary>
    public const string ConvActivationTemplate = @"
template<typename T>
__global__ void {{kernel_name}}(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding
) {
    extern __shared__ T shared_mem[];

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_c = out_idx % out_channels;
    int out_w = (out_idx / out_channels) % out_width;
    int out_h = (out_idx / out_channels / out_width) % out_height;
    int batch_idx = out_idx / (out_channels * out_height * out_width);

    if (batch_idx >= batch_size || out_c >= out_channels ||
        out_h >= out_height || out_w >= out_width)
        return;

    T sum = bias[out_c];

    // Convolution computation
    #pragma unroll
    for (int ic = 0; ic < in_channels; ic++) {
        #pragma unroll
        for (int kh = 0; kh < kernel_size; kh++) {
            int in_h = out_h * stride + kh - padding;
            if (in_h < 0 || in_h >= in_height) continue;

            #pragma unroll
            for (int kw = 0; kw < kernel_size; kw++) {
                int in_w = out_w * stride + kw - padding;
                if (in_w < 0 || in_w >= in_width) continue;

                int input_idx = ((batch_idx * in_channels + ic) * in_height + in_h) * in_width + in_w;
                int weight_idx = ((out_c * in_channels + ic) * kernel_size + kh) * kernel_size + kw;

                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    {{#if has_activation}}
    // Activation function
    {{activation_name}}(sum);
    {{/if}}

    output[out_idx] = sum;
}";

    /// <summary>
    /// Linear (fully connected) kernel template
    /// </summary>
    public const string LinearTemplate = @"
template<typename T>
__global__ void {{kernel_name}}(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_c = out_idx % out_features;
    int batch_idx = out_idx / out_features;

    if (batch_idx >= batch_size || out_c >= out_features)
        return;

    T sum = bias[out_c];

    // Matrix-vector multiplication
    #pragma unroll
    for (int in_c = 0; in_c < in_features; in_c++) {
        int input_idx = batch_idx * in_features + in_c;
        int weight_idx = out_c * in_features + in_c;
        sum += input[input_idx] * weight[weight_idx];
    }

    output[out_idx] = sum;
}";

    /// <summary>
    /// Generic kernel template
    /// </summary>
    public const string GenericTemplate = @"
template<typename T>
__global__ void {{kernel_name}}(
    {{#each parameters}}
    {{#if (eq Direction 'Input')}}const {{/if}}T* {{#if (eq Direction 'Input')}}__restrict__ {{/if}}{{name}}{{#unless @last}}, {{/unless}}
    {{/each}}
    {{#each variables}}
    int {{name}}_size{{#unless @last}}, {{/unless}}
    {{/each}}
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    {{#each variables}}
    if (idx >= {{name}}_size)
        return;
    {{/each}}

    // Generic operation execution
    {{#each nodes}}
    {{OriginalOpType}}({{InputVars}}, {{OutputVar}});
    {{/each}}
}";
}
