namespace MLFramework.Fusion;

/// <summary>
/// Registers CUDA kernel templates with the template engine
/// </summary>
public static class CudaTemplateRegistrar
{
    /// <summary>
    /// Registers all CUDA kernel templates with the given template engine
    /// </summary>
    public static void RegisterTemplates(ICodeTemplateEngine engine)
    {
        engine.RegisterTemplate("cuda_elementwise_template", CudaKernelTemplates.ElementWiseTemplate);
        engine.RegisterTemplate("cuda_conv_activation_template", CudaKernelTemplates.ConvActivationTemplate);
        engine.RegisterTemplate("cuda_linear_template", CudaKernelTemplates.LinearTemplate);
        engine.RegisterTemplate("cuda_conv_template", CudaKernelTemplates.ConvActivationTemplate);
        engine.RegisterTemplate("cuda_generic_template", CudaKernelTemplates.GenericTemplate);
    }
}
