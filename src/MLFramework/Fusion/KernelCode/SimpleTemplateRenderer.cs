using System.Text;
using MLFramework.Core;

namespace MLFramework.Fusion;

/// <summary>
/// Simple template renderer with placeholder replacement
/// </summary>
public class SimpleTemplateRenderer : ITemplateRenderer
{
    public string Render(string template, TemplateContext context)
    {
        // Simple placeholder replacement
        var result = template;

        result = result.Replace("{{kernel_name}}", context.KernelName);
        result = result.Replace("{{shared_memory}}", context.MemoryLayout.SharedMemoryBytes.ToString());
        result = result.Replace("{{threads_per_block}}", context.ComputeRequirements.ThreadsPerBlock.ToString());

        // Generate parameter list
        var paramList = string.Join(", ", context.Parameters.Select(p =>
            $"{GetCudaTypeName(p.DataType)}* {p.Name}"));
        result = result.Replace("{{parameters}}", paramList);

        // Generate kernel body
        var body = GenerateKernelBody(context);
        result = result.Replace("{{kernel_body}}", body);

        return result;
    }

    private string GetCudaTypeName(DataType dtype)
    {
        return dtype switch
        {
            DataType.Float32 => "float",
            DataType.Float16 => "half",
            DataType.BFloat16 => "__nv_bfloat16",
            DataType.Int32 => "int",
            DataType.Int64 => "long long",
            DataType.Int16 => "short",
            DataType.Int8 => "sbyte",
            DataType.UInt8 => "unsigned char",
            DataType.Bool => "bool",
            _ => throw new NotSupportedException($"Unsupported dtype: {dtype}")
        };
    }

    private string GenerateKernelBody(TemplateContext context)
    {
        var sb = new StringBuilder();

        foreach (var node in context.Nodes)
        {
            sb.AppendLine(GenerateNodeCode(node, context));
        }

        return sb.ToString();
    }

    private string GenerateNodeCode(FusionOpNode node, TemplateContext context)
    {
        return node.OriginalOpType switch
        {
            "Add" => GenerateAddCode(node),
            "Mul" => GenerateMulCode(node),
            "ReLU" => GenerateReLUCode(node),
            "Sigmoid" => GenerateSigmoidCode(node),
            _ => $"// Unknown op type: {node.OriginalOpType}"
        };
    }

    private string GenerateAddCode(FusionOpNode node)
    {
        return $"{node.OutputVar} = {node.InputVars[0]} + {node.InputVars[1]};";
    }

    private string GenerateMulCode(FusionOpNode node)
    {
        return $"{node.OutputVar} = {node.InputVars[0]} * {node.InputVars[1]};";
    }

    private string GenerateReLUCode(FusionOpNode node)
    {
        return $"{node.OutputVar} = fmaxf(0.0f, {node.InputVars[0]});";
    }

    private string GenerateSigmoidCode(FusionOpNode node)
    {
        return $"{node.OutputVar} = 1.0f / (1.0f + expf(-{node.InputVars[0]}));";
    }
}
