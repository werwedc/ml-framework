using MLFramework.Core;

namespace MLFramework.Fusion;

/// <summary>
/// Generates CUDA code for specific operations
/// </summary>
public static class CudaOperationCodeGenerator
{
    public static string GenerateAddCode(FusionOpNode node)
    {
        if (node.InputVars.Count == 2)
        {
            return $"{node.OutputVar} = {node.InputVars[0]} + {node.InputVars[1]};";
        }
        else if (node.InputVars.Count == 1 && node.Attributes.ContainsKey("scalar"))
        {
            var scalar = node.Attributes["scalar"];
            return $"{node.OutputVar} = {node.InputVars[0]} + {scalar};";
        }
        else
        {
            throw new InvalidOperationException("Invalid Add operation");
        }
    }

    public static string GenerateSubCode(FusionOpNode node)
    {
        if (node.InputVars.Count == 2)
        {
            return $"{node.OutputVar} = {node.InputVars[0]} - {node.InputVars[1]};";
        }
        else if (node.InputVars.Count == 1 && node.Attributes.ContainsKey("scalar"))
        {
            var scalar = node.Attributes["scalar"];
            return $"{node.OutputVar} = {node.InputVars[0]} - {scalar};";
        }
        else
        {
            throw new InvalidOperationException("Invalid Sub operation");
        }
    }

    public static string GenerateMulCode(FusionOpNode node)
    {
        if (node.InputVars.Count == 2)
        {
            return $"{node.OutputVar} = {node.InputVars[0]} * {node.InputVars[1]};";
        }
        else if (node.InputVars.Count == 1 && node.Attributes.ContainsKey("scalar"))
        {
            var scalar = node.Attributes["scalar"];
            return $"{node.OutputVar} = {node.InputVars[0]} * {scalar};";
        }
        else
        {
            throw new InvalidOperationException("Invalid Mul operation");
        }
    }

    public static string GenerateDivCode(FusionOpNode node)
    {
        if (node.InputVars.Count == 2)
        {
            return $"{node.OutputVar} = {node.InputVars[0]} / {node.InputVars[1]};";
        }
        else if (node.InputVars.Count == 1 && node.Attributes.ContainsKey("scalar"))
        {
            var scalar = node.Attributes["scalar"];
            return $"{node.OutputVar} = {node.InputVars[0]} / {scalar};";
        }
        else
        {
            throw new InvalidOperationException("Invalid Div operation");
        }
    }

    public static string GenerateReLUCode(FusionOpNode node)
    {
        return $"{node.OutputVar} = fmaxf(0.0f, {node.InputVars[0]});";
    }

    public static string GenerateSigmoidCode(FusionOpNode node)
    {
        return $"{node.OutputVar} = 1.0f / (1.0f + expf(-{node.InputVars[0]}));";
    }

    public static string GenerateTanhCode(FusionOpNode node)
    {
        return $"{node.OutputVar} = tanhf({node.InputVars[0]});";
    }

    public static string GenerateLeakyReLUCode(FusionOpNode node)
    {
        var alpha = node.Attributes.TryGetValue("alpha", out var a) ? a : 0.01f;
        return $"float leaky_{node.Id} = {alpha}f * {node.InputVars[0]}; " +
               $"{node.OutputVar} = {node.InputVars[0]} * ({node.InputVars[0]} > 0.0f ? 1.0f : 0.0f) + " +
               $"leaky_{node.Id} * ({node.InputVars[0]} <= 0.0f ? 1.0f : 0.0f);";
    }

    public static string GenerateExpCode(FusionOpNode node)
    {
        return $"{node.OutputVar} = expf({node.InputVars[0]});";
    }

    public static string GenerateLogCode(FusionOpNode node)
    {
        return $"{node.OutputVar} = logf({node.InputVars[0]});";
    }

    public static string GenerateSqrtCode(FusionOpNode node)
    {
        return $"{node.OutputVar} = sqrtf({node.InputVars[0]});";
    }

    public static string GenerateAbsCode(FusionOpNode node)
    {
        return $"{node.OutputVar} = fabsf({node.InputVars[0]});";
    }

    public static string GenerateOperationCode(FusionOpNode node)
    {
        return node.OriginalOpType switch
        {
            "Add" => GenerateAddCode(node),
            "Sub" => GenerateSubCode(node),
            "Mul" => GenerateMulCode(node),
            "Div" => GenerateDivCode(node),
            "ReLU" => GenerateReLUCode(node),
            "Sigmoid" => GenerateSigmoidCode(node),
            "Tanh" => GenerateTanhCode(node),
            "LeakyReLU" => GenerateLeakyReLUCode(node),
            "Exp" => GenerateExpCode(node),
            "Log" => GenerateLogCode(node),
            "Sqrt" => GenerateSqrtCode(node),
            "Abs" => GenerateAbsCode(node),
            _ => throw new NotSupportedException($"Unsupported operation type: {node.OriginalOpType}")
        };
    }
}
