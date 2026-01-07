using System;
using System.Collections.Generic;
using System.Linq;
using RitterFramework.Core.Tensor;
using MLFramework.Modules;
using MLFramework.NN;

namespace MLFramework.Conversion;

/// <summary>
/// Provides static methods to analyze models for tensor parallelism compatibility.
/// </summary>
public static class TPModelAnalyzer
{
    /// <summary>
    /// Analyze a model to determine TP compatibility and configuration.
    /// </summary>
    /// <param name="model">The model to analyze.</param>
    /// <param name="maxWorldSize">The maximum world size to consider.</param>
    /// <returns>A report containing the analysis results.</returns>
    public static TPAnalysisReport Analyze(IModule model, int maxWorldSize = 8)
    {
        var report = new TPAnalysisReport();
        AnalyzeModule(model, report, "");

        // Determine suggested world size
        report.SuggestedWorldSize = CalculateSuggestedWorldSize(report, maxWorldSize);

        // Generate recommendations
        GenerateRecommendations(report);

        return report;
    }

    private static void AnalyzeModule(IModule module, TPAnalysisReport report, string prefix)
    {
        // Check if this module is a container with parameters
        if (module is Modules.Linear linearLayer)
        {
            AnalyzeLinearLayer(linearLayer, report, prefix);
        }
        else if (module is Conv2d convLayer)
        {
            AnalyzeConv2dLayer(convLayer, report, prefix);
        }
        else if (module is HierarchicalWrapper wrapper)
        {
            // For hierarchical wrappers, analyze the wrapped module
            string newPrefix = string.IsNullOrEmpty(prefix)
                ? wrapper.Name
                : $"{prefix}.{wrapper.Name}";
            AnalyzeModule(wrapper.Module, report, newPrefix);
        }
        else
        {
            // For other modules, try to get parameters if available
            AnalyzeGenericModule(module, report, prefix);
        }
    }

    private static void AnalyzeLinearLayer(Modules.Linear linearLayer, TPAnalysisReport report, string prefix)
    {
        string fullName = string.IsNullOrEmpty(prefix) ? "linear" : $"{prefix}.linear";

        // Analyze weight
        if (linearLayer.Weight != null)
        {
            var result = DetermineLinearParallelism(
                linearLayer.Weight,
                fullName,
                "weight",
                linearLayer.InFeatures,
                linearLayer.OutFeatures);
            report.AddLayer(result);
        }

        // Analyze bias
        if (linearLayer.Bias != null)
        {
            long memoryBytes = linearLayer.Bias.Data.Length * sizeof(float);
            long paramCount = linearLayer.Bias.Data.Length;

            // Bias is typically not parallelized separately
            var result = LayerAnalysisResult.NotParallelizable(
                $"{fullName}.bias",
                "LinearBias",
                memoryBytes,
                paramCount);
            report.AddLayer(result);
        }
    }

    private static void AnalyzeConv2dLayer(Conv2d convLayer, TPAnalysisReport report, string prefix)
    {
        string fullName = string.IsNullOrEmpty(prefix) ? "conv2d" : $"{prefix}.conv2d";

        // Analyze weight
        if (convLayer.Weight != null)
        {
            var result = DetermineConv2dParallelism(
                convLayer.Weight,
                fullName,
                convLayer.InChannels,
                convLayer.OutChannels);
            report.AddLayer(result);
        }

        // Analyze bias
        if (convLayer.Bias != null)
        {
            long memoryBytes = convLayer.Bias.Data.Length * sizeof(float);
            long paramCount = convLayer.Bias.Data.Length;

            // Bias is typically not parallelized separately
            var result = LayerAnalysisResult.NotParallelizable(
                $"{fullName}.bias",
                "Conv2dBias",
                memoryBytes,
                paramCount);
            report.AddLayer(result);
        }
    }

    private static void AnalyzeGenericModule(IModule module, TPAnalysisReport report, string prefix)
    {
        // Try to get parameters via reflection or interface
        // This is a simplified version - in practice, you'd have better introspection
        var paramProperty = module.GetType().GetProperty("Parameters");
        if (paramProperty != null)
        {
            var parameters = paramProperty.GetValue(module) as IEnumerable<Tensor>;
            if (parameters != null)
            {
                int paramIndex = 0;
                foreach (var param in parameters)
                {
                    if (param != null)
                    {
                        string fullName = string.IsNullOrEmpty(prefix)
                            ? $"param_{paramIndex}"
                            : $"{prefix}.param_{paramIndex}";

                        long memoryBytes = param.Data.Length * sizeof(float);
                        long paramCount = param.Data.Length;

                        // Try to determine if parameter is parallelizable based on shape
                        var result = DetermineParameterParallelism(
                            param,
                            fullName,
                            memoryBytes,
                            paramCount);

                        report.AddLayer(result);
                        paramIndex++;
                    }
                }
            }
        }
    }

    private static LayerAnalysisResult DetermineLinearParallelism(
        Tensor weight,
        string fullName,
        string paramName,
        int inFeatures,
        int outFeatures)
    {
        long memoryBytes = weight.Data.Length * sizeof(float);
        long paramCount = weight.Data.Length;

        // Determine if layer is parallelizable based on dimensions
        // Large output dimension suggests column parallelism
        if (outFeatures > inFeatures * 2)
        {
            return LayerAnalysisResult.ColumnParallel(
                $"{fullName}.{paramName}",
                "Linear",
                memoryBytes,
                paramCount);
        }
        // Large input dimension suggests row parallelism
        else if (inFeatures > outFeatures * 2)
        {
            return LayerAnalysisResult.RowParallel(
                $"{fullName}.{paramName}",
                "Linear",
                memoryBytes,
                paramCount);
        }

        // Not clearly parallelizable
        return LayerAnalysisResult.NotParallelizable(
            $"{fullName}.{paramName}",
            "Linear",
            memoryBytes,
            paramCount);
    }

    private static LayerAnalysisResult DetermineConv2dParallelism(
        Tensor weight,
        string fullName,
        int inChannels,
        int outChannels)
    {
        long memoryBytes = weight.Data.Length * sizeof(float);
        long paramCount = weight.Data.Length;

        // Output channel parallelism is more common for Conv2d
        if (outChannels > 64) // Threshold for parallelism
        {
            return LayerAnalysisResult.ColumnParallel(
                $"{fullName}.weight",
                "Conv2d",
                memoryBytes,
                paramCount);
        }

        // Not clearly parallelizable
        return LayerAnalysisResult.NotParallelizable(
            $"{fullName}.weight",
            "Conv2d",
            memoryBytes,
            paramCount);
    }

    private static LayerAnalysisResult DetermineParameterParallelism(
        Tensor param,
        string fullName,
        long memoryBytes,
        long paramCount)
    {
        // Check parameter shape to guess if it's parallelizable
        if (param.Shape.Length == 2)
        {
            int dim0 = param.Shape[0];
            int dim1 = param.Shape[1];

            // Large dimension ratio suggests parallelizability
            if (dim0 > dim1 * 2)
            {
                return LayerAnalysisResult.ColumnParallel(
                    fullName,
                    "Linear",
                    memoryBytes,
                    paramCount);
            }
            else if (dim1 > dim0 * 2)
            {
                return LayerAnalysisResult.RowParallel(
                    fullName,
                    "Linear",
                    memoryBytes,
                    paramCount);
            }
        }

        // Not parallelizable by default
        return LayerAnalysisResult.NotParallelizable(
            fullName,
            "Unknown",
            memoryBytes,
            paramCount);
    }

    private static int CalculateSuggestedWorldSize(TPAnalysisReport report, int maxWorldSize)
    {
        if (report.ParallelizableMemoryBytes == 0)
            return 1;

        // Simple heuristic: divide memory by target per-device memory
        // Assume target of 4GB per device for now
        const long targetMemoryPerDevice = 4L * 1024 * 1024 * 1024;

        int suggestedWorldSize = (int)Math.Ceiling(
            (double)report.ParallelizableMemoryBytes / targetMemoryPerDevice);

        return Math.Clamp(suggestedWorldSize, 1, maxWorldSize);
    }

    private static void GenerateRecommendations(TPAnalysisReport report)
    {
        report.Recommendations.Clear();

        if (report.ParallelizablePercentage > 80)
        {
            report.Recommendations["Parallelization"] = "Highly recommended for this model";
        }
        else if (report.ParallelizablePercentage > 50)
        {
            report.Recommendations["Parallelization"] = "Recommended with significant memory savings";
        }
        else if (report.ParallelizablePercentage > 20)
        {
            report.Recommendations["Parallelization"] = "Moderate benefit, consider hybrid approach";
        }
        else
        {
            report.Recommendations["Parallelization"] = "Low benefit, consider other strategies";
        }

        if (report.SuggestedWorldSize > 4)
        {
            report.Recommendations["World Size"] =
                $"Large world size suggested ({report.SuggestedWorldSize}). Consider multi-node setup.";
        }
    }
}
