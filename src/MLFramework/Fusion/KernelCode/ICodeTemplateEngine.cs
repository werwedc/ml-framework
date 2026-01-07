namespace MLFramework.Fusion;

/// <summary>
/// Template system for generating kernel code
/// </summary>
public interface ICodeTemplateEngine
{
    string Render(string template, TemplateContext context);
    string LoadTemplate(string templateName);
    void RegisterTemplate(string templateName, string templateContent);
}

/// <summary>
/// Context for template rendering
/// </summary>
public record TemplateContext
{
    public required string KernelName { get; init; }
    public required IReadOnlyList<KernelParameter> Parameters { get; init; }
    public required IReadOnlyList<FusionOpNode> Nodes { get; init; }
    public required MemoryLayout MemoryLayout { get; init; }
    public required ComputeRequirements ComputeRequirements { get; init; }
    public required GenerationOptions Options { get; init; }
}

/// <summary>
/// Interface for template rendering
/// </summary>
public interface ITemplateRenderer
{
    string Render(string template, TemplateContext context);
}
