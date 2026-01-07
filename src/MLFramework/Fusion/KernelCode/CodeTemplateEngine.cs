namespace MLFramework.Fusion;

/// <summary>
/// Code template engine for kernel generation
/// </summary>
public class CodeTemplateEngine : ICodeTemplateEngine
{
    private readonly Dictionary<string, string> _templates = new();
    private readonly ITemplateRenderer _renderer;

    public CodeTemplateEngine(ITemplateRenderer renderer)
    {
        _renderer = renderer;
    }

    public void RegisterTemplate(string templateName, string templateContent)
    {
        _templates[templateName] = templateContent;
    }

    public string LoadTemplate(string templateName)
    {
        if (!_templates.TryGetValue(templateName, out var template))
            throw new KeyNotFoundException($"Template '{templateName}' not found");

        return template;
    }

    public string Render(string template, TemplateContext context)
    {
        return _renderer.Render(template, context);
    }
}
