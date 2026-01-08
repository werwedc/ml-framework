using System.Collections.Concurrent;
using System.Text.RegularExpressions;
using MLFramework.Fusion.Backends;

namespace MLFramework.Serving.Routing;

/// <summary>
/// Implementation of header-based routing
/// </summary>
public class HeaderRouter : IHeaderRouter
{
    private readonly ConcurrentDictionary<string, ConcurrentDictionary<string, RoutingRule>> _modelRules = new();
    private readonly ConcurrentDictionary<string, Regex> _regexCache = new();
    private readonly ILogger _logger;

    public HeaderRouter(ILogger logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <inheritdoc/>
    public void RegisterRoutingRule(string modelName, RoutingRule rule)
    {
        if (string.IsNullOrWhiteSpace(modelName))
            throw new ArgumentException("Model name cannot be empty", nameof(modelName));

        if (rule == null)
            throw new ArgumentNullException(nameof(rule));

        // Validate rule
        if (string.IsNullOrWhiteSpace(rule.HeaderName))
            throw new ArgumentException("Header name cannot be empty", nameof(rule));

        if (string.IsNullOrWhiteSpace(rule.HeaderValue))
            throw new ArgumentException("Header value cannot be empty", nameof(rule));

        if (string.IsNullOrWhiteSpace(rule.TargetVersion))
            throw new ArgumentException("Target version cannot be empty", nameof(rule));

        // Pre-compile and cache regex for performance
        if (rule.MatchType == MatchType.Regex)
        {
            try
            {
                var regex = new Regex(rule.HeaderValue, RegexOptions.Compiled);
                _regexCache[rule.Id] = regex;
            }
            catch (ArgumentException ex)
            {
                throw new ArgumentException($"Invalid regex pattern: {rule.HeaderValue}", ex);
            }
        }

        rule.ModelName = modelName;

        // Add to model's rules
        var rules = _modelRules.GetOrAdd(modelName, _ => new ConcurrentDictionary<string, RoutingRule>());
        rules[rule.Id] = rule;

        _logger.LogInformation($"Registered routing rule {rule.Id} for model {modelName}: {rule.HeaderName} -> {rule.TargetVersion}");
    }

    /// <inheritdoc/>
    public void UnregisterRoutingRule(string modelName, string ruleId)
    {
        if (string.IsNullOrWhiteSpace(modelName))
            throw new ArgumentException("Model name cannot be empty", nameof(modelName));

        if (string.IsNullOrWhiteSpace(ruleId))
            throw new ArgumentException("Rule ID cannot be empty", nameof(ruleId));

        if (_modelRules.TryGetValue(modelName, out var rules))
        {
            if (rules.TryRemove(ruleId, out _))
            {
                _regexCache.TryRemove(ruleId, out _);
                _logger.LogInformation($"Unregistered routing rule {ruleId} for model {modelName}");
            }
        }
    }

    /// <inheritdoc/>
    public string? RouteByHeaders(string modelName, RoutingContext context)
    {
        if (string.IsNullOrWhiteSpace(modelName))
            throw new ArgumentException("Model name cannot be empty", nameof(modelName));

        if (context == null || context.Headers == null || context.Headers.Count == 0)
            return null;

        if (!_modelRules.TryGetValue(modelName, out var rules))
            return null;

        // Get rules sorted by priority (highest first)
        var sortedRules = rules.Values
            .OrderByDescending(r => r.Priority)
            .ToList();

        foreach (var rule in sortedRules)
        {
            if (!context.Headers.TryGetValue(rule.HeaderName, out var headerValue))
                continue;

            if (EvaluateRule(rule, headerValue))
            {
                _logger.LogInformation($"Rule {rule.Id} matched for model {modelName}: {rule.HeaderName}={headerValue} -> {rule.TargetVersion}");
                return rule.TargetVersion;
            }
        }

        return null;
    }

    /// <inheritdoc/>
    public IEnumerable<RoutingRule> GetRules(string modelName)
    {
        if (string.IsNullOrWhiteSpace(modelName))
            throw new ArgumentException("Model name cannot be empty", nameof(modelName));

        if (_modelRules.TryGetValue(modelName, out var rules))
        {
            return rules.Values.OrderByDescending(r => r.Priority).ToList();
        }

        return Enumerable.Empty<RoutingRule>();
    }

    /// <inheritdoc/>
    public void ClearRules(string modelName)
    {
        if (string.IsNullOrWhiteSpace(modelName))
            throw new ArgumentException("Model name cannot be empty", nameof(modelName));

        if (_modelRules.TryRemove(modelName, out var rules))
        {
            foreach (var ruleId in rules.Keys)
            {
                _regexCache.TryRemove(ruleId, out _);
            }

            _logger.LogInformation($"Cleared all routing rules for model {modelName}");
        }
    }

    private bool EvaluateRule(RoutingRule rule, string headerValue)
    {
        switch (rule.MatchType)
        {
            case MatchType.Exact:
                return string.Equals(headerValue, rule.HeaderValue, StringComparison.Ordinal);

            case MatchType.Prefix:
                return headerValue.StartsWith(rule.HeaderValue, StringComparison.Ordinal);

            case MatchType.Contains:
                return headerValue.Contains(rule.HeaderValue, StringComparison.Ordinal);

            case MatchType.Regex:
                if (_regexCache.TryGetValue(rule.Id, out var regex))
                {
                    return regex.IsMatch(headerValue);
                }
                return false;

            default:
                return false;
        }
    }
}
