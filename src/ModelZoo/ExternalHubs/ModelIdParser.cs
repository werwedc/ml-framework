namespace MLFramework.ModelZoo.ExternalHubs;

/// <summary>
/// Represents a parsed model identifier with its components.
/// </summary>
public class ModelIdComponents
{
    /// <summary>
    /// Gets or sets the hub name (e.g., "huggingface", "tensorflow", "onnx", null for local).
    /// </summary>
    public string? HubName { get; set; }

    /// <summary>
    /// Gets or sets the model name (e.g., "bert-base-uncased").
    /// </summary>
    public string ModelName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the model version (optional).
    /// </summary>
    public string? Version { get; set; }

    /// <summary>
    /// Gets or sets the model variant (optional).
    /// </summary>
    public string? Variant { get; set; }

    /// <summary>
    /// Gets a value indicating whether this is a local model (no hub specified).
    /// </summary>
    public bool IsLocal => HubName == null;

    /// <summary>
    /// Gets the full model identifier string.
    /// </summary>
    public string FullId
    {
        get
        {
            if (IsLocal)
            {
                return ModelName;
            }

            var parts = new List<string> { HubName!, ModelName };
            if (!string.IsNullOrEmpty(Version))
            {
                parts.Add(Version);
            }
            if (!string.IsNullOrEmpty(Variant))
            {
                parts.Add(Variant);
            }

            return $"hub:{string.Join("/", parts)}";
        }
    }
}

/// <summary>
/// Parser for model identifiers.
/// </summary>
public static class ModelIdParser
{
    /// <summary>
    /// Parses a model identifier string into its components.
    /// </summary>
    /// <param name="modelId">The model identifier string to parse.</param>
    /// <returns>A ModelIdComponents object representing the parsed identifier.</returns>
    /// <exception cref="ArgumentException">Thrown when the model ID is null or empty.</exception>
    /// <exception cref="FormatException">Thrown when the model ID format is invalid.</exception>
    public static ModelIdComponents Parse(string modelId)
    {
        if (string.IsNullOrWhiteSpace(modelId))
        {
            throw new ArgumentException("Model ID cannot be null or empty.", nameof(modelId));
        }

        var components = new ModelIdComponents();

        // Check if it's a hub-prefixed model ID
        if (modelId.StartsWith("hub:"))
        {
            var unprefixed = modelId[4..]; // Remove "hub:" prefix

            // Check for custom registry format: hub:custom:registry/model
            if (unprefixed.Contains(':'))
            {
                var colonIndex = unprefixed.IndexOf(':');
                components.HubName = unprefixed[..colonIndex];
                unprefixed = unprefixed[(colonIndex + 1)..];
            }
            else
            {
                // Parse hub and model from format: hub:huggingface/bert-base-uncased
                var slashIndex = unprefixed.IndexOf('/');
                if (slashIndex < 0)
                {
                    throw new FormatException($"Invalid model ID format: '{modelId}'. Expected format: 'hub:huggingface/bert-base-uncased'");
                }

                components.HubName = unprefixed[..slashIndex];
                unprefixed = unprefixed[(slashIndex + 1)..];
            }

            // Parse model name, version, and variant
            var parts = unprefixed.Split('/');
            if (parts.Length == 0)
            {
                throw new FormatException($"Invalid model ID format: '{modelId}'. Model name is required.");
            }

            components.ModelName = parts[0];
            if (parts.Length > 1)
            {
                components.Version = parts[1];
            }
            if (parts.Length > 2)
            {
                components.Variant = parts[2];
            }
        }
        else
        {
            // Local model (no hub prefix)
            components.ModelName = modelId;
        }

        return components;
    }

    /// <summary>
    /// Attempts to parse a model identifier string.
    /// </summary>
    /// <param name="modelId">The model identifier string to parse.</param>
    /// <param name="components">When this method returns, contains the parsed components if successful; otherwise, null.</param>
    /// <returns>True if the parsing was successful; otherwise, false.</returns>
    public static bool TryParse(string modelId, out ModelIdComponents? components)
    {
        try
        {
            components = Parse(modelId);
            return true;
        }
        catch
        {
            components = null;
            return false;
        }
    }

    /// <summary>
    /// Determines if a model ID is for a local model or a hub model.
    /// </summary>
    /// <param name="modelId">The model identifier string.</param>
    /// <returns>True if the model ID represents a local model; false if it's a hub model.</returns>
    public static bool IsLocalModel(string modelId)
    {
        return !modelId.StartsWith("hub:");
    }
}
