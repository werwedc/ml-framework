using System.Text.Json;
using System.Text.Json.Serialization;

namespace MLFramework.Serving.Deployment;

/// <summary>
/// JSON-based implementation of IMetadataSerializer using System.Text.Json.
/// </summary>
public class MetadataSerializer : IMetadataSerializer
{
    private readonly JsonSerializerOptions _options;

    /// <summary>
    /// Creates a new MetadataSerializer with default JSON options.
    /// </summary>
    public MetadataSerializer()
    {
        _options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            Converters = { new ObjectDictionaryConverter() }
        };
    }

    /// <summary>
    /// Creates a new MetadataSerializer with custom JSON options.
    /// </summary>
    /// <param name="options">Custom JSON serializer options.</param>
    public MetadataSerializer(JsonSerializerOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <inheritdoc />
    public string Serialize(ModelMetadata metadata)
    {
        if (metadata == null)
            throw new ArgumentNullException(nameof(metadata));

        ValidateMetadata(metadata);

        return JsonSerializer.Serialize(metadata, _options);
    }

    /// <inheritdoc />
    public ModelMetadata Deserialize(string json)
    {
        if (string.IsNullOrWhiteSpace(json))
            throw new ArgumentException("JSON string cannot be null or whitespace", nameof(json));

        var metadata = JsonSerializer.Deserialize<ModelMetadata>(json, _options)
            ?? throw new JsonException("Failed to deserialize ModelMetadata: null result");

        ValidateMetadata(metadata);

        return metadata;
    }

    /// <inheritdoc />
    public void SaveToFile(string filePath, ModelMetadata metadata)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be null or whitespace", nameof(filePath));

        if (metadata == null)
            throw new ArgumentNullException(nameof(metadata));

        var json = Serialize(metadata);
        File.WriteAllText(filePath, json);
    }

    /// <inheritdoc />
    public ModelMetadata LoadFromFile(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be null or whitespace", nameof(filePath));

        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Metadata file not found: {filePath}", filePath);

        var json = File.ReadAllText(filePath);
        return Deserialize(json);
    }

    /// <inheritdoc />
    public Task<string> SerializeAsync(ModelMetadata metadata)
    {
        return Task.FromResult(Serialize(metadata));
    }

    /// <inheritdoc />
    public Task<ModelMetadata> DeserializeAsync(string json)
    {
        return Task.FromResult(Deserialize(json));
    }

    /// <summary>
    /// Validates that required metadata fields are present and valid.
    /// </summary>
    private static void ValidateMetadata(ModelMetadata metadata)
    {
        if (string.IsNullOrWhiteSpace(metadata.Version))
            throw new InvalidOperationException("ModelMetadata.Version is required and cannot be empty");

        if (string.IsNullOrWhiteSpace(metadata.ArtifactPath))
            throw new InvalidOperationException("ModelMetadata.ArtifactPath is required and cannot be empty");
    }

    /// <summary>
    /// Custom JSON converter for handling Dictionary<string, object> with mixed value types.
    /// </summary>
    private class ObjectDictionaryConverter : JsonConverter<Dictionary<string, object>>
    {
        public override Dictionary<string, object> Read(
            ref Utf8JsonReader reader,
            Type typeToConvert,
            JsonSerializerOptions options)
        {
            if (reader.TokenType != JsonTokenType.StartObject)
                throw new JsonException("Expected StartObject token");

            var dictionary = new Dictionary<string, object>();

            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndObject)
                    break;

                // Get the property name
                if (reader.TokenType != JsonTokenType.PropertyName)
                    throw new JsonException("Expected PropertyName token");

                var propertyName = reader.GetString();

                // Read the value
                reader.Read();
                var value = ReadValue(ref reader, options);

                // propertyName is guaranteed to be non-null here since GetString() would have thrown
                dictionary[propertyName ?? string.Empty] = value;
            }

            return dictionary;
        }

        public override void Write(
            Utf8JsonWriter writer,
            Dictionary<string, object> value,
            JsonSerializerOptions options)
        {
            writer.WriteStartObject();

            foreach (var kvp in value)
            {
                writer.WritePropertyName(kvp.Key);
                WriteValue(writer, kvp.Value, options);
            }

            writer.WriteEndObject();
        }

        private static object ReadValue(ref Utf8JsonReader reader, JsonSerializerOptions options)
        {
            switch (reader.TokenType)
            {
                case JsonTokenType.String:
                    return reader.GetString() ?? string.Empty;

                case JsonTokenType.Number:
                    if (reader.TryGetInt32(out int intValue))
                        return intValue;
                    if (reader.TryGetInt64(out long longValue))
                        return longValue;
                    return reader.GetDouble();

                case JsonTokenType.True:
                case JsonTokenType.False:
                    return reader.GetBoolean();

                case JsonTokenType.Null:
                    return null!;

                case JsonTokenType.StartObject:
                    return JsonSerializer.Deserialize<Dictionary<string, object>>(ref reader, options)
                        ?? new Dictionary<string, object>();

                case JsonTokenType.StartArray:
                    var list = new List<object>();
                    while (reader.Read() && reader.TokenType != JsonTokenType.EndArray)
                    {
                        list.Add(ReadValue(ref reader, options));
                    }
                    return list;

                default:
                    throw new JsonException($"Unsupported token type: {reader.TokenType}");
            }
        }

        private static void WriteValue(Utf8JsonWriter writer, object value, JsonSerializerOptions options)
        {
            switch (value)
            {
                case null:
                    writer.WriteNullValue();
                    break;

                case string str:
                    writer.WriteStringValue(str);
                    break;

                case int i:
                    writer.WriteNumberValue(i);
                    break;

                case long l:
                    writer.WriteNumberValue(l);
                    break;

                case double d:
                    writer.WriteNumberValue(d);
                    break;

                case float f:
                    writer.WriteNumberValue(f);
                    break;

                case decimal dec:
                    writer.WriteNumberValue(dec);
                    break;

                case bool b:
                    writer.WriteBooleanValue(b);
                    break;

                case DateTime dt:
                    writer.WriteStringValue(dt);
                    break;

                case DateTimeOffset dto:
                    writer.WriteStringValue(dto);
                    break;

                case Guid guid:
                    writer.WriteStringValue(guid);
                    break;

                case Dictionary<string, object> dict:
                    JsonSerializer.Serialize(writer, dict, options);
                    break;

                case IEnumerable<object> enumerable:
                    writer.WriteStartArray();
                    foreach (var item in enumerable)
                    {
                        WriteValue(writer, item, options);
                    }
                    writer.WriteEndArray();
                    break;

                default:
                    // For unknown types, try to serialize as JSON string
                    JsonSerializer.Serialize(writer, value, value.GetType(), options);
                    break;
            }
        }
    }
}
