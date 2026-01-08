using MLFramework.Core;
using MLFramework.Visualization.Graphs;
using System.Text.Json;

namespace MLFramework.Visualization.Graphs.Serialization;

/// <summary>
/// Serializes and deserializes computational graphs.
/// </summary>
public class GraphSerializer
{
    /// <summary>
    /// Serializes a computational graph to JSON.
    /// </summary>
    /// <param name="graph">The graph to serialize.</param>
    /// <returns>JSON string representation of the graph.</returns>
    public static string SerializeToJson(ComputationalGraph graph)
    {
        if (graph == null) throw new ArgumentNullException(nameof(graph));

        var graphData = new
        {
            name = graph.Name,
            step = graph.Step,
            timestamp = graph.Timestamp,
            nodeCount = graph.NodeCount,
            edgeCount = graph.EdgeCount,
            depth = graph.Depth,
            nodes = graph.Nodes.Values.Select(n => new
            {
                id = n.Id,
                name = n.Name,
                type = n.Type.ToString(),
                opType = n.OpType,
                shape = n.Shape,
                dataType = n.DataType.ToString(),
                inputs = n.InputIds,
                outputs = n.OutputIds,
                controlDependencies = n.ControlDependencies,
                attributes = n.Attributes,
                metadata = n.Metadata
            }),
            edges = graph.Edges
        };

        return JsonSerializer.Serialize(graphData, new JsonSerializerOptions
        {
            WriteIndented = true
        });
    }

    /// <summary>
    /// Serializes a computational graph to JSON bytes.
    /// </summary>
    /// <param name="graph">The graph to serialize.</param>
    /// <returns>JSON bytes representation of the graph.</returns>
    public static byte[] SerializeToJsonBytes(ComputationalGraph graph)
    {
        if (graph == null) throw new ArgumentNullException(nameof(graph));

        return JsonSerializer.SerializeToUtf8Bytes(SerializeToJson(graph));
    }

    /// <summary>
    /// Deserializes a computational graph from JSON.
    /// </summary>
    /// <param name="json">JSON string to deserialize.</param>
    /// <returns>Deserialized computational graph.</returns>
    public static ComputationalGraph DeserializeFromJson(string json)
    {
        if (string.IsNullOrEmpty(json)) throw new ArgumentException("JSON cannot be null or empty.", nameof(json));

        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;

        var name = root.GetProperty("name").GetString() ?? "Unknown";
        var step = root.GetProperty("step").GetInt64();

        var graph = new ComputationalGraph(name, step);

        // Deserialize nodes
        var nodesElement = root.GetProperty("nodes");
        foreach (var nodeElement in nodesElement.EnumerateArray())
        {
            var node = DeserializeNode(nodeElement);
            graph.AddNode(node);
        }

        // Deserialize edges
        var edgesElement = root.GetProperty("edges");
        foreach (var edgeElement in edgesElement.EnumerateArray())
        {
            var from = edgeElement.GetProperty("Item1").GetString();
            var to = edgeElement.GetProperty("Item2").GetString();

            if (from != null && to != null)
            {
                try
                {
                    graph.AddEdge(from, to);
                }
                catch (ArgumentException)
                {
                    // Edge may already exist or nodes may not exist, continue
                }
            }
        }

        return graph;
    }

    /// <summary>
    /// Deserializes a computational graph from JSON bytes.
    /// </summary>
    /// <param name="bytes">JSON bytes to deserialize.</param>
    /// <returns>Deserialized computational graph.</returns>
    public static ComputationalGraph DeserializeFromJsonBytes(byte[] bytes)
    {
        if (bytes == null || bytes.Length == 0) throw new ArgumentException("Bytes cannot be null or empty.", nameof(bytes));

        var json = System.Text.Encoding.UTF8.GetString(bytes);
        return DeserializeFromJson(json);
    }

    /// <summary>
    /// Serializes a computational graph to a binary format (for protobuf compatibility).
    /// </summary>
    /// <param name="graph">The graph to serialize.</param>
    /// <returns>Binary representation of the graph.</returns>
    public static byte[] SerializeToBinary(ComputationalGraph graph)
    {
        // For now, use JSON as the binary format
        // In a production system, you would use protobuf here
        return SerializeToJsonBytes(graph);
    }

    /// <summary>
    /// Deserializes a computational graph from binary format.
    /// </summary>
    /// <param name="bytes">Binary data to deserialize.</param>
    /// <returns>Deserialized computational graph.</returns>
    public static ComputationalGraph DeserializeFromBinary(byte[] bytes)
    {
        // For now, use JSON as the binary format
        // In a production system, you would use protobuf here
        return DeserializeFromJsonBytes(bytes);
    }

    private static GraphNode DeserializeNode(JsonElement nodeElement)
    {
        var id = nodeElement.GetProperty("id").GetString() ?? "";
        var name = nodeElement.GetProperty("name").GetString() ?? "";
        var typeString = nodeElement.GetProperty("type").GetString() ?? "";
        var opType = nodeElement.TryGetProperty("opType", out var opTypeElement) ? opTypeElement.GetString() ?? "" : "";

        var type = Enum.Parse<NodeType>(typeString);

        // Deserialize shape
        long[] shape = Array.Empty<long>();
        if (nodeElement.TryGetProperty("shape", out var shapeElement))
        {
            shape = new long[shapeElement.GetArrayLength()];
            int i = 0;
            foreach (var dimElement in shapeElement.EnumerateArray())
            {
                shape[i++] = dimElement.GetInt64();
            }
        }

        // Deserialize data type
        var dataType = DataType.Float32;
        if (nodeElement.TryGetProperty("dataType", out var dataTypeElement))
        {
            var dataTypeString = dataTypeElement.GetString() ?? "Float32";
            dataType = Enum.Parse<DataType>(dataTypeString);
        }

        // Deserialize inputs
        var inputIds = new List<string>();
        if (nodeElement.TryGetProperty("inputs", out var inputsElement))
        {
            foreach (var inputElement in inputsElement.EnumerateArray())
            {
                inputIds.Add(inputElement.GetString() ?? "");
            }
        }

        // Deserialize outputs
        var outputIds = new List<string>();
        if (nodeElement.TryGetProperty("outputs", out var outputsElement))
        {
            foreach (var outputElement in outputsElement.EnumerateArray())
            {
                outputIds.Add(outputElement.GetString() ?? "");
            }
        }

        // Deserialize control dependencies
        var controlDependencies = new List<string>();
        if (nodeElement.TryGetProperty("controlDependencies", out var controlDepsElement))
        {
            foreach (var depElement in controlDepsElement.EnumerateArray())
            {
                controlDependencies.Add(depElement.GetString() ?? "");
            }
        }

        // Deserialize attributes
        var attributes = new Dictionary<string, object>();
        if (nodeElement.TryGetProperty("attributes", out var attributesElement))
        {
            foreach (var attributeProperty in attributesElement.EnumerateObject())
            {
                var value = attributeProperty.Value;
                switch (value.ValueKind)
                {
                    case JsonValueKind.String:
                        attributes[attributeProperty.Name] = value.GetString() ?? "";
                        break;
                    case JsonValueKind.Number:
                        if (value.TryGetInt64(out long longValue))
                        {
                            attributes[attributeProperty.Name] = longValue;
                        }
                        else
                        {
                            attributes[attributeProperty.Name] = value.GetDouble();
                        }
                        break;
                    case JsonValueKind.True:
                    case JsonValueKind.False:
                        attributes[attributeProperty.Name] = value.GetBoolean();
                        break;
                    default:
                        attributes[attributeProperty.Name] = value.ToString();
                        break;
                }
            }
        }

        // Deserialize metadata
        var metadata = new Dictionary<string, string>();
        if (nodeElement.TryGetProperty("metadata", out var metadataElement))
        {
            foreach (var metadataProperty in metadataElement.EnumerateObject())
            {
                metadata[metadataProperty.Name] = metadataProperty.Value.GetString() ?? "";
            }
        }

        return new GraphNode(
            id,
            name,
            type,
            opType,
            shape,
            dataType,
            inputIds,
            outputIds,
            controlDependencies,
            attributes,
            metadata);
    }
}
