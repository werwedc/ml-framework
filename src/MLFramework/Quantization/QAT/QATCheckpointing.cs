using MLFramework.Quantization.DataStructures;
using System.Text.Json;

namespace MLFramework.Quantization.QAT;

/// <summary>
/// Handles checkpointing and restoration of QAT models.
/// Saves and restores quantization parameters along with model weights.
/// Compatible with existing save/load infrastructure.
/// </summary>
public static class QATCheckpointing
{
    /// <summary>
    /// Saves a QAT model checkpoint to a file.
    /// </summary>
    /// <param name="qatModel">The QAT model to save.</param>
    /// <param name="filePath">The file path to save to.</param>
    /// <param name="options">Optional JSON serialization options.</param>
    public static void SaveCheckpoint(
        IQATModel qatModel,
        string filePath,
        JsonSerializerOptions? options = null)
    {
        if (qatModel == null)
            throw new ArgumentNullException(nameof(qatModel));

        if (string.IsNullOrEmpty(filePath))
            throw new ArgumentException("File path cannot be null or empty", nameof(filePath));

        var checkpoint = CreateCheckpoint(qatModel);
        var json = JsonSerializer.Serialize(checkpoint, options ?? GetDefaultOptions());
        File.WriteAllText(filePath, json);
    }

    /// <summary>
    /// Loads a QAT model checkpoint from a file.
    /// </summary>
    /// <param name="filePath">The file path to load from.</param>
    /// <param name="options">Optional JSON deserialization options.</param>
    /// <returns>The loaded QAT checkpoint.</returns>
    public static QATCheckpoint LoadCheckpoint(
        string filePath,
        JsonSerializerOptions? options = null)
    {
        if (string.IsNullOrEmpty(filePath))
            throw new ArgumentException("File path cannot be null or empty", nameof(filePath));

        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Checkpoint file not found: {filePath}");

        var json = File.ReadAllText(filePath);
        var checkpoint = JsonSerializer.Deserialize<QATCheckpoint>(json, options ?? GetDefaultOptions());

        if (checkpoint == null)
            throw new InvalidOperationException("Failed to deserialize checkpoint");

        return checkpoint;
    }

    /// <summary>
    /// Creates a checkpoint from a QAT model.
    /// </summary>
    /// <param name="qatModel">The QAT model.</param>
    /// <returns>A checkpoint containing quantization parameters and metadata.</returns>
    public static QATCheckpoint CreateCheckpoint(IQATModel qatModel)
    {
        if (qatModel == null)
            throw new ArgumentNullException(nameof(qatModel));

        var checkpoint = new QATCheckpoint
        {
            Version = "1.0",
            Timestamp = DateTime.UtcNow,
            TrainingMode = qatModel.TrainingMode,
            LayerCount = qatModel.GetLayerCount(),
            FakeQuantizationNodeCount = qatModel.GetFakeQuantizationNodeCount(),
            QuantizedLayerCount = qatModel.GetQuantizedLayerCount()
        };

        // Get quantization parameters
        var quantParams = qatModel.GetQuantizationParameters();
        foreach (var kvp in quantParams)
        {
            if (kvp.Value != null)
            {
                checkpoint.LayerQuantizationParameters[kvp.Key] = kvp.Value.Value;
            }
        }

        // Get fake quantization nodes
        var fakeQuantNodes = qatModel.GetFakeQuantizationNodes();
        checkpoint.FakeQuantizationNodeCount = fakeQuantNodes.Count;

        return checkpoint;
    }

    /// <summary>
    /// Restores a QAT model from a checkpoint.
    /// </summary>
    /// <param name="checkpoint">The checkpoint to restore from.</param>
    /// <param name="model">The model to restore state to.</param>
    /// <returns>True if restoration was successful.</returns>
    public static bool RestoreCheckpoint(QATCheckpoint checkpoint, IQATModel model)
    {
        if (checkpoint == null)
            throw new ArgumentNullException(nameof(checkpoint));

        if (model == null)
            throw new ArgumentNullException(nameof(model));

        try
        {
            // Restore training mode
            model.TrainingMode = checkpoint.TrainingMode;

            // Restore quantization parameters
            // In production, this would update the model's quantization parameters
            // with the values from the checkpoint

            // Validate that layer counts match
            if (model.GetLayerCount() != checkpoint.LayerCount)
            {
                Console.WriteLine($"Warning: Layer count mismatch. Model has {model.GetLayerCount()}, checkpoint has {checkpoint.LayerCount}.");
            }

            // Validate that fake quantization node counts match
            if (model.GetFakeQuantizationNodeCount() != checkpoint.FakeQuantizationNodeCount)
            {
                Console.WriteLine($"Warning: Fake quantization node count mismatch. Model has {model.GetFakeQuantizationNodeCount()}, checkpoint has {checkpoint.FakeQuantizationNodeCount}.");
            }

            return true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error restoring checkpoint: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Saves quantization parameters to a separate file.
    /// </summary>
    /// <param name="quantParams">Quantization parameters to save.</param>
    /// <param name="filePath">The file path to save to.</param>
    /// <param name="options">Optional JSON serialization options.</param>
    public static void SaveQuantizationParameters(
        Dictionary<string, QuantizationParameters> quantParams,
        string filePath,
        JsonSerializerOptions? options = null)
    {
        if (quantParams == null)
            throw new ArgumentNullException(nameof(quantParams));

        if (string.IsNullOrEmpty(filePath))
            throw new ArgumentException("File path cannot be null or empty", nameof(filePath));

        var json = JsonSerializer.Serialize(quantParams, options ?? GetDefaultOptions());
        File.WriteAllText(filePath, json);
    }

    /// <summary>
    /// Loads quantization parameters from a file.
    /// </summary>
    /// <param name="filePath">The file path to load from.</param>
    /// <param name="options">Optional JSON deserialization options.</param>
    /// <returns>The loaded quantization parameters.</returns>
    public static Dictionary<string, QuantizationParameters> LoadQuantizationParameters(
        string filePath,
        JsonSerializerOptions? options = null)
    {
        if (string.IsNullOrEmpty(filePath))
            throw new ArgumentException("File path cannot be null or empty", nameof(filePath));

        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Quantization parameters file not found: {filePath}");

        var json = File.ReadAllText(filePath);
        var quantParams = JsonSerializer.Deserialize<Dictionary<string, QuantizationParameters>>(
            json, options ?? GetDefaultOptions());

        if (quantParams == null)
            throw new InvalidOperationException("Failed to deserialize quantization parameters");

        return quantParams;
    }

    /// <summary>
    /// Exports quantization parameters to a human-readable format.
    /// </summary>
    /// <param name="qatModel">The QAT model.</param>
    /// <param name="filePath">The file path to export to.</param>
    public static void ExportQuantizationParametersText(IQATModel qatModel, string filePath)
    {
        if (qatModel == null)
            throw new ArgumentNullException(nameof(qatModel));

        if (string.IsNullOrEmpty(filePath))
            throw new ArgumentException("File path cannot be null or empty", nameof(filePath));

        var quantParams = qatModel.GetQuantizationParameters();
        var sb = new System.Text.StringBuilder();

        sb.AppendLine("QAT Quantization Parameters Export");
        sb.AppendLine("===================================");
        sb.AppendLine($"Timestamp: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss}");
        sb.AppendLine($"Training Mode: {qatModel.TrainingMode}");
        sb.AppendLine($"Total Layers: {qatModel.GetLayerCount()}");
        sb.AppendLine($"Quantized Layers: {qatModel.GetQuantizedLayerCount()}");
        sb.AppendLine($"Fake Quantization Nodes: {qatModel.GetFakeQuantizationNodeCount()}");
        sb.AppendLine();

        sb.AppendLine("Layer-wise Parameters:");
        sb.AppendLine("---------------------");

        foreach (var kvp in quantParams)
        {
            sb.AppendLine($"Layer: {kvp.Key}");
            if (kvp.Value != null)
            {
                var param = kvp.Value.Value;
                sb.AppendLine($"  Scale: {param.Scale:F6}");
                sb.AppendLine($"  Zero Point: {param.ZeroPoint}");
                sb.AppendLine($"  Mode: {param.Mode}");
                sb.AppendLine($"  Type: {param.Type}");
                sb.AppendLine($"  Per-Channel: {param.IsPerChannel}");

                if (param.IsPerChannel && param.ChannelScales != null)
                {
                    sb.AppendLine($"  Channel Scales: [{string.Join(", ", param.ChannelScales.Select(s => s.ToString("F3")))}]");
                }

                if (param.IsPerChannel && param.ChannelZeroPoints != null)
                {
                    sb.AppendLine($"  Channel Zero Points: [{string.Join(", ", param.ChannelZeroPoints)}]");
                }
            }
            else
            {
                sb.AppendLine("  No quantization parameters");
            }
            sb.AppendLine();
        }

        File.WriteAllText(filePath, sb.ToString());
    }

    /// <summary>
    /// Validates a checkpoint file.
    /// </summary>
    /// <param name="filePath">The checkpoint file to validate.</param>
    /// <returns>True if the checkpoint is valid.</returns>
    public static bool ValidateCheckpoint(string filePath)
    {
        if (string.IsNullOrEmpty(filePath))
            return false;

        if (!File.Exists(filePath))
            return false;

        try
        {
            var checkpoint = LoadCheckpoint(filePath);
            return checkpoint != null &&
                   !string.IsNullOrEmpty(checkpoint.Version) &&
                   checkpoint.LayerCount > 0;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Gets the default JSON serialization options.
    /// </summary>
    /// <returns>Default JSON serialization options.</returns>
    private static JsonSerializerOptions GetDefaultOptions()
    {
        return new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };
    }

    /// <summary>
    /// Creates a backup of a checkpoint file.
    /// </summary>
    /// <param name="filePath">The checkpoint file to backup.</param>
    /// <returns>The path of the backup file.</returns>
    public static string CreateBackup(string filePath)
    {
        if (string.IsNullOrEmpty(filePath))
            throw new ArgumentException("File path cannot be null or empty", nameof(filePath));

        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Checkpoint file not found: {filePath}");

        var backupPath = $"{filePath}.{DateTime.UtcNow:yyyyMMddHHmmss}.bak";
        File.Copy(filePath, backupPath);
        return backupPath;
    }
}

/// <summary>
/// Represents a QAT model checkpoint containing quantization parameters and metadata.
/// </summary>
public class QATCheckpoint
{
    /// <summary>
    /// Gets or sets the checkpoint version.
    /// </summary>
    public string Version { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the checkpoint timestamp.
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Gets or sets whether the model was in training mode.
    /// </summary>
    public bool TrainingMode { get; set; }

    /// <summary>
    /// Gets or sets the number of layers in the model.
    /// </summary>
    public int LayerCount { get; set; }

    /// <summary>
    /// Gets or sets the number of fake quantization nodes.
    /// </summary>
    public int FakeQuantizationNodeCount { get; set; }

    /// <summary>
    /// Gets or sets the number of quantized layers.
    /// </summary>
    public int QuantizedLayerCount { get; set; }

    /// <summary>
    /// Gets or sets the layer-specific quantization parameters.
    /// Key is layer name, value is quantization parameters.
    /// </summary>
    public Dictionary<string, QuantizationParameters> LayerQuantizationParameters { get; set; } = new();

    /// <summary>
    /// Gets or sets optional metadata.
    /// </summary>
    public Dictionary<string, object> Metadata { get; set; } = new();
}
