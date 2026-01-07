using System.Text.Json;

namespace RitterFramework.Core.LoRA
{
    /// <summary>
    /// Serializer for LoRA adapters (binary and JSON formats).
    /// </summary>
    public static class AdapterSerializer
    {
        private const int MagicNumber = 0x4C4F5241; // "LORA" in hex
        private const int CurrentVersion = 1;

        /// <summary>
        /// Save adapter to binary file format.
        /// </summary>
        public static void Save(LoraAdapter adapter, string path)
        {
            if (adapter == null) throw new ArgumentNullException(nameof(adapter));
            if (string.IsNullOrEmpty(path)) throw new ArgumentException("Path cannot be null or empty", nameof(path));

            var directory = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            using var stream = new FileStream(path, FileMode.Create);
            using var writer = new BinaryWriter(stream);

            // Write header
            writer.Write(MagicNumber);
            writer.Write(CurrentVersion);

            // Write metadata
            writer.Write(adapter.Name);
            writer.Write(adapter.Metadata.CreatedAt.ToBinary());
            writer.Write(adapter.Metadata.Creator ?? "");
            writer.Write(adapter.Metadata.Description ?? "");
            writer.Write(adapter.Metadata.Version ?? "1.0");

            // Write config
            writer.Write(adapter.Config.Rank);
            writer.Write(adapter.Config.Alpha);
            writer.Write(adapter.Config.Dropout);
            writer.Write(adapter.Config.Bias ?? "none");
            writer.Write(adapter.Config.LoraType ?? "default");

            var targetModules = adapter.Config.TargetModules ?? Array.Empty<string>();
            writer.Write(targetModules.Length);
            foreach (var module in targetModules)
            {
                writer.Write(module ?? "");
            }

            // Write weights
            writer.Write(adapter.Weights.Count);
            foreach (var kvp in adapter.Weights)
            {
                writer.Write(kvp.Key);
                WriteTensor(writer, kvp.Value.LoraA);
                WriteTensor(writer, kvp.Value.LoraB);
            }
        }

        /// <summary>
        /// Load adapter from binary file format.
        /// </summary>
        public static LoraAdapter Load(string path)
        {
            if (string.IsNullOrEmpty(path)) throw new ArgumentException("Path cannot be null or empty", nameof(path));
            if (!File.Exists(path)) throw new FileNotFoundException($"Adapter file not found: {path}", path);

            using var stream = new FileStream(path, FileMode.Open);
            using var reader = new BinaryReader(stream);

            // Read and verify header
            var magic = reader.ReadInt32();
            if (magic != MagicNumber)
            {
                throw new InvalidDataException($"Invalid file format. Expected magic number 0x{MagicNumber:X}, got 0x{magic:X}");
            }

            var version = reader.ReadInt32();
            if (version != CurrentVersion)
            {
                throw new InvalidDataException($"Unsupported version: {version}. Current version: {CurrentVersion}");
            }

            // Create adapter
            var adapter = new LoraAdapter();

            // Read metadata
            adapter.Name = reader.ReadString();
            adapter.Metadata.CreatedAt = DateTime.FromBinary(reader.ReadInt64());
            adapter.Metadata.Creator = reader.ReadString();
            adapter.Metadata.Description = reader.ReadString();
            adapter.Metadata.Version = reader.ReadString();

            // Read config
            var rank = reader.ReadInt32();
            var alpha = reader.ReadInt32();
            var dropout = reader.ReadSingle();
            var bias = reader.ReadString();
            var loraType = reader.ReadString();

            var targetModuleCount = reader.ReadInt32();
            var targetModules = new string[targetModuleCount];
            for (int i = 0; i < targetModuleCount; i++)
            {
                targetModules[i] = reader.ReadString();
            }

            adapter.Config = new LoraConfig(rank, alpha, dropout, targetModules, bias, loraType);

            // Read weights
            var weightCount = reader.ReadInt32();
            for (int i = 0; i < weightCount; i++)
            {
                var moduleName = reader.ReadString();
                var loraA = ReadTensor(reader);
                var loraB = ReadTensor(reader);
                adapter.Weights[moduleName] = new LoraModuleWeights(loraA, loraB);
            }

            return adapter;
        }

        /// <summary>
        /// Save adapter to JSON format.
        /// </summary>
        public static void SaveJson(LoraAdapter adapter, string path)
        {
            if (adapter == null) throw new ArgumentNullException(nameof(adapter));
            if (string.IsNullOrEmpty(path)) throw new ArgumentException("Path cannot be null or empty", nameof(path));

            var directory = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var options = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            };

            var json = JsonSerializer.Serialize(adapter, options);
            File.WriteAllText(path, json);
        }

        /// <summary>
        /// Load adapter from JSON format.
        /// </summary>
        public static LoraAdapter LoadJson(string path)
        {
            if (string.IsNullOrEmpty(path)) throw new ArgumentException("Path cannot be null or empty", nameof(path));
            if (!File.Exists(path)) throw new FileNotFoundException($"Adapter file not found: {path}", path);

            var json = File.ReadAllText(path);
            var options = new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            };

            var adapter = JsonSerializer.Deserialize<LoraAdapter>(json, options);
            if (adapter == null)
            {
                throw new InvalidDataException("Failed to deserialize adapter from JSON");
            }

            return adapter;
        }

        private static void WriteTensor(BinaryWriter writer, Core.Tensor.Tensor tensor)
        {
            if (tensor == null) throw new ArgumentNullException(nameof(tensor));

            var shape = tensor.Shape;
            writer.Write(shape.Length);
            foreach (var dim in shape)
            {
                writer.Write(dim);
            }

            var data = tensor.Data;
            writer.Write(data.Length);
            foreach (var value in data)
            {
                writer.Write(value);
            }
        }

        private static Core.Tensor.Tensor ReadTensor(BinaryReader reader)
        {
            var dimCount = reader.ReadInt32();
            var shape = new int[dimCount];
            for (int i = 0; i < dimCount; i++)
            {
                shape[i] = reader.ReadInt32();
            }

            var dataLength = reader.ReadInt32();
            var data = new float[dataLength];
            for (int i = 0; i < dataLength; i++)
            {
                data[i] = reader.ReadSingle();
            }

            return new Core.Tensor.Tensor(data, shape);
        }
    }
}
