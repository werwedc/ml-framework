using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using RitterFramework.Core.LoRA;
using RitterFramework.Core.Tensor;

namespace MLFramework.LoRA
{
    /// <summary>
    /// Serializer for saving and loading LoRA adapters to/from disk.
    /// Supports both binary (efficient) and JSON (interoperable) formats.
    /// </summary>
    public static class AdapterSerializer
    {
        /// <summary>
        /// Save adapter to binary file.
        /// </summary>
        public static void Save(LoraAdapter adapter, string path)
        {
            if (adapter == null)
                throw new ArgumentNullException(nameof(adapter));
            if (string.IsNullOrEmpty(path))
                throw new ArgumentException("Path cannot be null or empty", nameof(path));

            var directory = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            using var stream = File.Create(path);
            using var writer = new BinaryWriter(stream);

            WriteHeader(writer, adapter);
            WriteConfig(writer, adapter.Config);
            WriteWeights(writer, adapter.Weights);
            WriteMetadata(writer, adapter.Metadata);
        }

        /// <summary>
        /// Load adapter from binary file.
        /// </summary>
        public static LoraAdapter Load(string path)
        {
            if (string.IsNullOrEmpty(path))
                throw new ArgumentException("Path cannot be null or empty", nameof(path));
            if (!File.Exists(path))
                throw new FileNotFoundException($"Adapter file not found: {path}");

            using var stream = File.OpenRead(path);
            using var reader = new BinaryReader(stream);

            var header = ReadHeader(reader);
            var config = ReadConfig(reader);
            var weights = ReadWeights(reader, config);
            var metadata = ReadMetadata(reader);

            var adapter = new LoraAdapter(header.Name, config)
            {
                Weights = weights,
                Metadata = metadata
            };

            return adapter;
        }

        /// <summary>
        /// Save adapter to JSON (for inspection/compatibility).
        /// </summary>
        public static void SaveJson(LoraAdapter adapter, string path)
        {
            if (adapter == null)
                throw new ArgumentNullException(nameof(adapter));
            if (string.IsNullOrEmpty(path))
                throw new ArgumentException("Path cannot be null or empty", nameof(path));

            var directory = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var options = new JsonSerializerOptions
            {
                WriteIndented = true,
                Converters = { new TensorJsonConverter() }
            };

            var json = JsonSerializer.Serialize(adapter, options);
            File.WriteAllText(path, json);
        }

        /// <summary>
        /// Load adapter from JSON.
        /// </summary>
        public static LoraAdapter LoadJson(string path)
        {
            if (string.IsNullOrEmpty(path))
                throw new ArgumentException("Path cannot be null or empty", nameof(path));
            if (!File.Exists(path))
                throw new FileNotFoundException($"Adapter file not found: {path}");

            var json = File.ReadAllText(path);
            var options = new JsonSerializerOptions
            {
                Converters = { new TensorJsonConverter() }
            };

            return JsonSerializer.Deserialize<LoraAdapter>(json, options);
        }

        private static void WriteHeader(BinaryWriter writer, LoraAdapter adapter)
        {
            // Magic bytes for format identification (LoRA)
            writer.Write(new byte[] { 0x4C, 0x6F, 0x52, 0x41 });

            // Version
            writer.Write((short)1);

            // Adapter name length and name
            var nameBytes = Encoding.UTF8.GetBytes(adapter.Name);
            writer.Write(nameBytes.Length);
            writer.Write(nameBytes);
        }

        private static AdapterHeader ReadHeader(BinaryReader reader)
        {
            // Verify magic bytes
            var magic = reader.ReadBytes(4);
            if (!magic.SequenceEqual(new byte[] { 0x4C, 0x6F, 0x52, 0x41 }))
            {
                throw new InvalidDataException("Invalid LoRA adapter file format: missing magic bytes");
            }

            // Read version
            var version = reader.ReadInt16();
            if (version != 1)
            {
                throw new InvalidDataException($"Unsupported LoRA adapter file version: {version}. Only version 1 is supported.");
            }

            // Read name
            var nameLength = reader.ReadInt32();
            var nameBytes = reader.ReadBytes(nameLength);
            var name = Encoding.UTF8.GetString(nameBytes);

            return new AdapterHeader { Name = name, Version = version };
        }

        private static void WriteConfig(BinaryWriter writer, LoraConfig config)
        {
            writer.Write(config.Rank);
            writer.Write(config.Alpha);
            writer.Write(config.Dropout);
            writer.Write(config.Bias ?? "none");
            writer.Write(config.LoraType ?? "default");

            // Write target modules
            writer.Write(config.TargetModules.Length);
            foreach (var module in config.TargetModules)
            {
                var bytes = Encoding.UTF8.GetBytes(module);
                writer.Write(bytes.Length);
                writer.Write(bytes);
            }
        }

        private static LoraConfig ReadConfig(BinaryReader reader)
        {
            var config = new LoraConfig
            {
                Rank = reader.ReadInt32(),
                Alpha = reader.ReadInt32(),
                Dropout = reader.ReadSingle(),
                Bias = reader.ReadString(),
                LoraType = reader.ReadString()
            };

            var targetModuleCount = reader.ReadInt32();
            config.TargetModules = new string[targetModuleCount];
            for (int i = 0; i < targetModuleCount; i++)
            {
                var length = reader.ReadInt32();
                config.TargetModules[i] = Encoding.UTF8.GetString(reader.ReadBytes(length));
            }

            return config;
        }

        private static void WriteWeights(BinaryWriter writer, Dictionary<string, LoraModuleWeights> weights)
        {
            writer.Write(weights.Count);

            foreach (var kvp in weights)
            {
                // Module name
                var nameBytes = Encoding.UTF8.GetBytes(kvp.Key);
                writer.Write(nameBytes.Length);
                writer.Write(nameBytes);

                // LoraA tensor
                WriteTensor(writer, kvp.Value.LoraA);

                // LoraB tensor
                WriteTensor(writer, kvp.Value.LoraB);
            }
        }

        private static Dictionary<string, LoraModuleWeights> ReadWeights(BinaryReader reader, LoraConfig config)
        {
            var weightCount = reader.ReadInt32();
            var weights = new Dictionary<string, LoraModuleWeights>();

            for (int i = 0; i < weightCount; i++)
            {
                var nameLength = reader.ReadInt32();
                var name = Encoding.UTF8.GetString(reader.ReadBytes(nameLength));

                var loraA = ReadTensor(reader);
                var loraB = ReadTensor(reader);

                weights[name] = new LoraModuleWeights { LoraA = loraA, LoraB = loraB };
            }

            return weights;
        }

        private static void WriteTensor(BinaryWriter writer, Tensor tensor)
        {
            writer.Write(tensor.Dimensions);
            for (int i = 0; i < tensor.Dimensions; i++)
            {
                writer.Write(tensor.Shape[i]);
            }

            writer.Write(tensor.NumElements());

            // Get raw float array
            var data = tensor.Data;
            var bytes = new byte[data.Length * sizeof(float)];
            Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
            writer.Write(bytes);
        }

        private static Tensor ReadTensor(BinaryReader reader)
        {
            var rank = reader.ReadInt32();
            var shape = new int[rank];
            for (int i = 0; i < rank; i++)
            {
                shape[i] = reader.ReadInt32();
            }

            var numElements = reader.ReadInt32();
            var numBytes = numElements * sizeof(float);
            var bytes = reader.ReadBytes(numBytes);
            var data = new float[numElements];
            Buffer.BlockCopy(bytes, 0, data, 0, numBytes);

            return new Tensor(data, shape);
        }

        private static void WriteMetadata(BinaryWriter writer, AdapterMetadata metadata)
        {
            writer.Write(metadata.CreatedAt.ToBinary());
            writer.Write(metadata.UpdatedAt?.ToBinary() ?? 0);
            writer.Write(metadata.BaseModel ?? "");
            writer.Write(metadata.TrainingEpochs ?? -1);
            writer.Write(metadata.FinalLoss ?? float.NaN);

            // Custom fields
            writer.Write(metadata.CustomFields.Count);
            foreach (var kvp in metadata.CustomFields)
            {
                var keyBytes = Encoding.UTF8.GetBytes(kvp.Key);
                var valueBytes = Encoding.UTF8.GetBytes(kvp.Value);
                writer.Write(keyBytes.Length);
                writer.Write(keyBytes);
                writer.Write(valueBytes.Length);
                writer.Write(valueBytes);
            }
        }

        private static AdapterMetadata ReadMetadata(BinaryReader reader)
        {
            var metadata = new AdapterMetadata();
            metadata.CreatedAt = DateTime.FromBinary(reader.ReadInt64());
            var updatedAt = reader.ReadInt64();
            metadata.UpdatedAt = updatedAt != 0 ? DateTime.FromBinary(updatedAt) : null;
            metadata.BaseModel = reader.ReadString();
            var epochs = reader.ReadInt32();
            metadata.TrainingEpochs = epochs != -1 ? epochs : null;
            var finalLoss = reader.ReadSingle();
            metadata.FinalLoss = !float.IsNaN(finalLoss) ? finalLoss : null;

            var fieldCount = reader.ReadInt32();
            for (int i = 0; i < fieldCount; i++)
            {
                var keyLength = reader.ReadInt32();
                var key = Encoding.UTF8.GetString(reader.ReadBytes(keyLength));
                var valueLength = reader.ReadInt32();
                var value = Encoding.UTF8.GetString(reader.ReadBytes(valueLength));
                metadata.CustomFields[key] = value;
            }

            return metadata;
        }
    }

    /// <summary>
    /// Header information for a serialized LoRA adapter.
    /// </summary>
    public class AdapterHeader
    {
        /// <summary>Adapter name</summary>
        public string Name { get; set; }

        /// <summary>File format version</summary>
        public short Version { get; set; }
    }

    /// <summary>
    /// JSON converter for Tensor objects.
    /// </summary>
    public class TensorJsonConverter : JsonConverter<Tensor>
    {
        public override Tensor Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType != JsonTokenType.StartObject)
                throw new JsonException();

            int[] shape = null;
            float[] data = null;

            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndObject)
                    break;

                if (reader.TokenType == JsonTokenType.PropertyName)
                {
                    var propertyName = reader.GetString();
                    reader.Read();

                    if (propertyName == "shape")
                    {
                        var shapeList = JsonSerializer.Deserialize<List<int>>(ref reader, options);
                        shape = shapeList.ToArray();
                    }
                    else if (propertyName == "data")
                    {
                        var dataList = JsonSerializer.Deserialize<List<float>>(ref reader, options);
                        data = dataList.ToArray();
                    }
                }
            }

            if (shape == null || data == null)
                throw new JsonException("Tensor must have both 'shape' and 'data' properties");

            return new Tensor(data, shape);
        }

        public override void Write(Utf8JsonWriter writer, Tensor value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();

            writer.WritePropertyName("shape");
            JsonSerializer.Serialize(writer, value.Shape.ToList(), options);

            writer.WritePropertyName("data");
            var dataArray = value.Data;
            JsonSerializer.Serialize(writer, dataArray, options);

            writer.WriteEndObject();
        }
    }
}
