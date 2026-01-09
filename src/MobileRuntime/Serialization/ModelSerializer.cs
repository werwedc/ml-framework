using System;
using System.IO;
using System.Linq;
using MobileRuntime.Serialization.Models;

namespace MobileRuntime.Serialization
{
    /// <summary>
    /// Utility class for model serialization/deserialization
    /// </summary>
    public static class ModelSerializer
    {
        /// <summary>
        /// Load a model from a file path
        /// </summary>
        public static MobileModelFormat Load(string filePath)
        {
            var model = new MobileModelFormat(filePath);
            model.Read();
            return model;
        }

        /// <summary>
        /// Load a model from a byte array
        /// </summary>
        public static MobileModelFormat Load(byte[] data)
        {
            var model = new MobileModelFormat(data);
            model.Read();
            return model;
        }

        /// <summary>
        /// Load a model from a stream
        /// </summary>
        public static MobileModelFormat Load(Stream stream)
        {
            var model = new MobileModelFormat(stream);
            model.Read();
            return model;
        }

        /// <summary>
        /// Save a model configuration to a byte array
        /// </summary>
        public static byte[] Save(ModelWriterConfig config)
        {
            var writer = new MobileModelWriter();
            return writer.WriteModel(config);
        }

        /// <summary>
        /// Save a model configuration to a file
        /// </summary>
        public static void Save(string filePath, ModelWriterConfig config)
        {
            var data = Save(config);
            File.WriteAllBytes(filePath, data);
        }

        /// <summary>
        /// Validate a model file
        /// </summary>
        public static bool Validate(string filePath)
        {
            try
            {
                using (var stream = File.OpenRead(filePath))
                {
                    return ValidateStream(stream);
                }
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Validate a model byte array
        /// </summary>
        public static bool Validate(byte[] data)
        {
            try
            {
                using (var stream = new MemoryStream(data))
                {
                    return ValidateStream(stream);
                }
            }
            catch
            {
                return false;
            }
        }

        private static bool ValidateStream(Stream stream)
        {
            var reader = new BinaryReader(stream);

            // Check magic number
            var magic = reader.ReadUInt32();
            if (magic != ModelHeader.MAGIC_NUMBER)
            {
                return false;
            }

            // Check version
            var version = reader.ReadUInt16();
            if (version > ModelHeader.CURRENT_VERSION)
            {
                return false;
            }

            // Validate header checksum
            // Note: In a real implementation, we would calculate and verify the checksum

            return true;
        }

        /// <summary>
        /// Create an InputOutputSpec helper
        /// </summary>
        public static InputOutputSpec CreateSpec(string name, DataType dataType, params int[] shape)
        {
            return new InputOutputSpec
            {
                Name = name,
                Rank = (ushort)shape.Length,
                DataType = dataType,
                Shape = shape.Select(s => (ulong)s).ToArray()
            };
        }

        /// <summary>
        /// Create a constant tensor from a runtime Tensor
        /// </summary>
        public static ConstantTensor CreateTensor(uint id, Tensor tensor)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            byte[] data;
            switch (tensor.DataType)
            {
                case DataType.Float32:
                    {
                        var floatData = tensor.ToArray<float>();
                        data = new byte[floatData.Length * 4];
                        System.Buffer.BlockCopy(floatData, 0, data, 0, data.Length);
                        break;
                    }
                case DataType.Int32:
                    {
                        var intData = tensor.ToArray<int>();
                        data = new byte[intData.Length * 4];
                        System.Buffer.BlockCopy(intData, 0, data, 0, data.Length);
                        break;
                    }
                case DataType.Int8:
                    {
                        var sbyteData = tensor.ToArray<sbyte>();
                        data = new byte[sbyteData.Length];
                        for (int i = 0; i < sbyteData.Length; i++)
                        {
                            data[i] = (byte)sbyteData[i];
                        }
                        break;
                    }
                case DataType.Int16:
                    {
                        var shortData = tensor.ToArray<short>();
                        data = new byte[shortData.Length * 2];
                        System.Buffer.BlockCopy(shortData, 0, data, 0, data.Length);
                        break;
                    }
                case DataType.Float16:
                    {
                        // Float16 support not yet implemented
                        data = tensor.ToArray<byte>();
                        break;
                    }
                default:
                    throw new NotSupportedException($"Data type {tensor.DataType} is not supported for serialization");
            }

            return new ConstantTensor
            {
                Id = id,
                DataType = tensor.DataType,
                Rank = (ushort)tensor.Shape.Length,
                Shape = tensor.Shape.Select(s => (ulong)s).ToArray(),
                DataSize = (ulong)data.Length,
                Data = data
            };
        }

        /// <summary>
        /// Create a constant tensor from raw data
        /// </summary>
        public static ConstantTensor CreateTensor(uint id, int[] shape, DataType dataType, byte[] data)
        {
            return new ConstantTensor
            {
                Id = id,
                DataType = dataType,
                Rank = (ushort)shape.Length,
                Shape = shape.Select(s => (ulong)s).ToArray(),
                DataSize = (ulong)data.Length,
                Data = data
            };
        }
    }
}
