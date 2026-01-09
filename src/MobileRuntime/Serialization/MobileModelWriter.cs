using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Text;
using MobileRuntime.Serialization.Models;

namespace MobileRuntime.Serialization
{
    /// <summary>
    /// Writer for the mobile model binary format
    /// </summary>
    public sealed class MobileModelWriter
    {
        private BinaryWriter _writer;
        private MemoryStream _stream;
        private readonly List<long> _offsets;

        public MobileModelWriter()
        {
            _stream = new MemoryStream();
            _writer = new BinaryWriter(_stream, Encoding.UTF8, leaveOpen: false);
            _offsets = new List<long>();
        }

        public byte[] WriteModel(ModelWriterConfig config)
        {
            // Write placeholder header (will be updated later)
            WritePlaceholderHeader(config);

            // Write metadata
            _header.ModelMetadataOffset = (uint)_stream.Position;
            WriteMetadata(config);

            // Write input/output specs
            WriteInputOutputSpecs(config.Inputs);
            WriteInputOutputSpecs(config.Outputs);

            // Write constant tensors
            _header.TensorDataOffset = (uint)_stream.Position;
            WriteConstantTensors(config.ConstantTensors);

            // Write operator graph
            _header.OperatorGraphOffset = (uint)_stream.Position;
            WriteOperatorGraph(config.Operators);

            // Update header with final values
            UpdateHeader(config);

            // Calculate and write checksum
            WriteChecksum();

            return _stream.ToArray();
        }

        private ModelHeader _header;

        private void WritePlaceholderHeader(ModelWriterConfig config)
        {
            _header = new ModelHeader
            {
                MagicNumber = ModelHeader.MAGIC_NUMBER,
                Version = ModelHeader.CURRENT_VERSION,
                HeaderChecksum = 0, // Will be calculated
                Flags = config.Flags,
                TotalFileSize = 0, // Will be updated
                ModelMetadataOffset = 0, // Will be updated
                TensorDataOffset = 0, // Will be updated
                OperatorGraphOffset = 0 // Will be updated
            };

            _writer.Write(_header.MagicNumber);
            _writer.Write(_header.Version);
            _writer.Write(_header.HeaderChecksum);
            _writer.Write(_header.Flags);
            _writer.Write(_header.TotalFileSize);
            _writer.Write(_header.ModelMetadataOffset);
            _writer.Write(_header.TensorDataOffset);
            _writer.Write(_header.OperatorGraphOffset);
        }

        private void UpdateHeader(ModelWriterConfig config)
        {
            _header.TotalFileSize = (uint)_stream.Length;

            // Write header at the beginning
            var currentPos = _stream.Position;
            _stream.Seek(0, SeekOrigin.Begin);

            _writer.Write(_header.MagicNumber);
            _writer.Write(_header.Version);
            _writer.Write(_header.HeaderChecksum);
            _writer.Write(_header.Flags);
            _writer.Write(_header.TotalFileSize);
            _writer.Write(_header.ModelMetadataOffset);
            _writer.Write(_header.TensorDataOffset);
            _writer.Write(_header.OperatorGraphOffset);

            _stream.Seek(currentPos, SeekOrigin.Begin);
        }

        private void WriteMetadata(ModelWriterConfig config)
        {
            var nameBytes = Encoding.UTF8.GetBytes(config.ModelName);
            _writer.Write((uint)nameBytes.Length);
            _writer.Write(nameBytes);
            _writer.Write(config.FrameworkVersion);
            _writer.Write((ulong)DateTimeOffset.UtcNow.ToUnixTimeSeconds());
            _writer.Write((uint)config.Inputs.Length);
            _writer.Write((uint)config.Outputs.Length);
        }

        private void WriteInputOutputSpecs(InputOutputSpec[] specs)
        {
            foreach (var spec in specs)
            {
                WriteInputOutputSpec(spec);
            }
        }

        private void WriteInputOutputSpec(InputOutputSpec spec)
        {
            var nameBytes = Encoding.UTF8.GetBytes(spec.Name);
            _writer.Write((uint)nameBytes.Length);
            _writer.Write(nameBytes);
            _writer.Write(spec.Rank);
            _writer.Write((ushort)spec.DataType);
            foreach (var dim in spec.Shape)
            {
                _writer.Write(dim);
            }
        }

        private void WriteConstantTensors(ConstantTensor[] tensors)
        {
            _writer.Write((uint)tensors.Length);

            foreach (var tensor in tensors)
            {
                WriteConstantTensor(tensor);
            }
        }

        private void WriteConstantTensor(ConstantTensor tensor)
        {
            _writer.Write(tensor.Id);
            _writer.Write((ushort)tensor.DataType);
            _writer.Write(tensor.Rank);

            foreach (var dim in tensor.Shape)
            {
                _writer.Write(dim);
            }

            // Compress data if enabled
            byte[] dataToWrite = tensor.Data;
            bool compressed = false;

            if (tensor.Data != null && tensor.Data.Length > 0)
            {
                try
                {
                    using (var input = new MemoryStream(tensor.Data))
                    using (var output = new MemoryStream())
                    {
                        using (var gzip = new GZipStream(output, CompressionLevel.Optimal))
                        {
                            input.CopyTo(gzip);
                        }
                        var compressedData = output.ToArray();

                        // Only use compression if it reduces size
                        if (compressedData.Length < tensor.Data.Length)
                        {
                            dataToWrite = compressedData;
                            compressed = true;
                        }
                    }
                }
                catch
                {
                    // Compression failed, use original data
                }
            }

            tensor.DataSize = (ulong)dataToWrite.Length;
            _writer.Write(tensor.DataSize);
            _writer.Write(dataToWrite);
        }

        private void WriteOperatorGraph(OperatorDescriptor[] operators)
        {
            _writer.Write((uint)operators.Length);

            foreach (var op in operators)
            {
                WriteOperator(op);
            }
        }

        private void WriteOperator(OperatorDescriptor op)
        {
            _writer.Write((ushort)op.Type);
            _writer.Write((ushort)op.InputTensorIds.Length);
            _writer.Write((ushort)op.OutputTensorIds.Length);
            _writer.Write((ushort)op.Parameters.Count);

            foreach (var tensorId in op.InputTensorIds)
            {
                _writer.Write(tensorId);
            }

            foreach (var tensorId in op.OutputTensorIds)
            {
                _writer.Write(tensorId);
            }

            foreach (var kvp in op.Parameters)
            {
                WriteParameter(kvp.Key, kvp.Value);
            }
        }

        private void WriteParameter(string name, object value)
        {
            var nameBytes = Encoding.UTF8.GetBytes(name);
            _writer.Write((ushort)nameBytes.Length);
            _writer.Write(nameBytes);

            if (value is int intVal)
            {
                _writer.Write((byte)ParameterType.Int32);
                _writer.Write(intVal);
            }
            else if (value is long longVal)
            {
                _writer.Write((byte)ParameterType.Int64);
                _writer.Write(longVal);
            }
            else if (value is float floatVal)
            {
                _writer.Write((byte)ParameterType.Float32);
                _writer.Write(floatVal);
            }
            else if (value is double doubleVal)
            {
                _writer.Write((byte)ParameterType.Float64);
                _writer.Write(doubleVal);
            }
            else if (value is bool boolVal)
            {
                _writer.Write((byte)ParameterType.Bool);
                _writer.Write((byte)(boolVal ? 1 : 0));
            }
            else if (value is int[] intArray)
            {
                _writer.Write((byte)ParameterType.Int32Array);
                _writer.Write(intArray.Length);
                foreach (var val in intArray)
                {
                    _writer.Write(val);
                }
            }
            else if (value is float[] floatArray)
            {
                _writer.Write((byte)ParameterType.Float32Array);
                _writer.Write(floatArray.Length);
                foreach (var val in floatArray)
                {
                    _writer.Write(val);
                }
            }
            else
            {
                throw new NotSupportedException($"Parameter type {value.GetType()} is not supported");
            }
        }

        private uint CalculateChecksum()
        {
            // Calculate Adler32 checksum for entire file
            var buffer = _stream.ToArray();
            return CalculateAdler32(buffer, buffer.Length - 4);
        }

        private void WriteChecksum()
        {
            var checksum = CalculateChecksum();
            _writer.Write(checksum);
        }

        private uint CalculateAdler32(byte[] data, int length)
        {
            const uint ADLER_MOD = 65521;
            uint a = 1, b = 0;

            for (int i = 0; i < length; i++)
            {
                a = (a + data[i]) % ADLER_MOD;
                b = (b + a) % ADLER_MOD;
            }

            return (b << 16) | a;
        }

        private enum ParameterType : byte
        {
            Int32,
            Int64,
            Float32,
            Float64,
            Bool,
            Int32Array,
            Float32Array
        }
    }
}
