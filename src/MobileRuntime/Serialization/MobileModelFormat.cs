using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using MobileRuntime.Serialization.Models;

namespace MobileRuntime.Serialization
{
    /// <summary>
    /// Reader for the mobile model binary format
    /// </summary>
    public sealed class MobileModelFormat : IDisposable
    {
        private BinaryReader _reader;
        private Stream _stream;
        private ModelHeader _header;
        private ModelMetadata _metadata;
        private List<InputOutputSpec> _inputs;
        private List<InputOutputSpec> _outputs;
        private Dictionary<uint, ConstantTensor> _constantTensors;
        private List<OperatorDescriptor> _operators;

        public MobileModelFormat(string filePath)
        {
            _stream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
            _reader = new BinaryReader(_stream, Encoding.UTF8, leaveOpen: false);
        }

        public MobileModelFormat(Stream stream)
        {
            _stream = stream;
            _reader = new BinaryReader(_stream, Encoding.UTF8, leaveOpen: false);
        }

        public MobileModelFormat(byte[] data)
        {
            _stream = new MemoryStream(data);
            _reader = new BinaryReader(_stream, Encoding.UTF8, leaveOpen: false);
        }

        public void Read()
        {
            ReadHeader();
            ValidateHeader();
            ReadMetadata();
            ReadInputOutputSpecs();
            ReadConstantTensors();
            ReadOperatorGraph();
            ValidateChecksum();
        }

        public ModelHeader Header => _header;
        public ModelMetadata Metadata => _metadata;
        public InputOutputSpec[] Inputs => _inputs.ToArray();
        public InputOutputSpec[] Outputs => _outputs.ToArray();
        public ConstantTensor[] ConstantTensors => _constantTensors.Values.ToArray();
        public OperatorDescriptor[] Operators => _operators.ToArray();

        public void Dispose()
        {
            _reader?.Dispose();
            _stream?.Dispose();
        }

        private void ValidateHeader()
        {
            if (_header.MagicNumber != ModelHeader.MAGIC_NUMBER)
            {
                throw new InvalidDataException($"Invalid magic number: 0x{_header.MagicNumber:X8}");
            }

            if (_header.Version > ModelHeader.CURRENT_VERSION)
            {
                throw new InvalidDataException(
                    $"Unsupported version: {_header.Version}. Supported: {ModelHeader.CURRENT_VERSION}"
                );
            }

            if (_header.TotalFileSize != _stream.Length)
            {
                throw new InvalidDataException(
                    $"File size mismatch: header={_header.TotalFileSize}, actual={_stream.Length}"
                );
            }
        }

        private void ReadHeader()
        {
            _stream.Seek(0, SeekOrigin.Begin);
            _header = new ModelHeader
            {
                MagicNumber = _reader.ReadUInt32(),
                Version = _reader.ReadUInt16(),
                HeaderChecksum = _reader.ReadUInt32(),
                Flags = _reader.ReadUInt32(),
                TotalFileSize = _reader.ReadUInt32(),
                ModelMetadataOffset = _reader.ReadUInt32(),
                TensorDataOffset = _reader.ReadUInt32(),
                OperatorGraphOffset = _reader.ReadUInt32()
            };
        }

        private void ReadMetadata()
        {
            _stream.Seek(_header.ModelMetadataOffset, SeekOrigin.Begin);

            var nameLength = _reader.ReadUInt32();
            var nameBytes = _reader.ReadBytes((int)nameLength);
            var name = Encoding.UTF8.GetString(nameBytes);

            _metadata = new ModelMetadata
            {
                Name = name,
                FrameworkVersion = _reader.ReadUInt32(),
                CreationTimestamp = _reader.ReadUInt64(),
                InputCount = _reader.ReadUInt32(),
                OutputCount = _reader.ReadUInt32()
            };
        }

        private void ReadInputOutputSpecs()
        {
            _inputs = new List<InputOutputSpec>();
            _outputs = new List<InputOutputSpec>();

            // Read inputs
            for (int i = 0; i < _metadata.InputCount; i++)
            {
                _inputs.Add(ReadInputOutputSpec());
            }

            // Read outputs
            for (int i = 0; i < _metadata.OutputCount; i++)
            {
                _outputs.Add(ReadInputOutputSpec());
            }
        }

        private InputOutputSpec ReadInputOutputSpec()
        {
            var nameLength = _reader.ReadUInt32();
            var nameBytes = _reader.ReadBytes((int)nameLength);
            var name = Encoding.UTF8.GetString(nameBytes);
            var rank = _reader.ReadUInt16();
            var dataType = (DataType)_reader.ReadUInt16();
            var shape = new ulong[rank];

            for (int i = 0; i < rank; i++)
            {
                shape[i] = _reader.ReadUInt64();
            }

            return new InputOutputSpec
            {
                Name = name,
                Rank = rank,
                DataType = dataType,
                Shape = shape
            };
        }

        private void ReadConstantTensors()
        {
            _constantTensors = new Dictionary<uint, ConstantTensor>();

            _stream.Seek(_header.TensorDataOffset, SeekOrigin.Begin);
            var tensorCount = _reader.ReadUInt32();

            for (int i = 0; i < tensorCount; i++)
            {
                var rank = _reader.ReadUInt16();
                var tensor = new ConstantTensor
                {
                    Id = _reader.ReadUInt32(),
                    DataType = (DataType)_reader.ReadUInt16(),
                    Rank = rank,
                    Shape = new ulong[rank]
                };

                // Read shape
                tensor.Shape = new ulong[tensor.Rank];
                for (int j = 0; j < tensor.Rank; j++)
                {
                    tensor.Shape[j] = _reader.ReadUInt64();
                }

                // Read data size and data
                tensor.DataSize = _reader.ReadUInt64();
                tensor.Data = _reader.ReadBytes((int)tensor.DataSize);

                // Apply decompression if needed
                if ((tensor.DataSize > 0 && tensor.Data[0] == 0x1F && tensor.Data[1] == 0x8B))
                {
                    // GZip magic number detected
                    try
                    {
                        using (var compressed = new MemoryStream(tensor.Data))
                        using (var gzip = new GZipStream(compressed, CompressionMode.Decompress))
                        using (var decompressed = new MemoryStream())
                        {
                            gzip.CopyTo(decompressed);
                            tensor.Data = decompressed.ToArray();
                        }
                    }
                    catch
                    {
                        // If decompression fails, keep original data
                    }
                }

                _constantTensors[tensor.Id] = tensor;
            }
        }

        private void ReadOperatorGraph()
        {
            _operators = new List<OperatorDescriptor>();

            _stream.Seek(_header.OperatorGraphOffset, SeekOrigin.Begin);
            var operatorCount = _reader.ReadUInt32();

            for (int i = 0; i < operatorCount; i++)
            {
                var op = new OperatorDescriptor
                {
                    Type = (OperatorType)_reader.ReadUInt16()
                };

                var inputCount = _reader.ReadUInt16();
                var outputCount = _reader.ReadUInt16();
                var paramCount = _reader.ReadUInt16();

                op.InputTensorIds = new uint[inputCount];
                for (int j = 0; j < inputCount; j++)
                {
                    op.InputTensorIds[j] = _reader.ReadUInt32();
                }

                op.OutputTensorIds = new uint[outputCount];
                for (int j = 0; j < outputCount; j++)
                {
                    op.OutputTensorIds[j] = _reader.ReadUInt32();
                }

                op.Parameters = new Dictionary<string, object>();
                for (int j = 0; j < paramCount; j++)
                {
                    var paramNameLength = _reader.ReadUInt16();
                    var paramNameBytes = _reader.ReadBytes(paramNameLength);
                    var paramName = Encoding.UTF8.GetString(paramNameBytes);

                    var paramType = (ParameterType)_reader.ReadByte();
                    var paramValue = ReadParameter(paramType);
                    op.Parameters[paramName] = paramValue;
                }

                _operators.Add(op);
            }
        }

        private object ReadParameter(ParameterType type)
        {
            switch (type)
            {
                case ParameterType.Int32:
                    return _reader.ReadInt32();
                case ParameterType.Int64:
                    return _reader.ReadInt64();
                case ParameterType.Float32:
                    return _reader.ReadSingle();
                case ParameterType.Float64:
                    return _reader.ReadDouble();
                case ParameterType.Bool:
                    return _reader.ReadByte() != 0;
                case ParameterType.Int32Array:
                    {
                        var count = _reader.ReadInt32();
                        var array = new int[count];
                        for (int k = 0; k < count; k++)
                        {
                            array[k] = _reader.ReadInt32();
                        }
                        return array;
                    }
                case ParameterType.Float32Array:
                    {
                        var count = _reader.ReadInt32();
                        var array = new float[count];
                        for (int k = 0; k < count; k++)
                        {
                            array[k] = _reader.ReadSingle();
                        }
                        return array;
                    }
                default:
                    throw new NotSupportedException($"Parameter type {type} is not supported");
            }
        }

        private void ValidateChecksum()
        {
            // Calculate Adler32 checksum for entire file
            _stream.Seek(0, SeekOrigin.Begin);
            var buffer = new byte[_stream.Length];
            _stream.Read(buffer, 0, buffer.Length);

            var adler = CalculateAdler32(buffer, buffer.Length - 4);
            _stream.Seek(-4, SeekOrigin.End);
            var storedChecksum = _reader.ReadUInt32();

            if (adler != storedChecksum)
            {
                throw new InvalidDataException(
                    $"Checksum mismatch: calculated=0x{adler:X8}, stored=0x{storedChecksum:X8}"
                );
            }
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
