using System;
using MobileRuntime;

namespace MobileRuntime.Serialization.Models
{
    /// <summary>
    /// Constant tensor data stored in the model file
    /// </summary>
    public class ConstantTensor
    {
        public uint Id { get; set; }
        public DataType DataType { get; set; }
        public ushort Rank { get; set; }
        public ulong[] Shape { get; set; } = Array.Empty<ulong>();
        public ulong DataSize { get; set; }
        public byte[] Data { get; set; } = Array.Empty<byte>();

        /// <summary>
        /// Convert constant tensor to a runtime Tensor
        /// </summary>
        public Tensor ToTensor()
        {
            if (Shape == null || Shape.Length == 0)
            {
                throw new InvalidOperationException("Cannot convert tensor with null or empty shape");
            }

            if (Data == null || Data.Length == 0)
            {
                throw new InvalidOperationException("Cannot convert tensor with null or empty data");
            }

            // Convert ulong[] shape to int[] shape
            var intShape = new int[Rank];
            for (int i = 0; i < Rank; i++)
            {
                intShape[i] = (int)Shape[i];
            }

            // Create tensor based on data type
            switch (DataType)
            {
                case DataType.Float32:
                    {
                        var floatData = new float[Data.Length / 4];
                        Buffer.BlockCopy(Data, 0, floatData, 0, Data.Length);
                        return Tensor.FromArray(floatData, intShape);
                    }
                case DataType.Float16:
                    // TODO: Implement FP16 support
                    throw new NotImplementedException("Float16 tensor conversion not yet implemented");
                case DataType.Int32:
                    {
                        var intData = new int[Data.Length / 4];
                        Buffer.BlockCopy(Data, 0, intData, 0, Data.Length);
                        return Tensor.FromArray(intData, intShape);
                    }
                case DataType.Int8:
                    {
                        var floatData = new float[Data.Length];
                        for (int i = 0; i < Data.Length; i++)
                        {
                            floatData[i] = (sbyte)Data[i];
                        }
                        return Tensor.FromArray(floatData, intShape);
                    }
                case DataType.Int16:
                    {
                        var shortData = new short[Data.Length / 2];
                        Buffer.BlockCopy(Data, 0, shortData, 0, Data.Length);
                        var floatData = new float[shortData.Length];
                        for (int i = 0; i < shortData.Length; i++)
                        {
                            floatData[i] = shortData[i];
                        }
                        return Tensor.FromArray(floatData, intShape);
                    }
                default:
                    throw new NotSupportedException($"Data type {DataType} is not supported");
            }
        }
    }
}
