using MLFramework.Distributed.NCCL;
using System;
using System.Runtime.InteropServices;

namespace MLFramework.Distributed.NCCL
{
    /// <summary>
    /// NCCL unique ID for initialization (128 bytes).
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Size = 128)]
    public struct NCCLUniqueId
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 128)]
        private readonly byte[] _data;

        /// <summary>
        /// Creates a new NCCLUniqueId from the given byte array.
        /// </summary>
        public NCCLUniqueId(byte[] data)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            if (data.Length != 128)
                throw new ArgumentException("NCCLUniqueId must be 128 bytes", nameof(data));

            _data = (byte[])data.Clone();
        }

        /// <summary>
        /// Gets the unique ID data.
        /// </summary>
        public byte[] Data => _data;

        /// <summary>
        /// Generates a new NCCL unique ID.
        /// </summary>
        public static NCCLUniqueId Generate()
        {
            var id = new byte[128];
            var error = NCCLNative.ncclGetUniqueId(id);

            NCCLNative.CheckError(error, 0, "ncclGetUniqueId");

            return new NCCLUniqueId(id);
        }

        /// <summary>
        /// Serializes the unique ID to a base64 string for network transmission.
        /// </summary>
        public string Serialize()
        {
            return Convert.ToBase64String(_data);
        }

        /// <summary>
        /// Deserializes a unique ID from a base64 string.
        /// </summary>
        public static NCCLUniqueId Deserialize(string serialized)
        {
            if (string.IsNullOrEmpty(serialized))
                throw new ArgumentException("Serialized unique ID cannot be null or empty", nameof(serialized));

            var data = Convert.FromBase64String(serialized);

            if (data.Length != 128)
                throw new ArgumentException("Deserialized unique ID must be 128 bytes", nameof(serialized));

            return new NCCLUniqueId(data);
        }
    }
}
