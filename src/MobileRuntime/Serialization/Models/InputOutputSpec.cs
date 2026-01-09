using MobileRuntime;

namespace MobileRuntime.Serialization.Models
{
    /// <summary>
    /// Specification for input or output tensors
    /// </summary>
    public class InputOutputSpec
    {
        public string Name { get; set; }
        public ushort Rank { get; set; }
        public DataType DataType { get; set; }
        public ulong[] Shape { get; set; }

        public override string ToString()
        {
            var shapeStr = Shape != null ? string.Join("x", Shape) : "unknown";
            return $"{Name}: {DataType}[{shapeStr}]";
        }
    }
}
