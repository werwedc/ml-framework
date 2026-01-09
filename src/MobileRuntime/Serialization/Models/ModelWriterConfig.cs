using MobileRuntime.Serialization.Models;

namespace MobileRuntime.Serialization.Models
{
    /// <summary>
    /// Configuration for writing a model to the binary format
    /// </summary>
    public class ModelWriterConfig
    {
        public string ModelName { get; set; }
        public uint FrameworkVersion { get; set; }
        public InputOutputSpec[] Inputs { get; set; }
        public InputOutputSpec[] Outputs { get; set; }
        public ConstantTensor[] ConstantTensors { get; set; }
        public OperatorDescriptor[] Operators { get; set; }
        public uint Flags { get; set; }

        public ModelWriterConfig()
        {
            FrameworkVersion = 1;
            Inputs = new InputOutputSpec[0];
            Outputs = new InputOutputSpec[0];
            ConstantTensors = new ConstantTensor[0];
            Operators = new OperatorDescriptor[0];
            Flags = 0;
        }
    }
}
