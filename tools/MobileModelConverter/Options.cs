using CommandLine;

namespace MobileModelConverter
{
    public class Options
    {
        [Option('i', "input", Required = true, HelpText = "Input model file path")]
        public string Input { get; set; } = string.Empty;

        [Option('o', "output", Required = true, HelpText = "Output mobile model file path")]
        public string Output { get; set; } = string.Empty;

        [Option('q', "quantize", Required = false, HelpText = "Quantization type (int8, uint8, fp16)", Default = null)]
        public string? Quantize { get; set; }

        [Option('v', "verbose", Required = false, HelpText = "Verbose output", Default = false)]
        public bool Verbose { get; set; }
    }
}
