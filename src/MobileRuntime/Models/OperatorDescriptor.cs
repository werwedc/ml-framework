using System;
using System.Collections.Generic;

namespace MobileRuntime.Models;

public class OperatorDescriptor
{
    public OperatorType Type { get; set; }
    public uint[] InputTensorIds { get; set; } = Array.Empty<uint>();
    public uint[] OutputTensorIds { get; set; } = Array.Empty<uint>();
    public Dictionary<string, object> Parameters { get; set; } = new Dictionary<string, object>();
}
