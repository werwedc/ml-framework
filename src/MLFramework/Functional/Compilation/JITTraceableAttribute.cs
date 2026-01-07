using System;

namespace MLFramework.Functional.Compilation
{
    [AttributeUsage(AttributeTargets.Method)]
    public class JITTraceableAttribute : Attribute
    {
        public bool ForceCompilation { get; set; }
    }
}
