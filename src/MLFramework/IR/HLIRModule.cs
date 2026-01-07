using System;
using System.Collections.Generic;
using RitterFramework.Core;

namespace MLFramework.IR
{
    /// <summary>
    /// High-level IR module containing operations and data flow
    /// </summary>
    public class HLIRModule
    {
        public string Name { get; set; }
        public List<HIROperation> Operations { get; set; } = new List<HIROperation>();
        public List<HIRValue> Values { get; set; } = new List<HIRValue>();
        public List<HIRFunction> Functions { get; set; } = new List<HIRFunction>();

        public HLIRModule(string name = "module")
        {
            Name = name;
        }
    }

    /// <summary>
    /// High-level IR operation
    /// </summary>
    public class HIROperation
    {
        public string Name { get; set; }
        public string OpCode { get; set; }
        public List<HIRValue> Operands { get; set; } = new List<HIRValue>();
        public HIRValue Result { get; set; }

        public HIROperation(string opCode, List<HIRValue> operands, string name = "")
        {
            OpCode = opCode;
            Operands = operands;
            Name = string.IsNullOrEmpty(name) ? $"op_{Guid.NewGuid()}" : name;
        }
    }

    /// <summary>
    /// High-level IR value (placeholder)
    /// </summary>
    public class HIRValue
    {
        public string Name { get; set; }
        public DataType DataType { get; set; }
        public int[] Shape { get; set; }

        public HIRValue(string name, DataType dataType, int[] shape)
        {
            Name = name;
            DataType = dataType;
            Shape = shape;
        }
    }

    /// <summary>
    /// High-level IR function
    /// </summary>
    public class HIRFunction
    {
        public string Name { get; set; }
        public List<HIROperation> Operations { get; set; } = new List<HIROperation>();
        public List<HIRValue> Parameters { get; set; } = new List<HIRValue>();
        public HIRValue Result { get; set; }
    }

    /// <summary>
    /// IR pass for transformation
    /// </summary>
    public interface IRPass
    {
        string Name { get; }
        void Run(HLIRModule module);
    }

    /// <summary>
    /// IR pass manager for running transformations
    /// </summary>
    public class IRPassManager
    {
        public enum PassType
        {
            Transformation,
            Optimization,
            Analysis
        }

        private Dictionary<PassType, List<IRPass>> _passes = new Dictionary<PassType, List<IRPass>>();

        public IRPassManager()
        {
            _passes[PassType.Transformation] = new List<IRPass>();
            _passes[PassType.Optimization] = new List<IRPass>();
            _passes[PassType.Analysis] = new List<IRPass>();
        }

        public void AddPass(IRPass pass, PassType type)
        {
            _passes[type].Add(pass);
        }

        public void RunAll(HLIRModule module)
        {
            foreach (var passList in _passes.Values)
            {
                foreach (var pass in passList)
                {
                    pass.Run(module);
                }
            }
        }

        public List<IRPass> GetPasses(PassType type) => _passes[type];
    }
}
