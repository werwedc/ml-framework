using MLFramework.MobileRuntime.Backends.Cpu.Interfaces;
using MLFramework.MobileRuntime.Backends.Cpu.Models;
using MLFramework.MobileRuntime.Backends.Cpu.Executors;
using MLFramework.MobileRuntime.Memory;

namespace MLFramework.MobileRuntime.Backends.Cpu
{
    using System;
    using System.Collections.Generic;
    using System.Runtime.InteropServices;

    /// <summary>
    /// CPU-based execution backend with ARM NEON/SVE and Intel AVX optimizations.
    /// </summary>
    public sealed class CpuBackend : ICpuBackend
    {
        private readonly IMemoryPool _memoryPool;
        private readonly ITensorFactory _tensorFactory;
        private readonly CpuInfo _cpuInfo;
        private readonly Dictionary<OperatorType, IOperatorExecutor> _executors;
        private bool _vectorizationEnabled;
        private bool _multiThreadingEnabled;
        private int _maxThreads;

        /// <summary>
        /// Creates a new CPU backend.
        /// </summary>
        /// <param name="memoryPool">Memory pool for tensor allocations.</param>
        /// <param name="tensorFactory">Tensor factory for creating tensors.</param>
        public CpuBackend(IMemoryPool memoryPool, ITensorFactory tensorFactory)
        {
            _memoryPool = memoryPool ?? throw new ArgumentNullException(nameof(memoryPool));
            _tensorFactory = tensorFactory ?? throw new ArgumentNullException(nameof(tensorFactory));

            _cpuInfo = DetectCpuCapabilities();
            _executors = new Dictionary<OperatorType, IOperatorExecutor>();

            InitializeExecutors();

            // Enable optimizations by default
            _vectorizationEnabled = true;
            _multiThreadingEnabled = true;
            _maxThreads = 0; // Auto-detect
        }

        public string Name => "CPU";

        public BackendCapabilities Capabilities => _cpuInfo.Capabilities;

        /// <summary>
        /// Executes a single operator.
        /// </summary>
        public ITensor Execute(OperatorDescriptor op, ITensor[] inputs, Dictionary<string, object> parameters)
        {
            if (op == null)
                throw new ArgumentNullException(nameof(op));

            var executor = GetExecutor(op.Type);
            return executor.Execute(inputs, parameters ?? op.Parameters);
        }

        /// <summary>
        /// Executes a batch of operators efficiently.
        /// </summary>
        public ITensor[] ExecuteBatch(OperatorDescriptor[] ops, Dictionary<uint, ITensor> tensorRegistry)
        {
            if (ops == null)
                throw new ArgumentNullException(nameof(ops));

            var results = new Dictionary<uint, ITensor>();

            for (int i = 0; i < ops.Length; i++)
            {
                var op = ops[i];
                var executor = GetExecutor(op.Type);

                // Collect input tensors from registry
                var inputs = new ITensor[0]; // Would be populated from tensorRegistry

                // Check if this operator can be fused with the next one
                if (i < ops.Length - 1)
                {
                    var nextExecutor = GetExecutor(ops[i + 1].Type);
                    if (executor.CanFuseWith(nextExecutor))
                    {
                        // Execute fused operation
                        var fusedExecutors = new[] { executor, nextExecutor };
                        var fusedInputs = new ITensor[][] { inputs }; // Would collect all inputs
                        var fusedParams = new Dictionary<string, object>[] { op.Parameters, ops[i + 1].Parameters };

                        var result = executor.ExecuteFused(fusedExecutors, fusedInputs, fusedParams);
                        results[op.Id] = result;
                        results[ops[i + 1].Id] = result; // Both operators produce the same output
                        i++; // Skip the next operator as it's fused
                        continue;
                    }
                }

                // Execute standalone operation
                var output = executor.Execute(inputs, op.Parameters);
                results[op.Id] = output;
            }

            // Return results in order
            var outputArray = new ITensor[results.Count];
            int idx = 0;
            foreach (var result in results.Values)
            {
                outputArray[idx++] = result;
            }

            return outputArray;
        }

        /// <summary>
        /// Gets CPU information.
        /// </summary>
        public CpuInfo GetCpuInfo() => _cpuInfo;

        /// <summary>
        /// Enables or disables vectorization.
        /// </summary>
        public void EnableVectorization(bool enable) => _vectorizationEnabled = enable;

        /// <summary>
        /// Enables or disables multi-threading.
        /// </summary>
        public void EnableMultiThreading(bool enable, int maxThreads = 0)
        {
            _multiThreadingEnabled = enable;
            if (maxThreads > 0)
            {
                _maxThreads = Math.Min(maxThreads, _cpuInfo.Capabilities.MaxThreads);
            }
            else
            {
                _maxThreads = _cpuInfo.Capabilities.MaxThreads;
            }
        }

        /// <summary>
        /// Gets an executor for the specified operator type.
        /// </summary>
        internal IOperatorExecutor GetExecutor(OperatorType type)
        {
            if (!_executors.TryGetValue(type, out var executor))
            {
                throw new NotSupportedException($"Operator type '{type}' is not supported by the CPU backend.");
            }
            return executor;
        }

        /// <summary>
        /// Initializes operator executors.
        /// </summary>
        private void InitializeExecutors()
        {
            _executors[OperatorType.Conv2D] = new Conv2DExecutor(this);
            _executors[OperatorType.Relu] = new ReluExecutor(this);
            _executors[OperatorType.MaxPool2D] = new MaxPool2DExecutor(this);
            _executors[OperatorType.FullyConnected] = new FullyConnectedExecutor(this);
            _executors[OperatorType.Add] = new AddExecutor(this);
            _executors[OperatorType.Multiply] = new MultiplyExecutor(this);
            _executors[OperatorType.Concat] = new ConcatExecutor();
            _executors[OperatorType.Reshape] = new ReshapeExecutor();
        }

        /// <summary>
        /// Detects CPU capabilities at runtime.
        /// </summary>
        private CpuInfo DetectCpuCapabilities()
        {
            var cpuInfo = new CpuInfo();
            var caps = new BackendCapabilities();

            // Get basic CPU info
            cpuInfo.CoreCount = Environment.ProcessorCount;
            cpuInfo.ThreadCount = Environment.ProcessorCount;
            caps.MaxThreads = Environment.ProcessorCount;

            // Detect architecture
            var arch = RuntimeInformation.OSArchitecture;

            if (arch == Architecture.Arm64 || arch == Architecture.Arm)
            {
                cpuInfo.Vendor = "ARM";

                // ARM NEON detection (simplified - in real implementation would use proper detection)
                // ARM64 always has NEON
                if (arch == Architecture.Arm64)
                {
                    caps.SupportsNeon = true;
                }

                // SVE detection (would require proper CPUID or feature detection)
                // For now, we'll assume it's not available unless explicitly detected
                caps.SupportsSve = false;

                cpuInfo.Model = arch == Architecture.Arm64 ? "ARM64" : "ARM";
            }
            else if (arch == Architecture.X64 || arch == Architecture.X86)
            {
                cpuInfo.Vendor = "x86/x64";

                // AVX detection (simplified - in real implementation would use CPUID)
                // Modern x64 CPUs typically support AVX
                if (arch == Architecture.X64)
                {
                    caps.SupportsAvx = true;
                    // AVX2 and AVX-512 would require CPUID detection
                    caps.SupportsAvx2 = true;
                    caps.SupportsAvx512 = false;
                }

                cpuInfo.Model = arch == Architecture.X64 ? "x64" : "x86";
            }

            // Cache line size (typical values)
            caps.CacheLineSize = 64; // Most modern CPUs have 64-byte cache lines

            cpuInfo.Capabilities = caps;

            return cpuInfo;
        }

        /// <summary>
        /// Gets whether vectorization is enabled.
        /// </summary>
        internal bool IsVectorizationEnabled() => _vectorizationEnabled;

        /// <summary>
        /// Gets whether multi-threading is enabled.
        /// </summary>
        internal bool IsMultiThreadingEnabled() => _multiThreadingEnabled;

        /// <summary>
        /// Gets the maximum number of threads to use.
        /// </summary>
        internal int GetMaxThreads() => _maxThreads;

        /// <summary>
        /// Gets the memory pool.
        /// </summary>
        internal IMemoryPool GetMemoryPool() => _memoryPool;

        /// <summary>
        /// Gets the tensor factory.
        /// </summary>
        internal ITensorFactory GetTensorFactory() => _tensorFactory;
    }
}
