using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace MobileRuntime.Backends.Metal
{
    /// <summary>
    /// Implementation of Metal-based GPU backend for iOS devices
    /// </summary>
    public sealed class MetalBackend : IMetalBackend, IDisposable
    {
        private IntPtr _device;
        private IntPtr _commandQueue;
        private readonly Dictionary<OperatorType, IMetalComputeShader> _shaders;
        private readonly List<IMetalBuffer> _allocatedBuffers;
        private readonly ITensorFactory _tensorFactory;
        private MetalBackendCapabilities _capabilities;
        private MetalDeviceInfo _deviceInfo;
        private bool _disposed;

        // Metal P/Invoke declarations
        [DllImport("__Internal", EntryPoint = "MTLCreateSystemDefaultDevice")]
        private static extern IntPtr MTLCreateSystemDefaultDevice();

        [DllImport("__Internal", EntryPoint = "MTLDevice_release")]
        private static extern void MTLDevice_release(IntPtr device);

        [DllImport("__Internal", EntryPoint = "MTLDevice_newCommandQueue")]
        private static extern IntPtr MTLDevice_newCommandQueue(IntPtr device);

        [DllImport("__Internal", EntryPoint = "MTLCommandQueue_release")]
        private static extern void MTLCommandQueue_release(IntPtr commandQueue);

        /// <summary>
        /// Creates a new Metal backend
        /// </summary>
        public MetalBackend(ITensorFactory tensorFactory)
        {
            _tensorFactory = tensorFactory;
            _shaders = new Dictionary<OperatorType, IMetalComputeShader>();
            _allocatedBuffers = new List<IMetalBuffer>();

            InitializeMetal();
            InitializeShaders();
            DetectCapabilities();
        }

        /// <summary>
        /// Finalizer
        /// </summary>
        ~MetalBackend()
        {
            Dispose(false);
        }

        /// <inheritdoc/>
        public string Name => "Metal";

        /// <inheritdoc/>
        public MetalBackendCapabilities Capabilities => _capabilities;

        /// <inheritdoc/>
        public IMetalBuffer AllocateBuffer(long size)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(MetalBackend));

            var buffer = new MetalBuffer(_device, size);
            _allocatedBuffers.Add(buffer);
            return buffer;
        }

        /// <inheritdoc/>
        public void FreeBuffer(IMetalBuffer buffer)
        {
            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer));

            _allocatedBuffers.Remove(buffer);
            buffer.Dispose();
        }

        /// <inheritdoc/>
        public ITensor Execute(OperatorDescriptor op, ITensor[] inputs, Dictionary<string, object> parameters)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(MetalBackend));

            if (op == null)
                throw new ArgumentNullException(nameof(op));

            if (inputs == null || inputs.Length == 0)
                throw new ArgumentException("Operator requires at least one input", nameof(inputs));

            // Get the shader for this operator
            var shader = GetComputeShader(op.Type);
            if (shader == null)
                throw new NotSupportedException($"Operator {op.Type} is not supported");

            // Allocate output tensor
            var outputShape = CalculateOutputShape(op.Type, inputs, parameters);
            var output = _tensorFactory.CreateTensor(outputShape, DataType.Float32);

            // Allocate Metal buffers for inputs and output
            var inputBuffers = new MetalBuffer[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                long bufferSize = inputs[i].Length * sizeof(float);
                inputBuffers[i] = (MetalBuffer)AllocateBuffer(bufferSize);

                // Copy input data to Metal buffer
                IntPtr inputDataPtr = Marshal.UnsafeAddrOfPinnedArrayElement(inputs[i].Data, 0);
                inputBuffers[i].CopyFrom(inputDataPtr, bufferSize);
            }

            var outputBuffer = (MetalBuffer)AllocateBuffer(output.Length * sizeof(float));

            // Create command buffer and execute shader
            using var commandBuffer = new MetalCommandBuffer(this);
            shader.Dispatch(commandBuffer, inputBuffers, new[] { outputBuffer }, parameters);
            commandBuffer.Commit();
            commandBuffer.WaitUntilCompleted();

            // Copy output data from Metal buffer
            IntPtr outputDataPtr = outputBuffer.Contents;
            output.Data = new float[output.Length];
            Marshal.Copy(outputDataPtr, output.Data, 0, output.Length);

            // Free temporary buffers
            foreach (var buffer in inputBuffers)
            {
                FreeBuffer(buffer);
            }
            FreeBuffer(outputBuffer);

            return output;
        }

        /// <inheritdoc/>
        public ITensor[] ExecuteBatch(OperatorDescriptor[] ops, Dictionary<uint, ITensor> tensorRegistry)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(MetalBackend));

            if (ops == null || ops.Length == 0)
                throw new ArgumentException("At least one operator is required", nameof(ops));

            var results = new List<ITensor>();

            // Execute each operator
            foreach (var op in ops)
            {
                // Get input tensors from registry
                var inputs = new ITensor[op.InputIds.Length];
                for (int i = 0; i < op.InputIds.Length; i++)
                {
                    if (!tensorRegistry.TryGetValue(op.InputIds[i], out var inputTensor))
                        throw new ArgumentException($"Input tensor with ID {op.InputIds[i]} not found in registry");

                    inputs[i] = inputTensor;
                }

                // Execute operator
                var output = Execute(op, inputs, op.Parameters);
                results.Add(output);

                // Register output tensor
                if (op.OutputIds.Length > 0)
                {
                    tensorRegistry[op.OutputIds[0]] = output;
                }
            }

            return results.ToArray();
        }

        /// <inheritdoc/>
        public IMetalComputeShader GetComputeShader(OperatorType opType)
        {
            if (_shaders.TryGetValue(opType, out var shader))
                return shader;

            return null;
        }

        /// <inheritdoc/>
        public void WaitForCompletion()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(MetalBackend));

            // Wait for all pending operations to complete
            // This would synchronize with the command queue
        }

        /// <inheritdoc/>
        public MetalDeviceInfo GetDeviceInfo()
        {
            return _deviceInfo;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Disposes of the backend
        /// </summary>
        private void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Free all allocated buffers
                    foreach (var buffer in _allocatedBuffers)
                    {
                        buffer.Dispose();
                    }
                    _allocatedBuffers.Clear();

                    // Dispose all shaders
                    foreach (var shader in _shaders.Values)
                    {
                        if (shader is IDisposable disposableShader)
                        {
                            disposableShader.Dispose();
                        }
                    }
                    _shaders.Clear();
                }

                // Release Metal resources
                if (_commandQueue != IntPtr.Zero)
                {
                    MTLCommandQueue_release(_commandQueue);
                    _commandQueue = IntPtr.Zero;
                }

                if (_device != IntPtr.Zero)
                {
                    MTLDevice_release(_device);
                    _device = IntPtr.Zero;
                }

                _disposed = true;
            }
        }

        /// <summary>
        /// Initializes Metal device and command queue
        /// </summary>
        private void InitializeMetal()
        {
            _device = MTLCreateSystemDefaultDevice();
            if (_device == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create Metal device. Metal may not be available on this device.");

            _commandQueue = MTLDevice_newCommandQueue(_device);
            if (_commandQueue == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create Metal command queue.");
        }

        /// <summary>
        /// Initializes all compute shaders
        /// </summary>
        private void InitializeShaders()
        {
            // Register compute shaders for each supported operator
            _shaders[OperatorType.Conv2D] = new MetalConv2DShader(this);
            _shaders[OperatorType.Relu] = new MetalReluShader(this);
            _shaders[OperatorType.MaxPool2D] = new MetalMaxPool2DShader(this);
            _shaders[OperatorType.FullyConnected] = new MetalFullyConnectedShader(this);
            _shaders[OperatorType.BatchNorm] = new MetalBatchNormShader(this);
            _shaders[OperatorType.Softmax] = new MetalSoftmaxShader(this);
            _shaders[OperatorType.Add] = new MetalElementWiseShader(this, OperatorType.Add);
            _shaders[OperatorType.Sub] = new MetalElementWiseShader(this, OperatorType.Sub);
            _shaders[OperatorType.Mul] = new MetalElementWiseShader(this, OperatorType.Mul);
            _shaders[OperatorType.Div] = new MetalElementWiseShader(this, OperatorType.Div);
            _shaders[OperatorType.Sigmoid] = new MetalElementWiseShader(this, OperatorType.Sigmoid);
            _shaders[OperatorType.Tanh] = new MetalElementWiseShader(this, OperatorType.Tanh);
            _shaders[OperatorType.LeakyRelu] = new MetalElementWiseShader(this, OperatorType.LeakyRelu);
        }

        /// <summary>
        /// Detects Metal device capabilities
        /// </summary>
        private void DetectCapabilities()
        {
            _capabilities = new MetalBackendCapabilities
            {
                SupportsMPS = CheckMPSSupport(),
                SupportsUnifiedMemory = CheckUnifiedMemorySupport(),
                MaxThreadsPerThreadgroup = 1024, // Standard for Apple GPUs
                MaxTextureWidth = 16384,
                MaxTextureHeight = 16384,
                MaxBufferLength = 256 * 1024 * 1024 // 256 MB
            };

            _deviceInfo = new MetalDeviceInfo
            {
                DeviceName = GetDeviceName(),
                FamilyId = GetDeviceFamilyId(),
                RecommendedMaxWorkingSetSize = GetRecommendedWorkingSetSize(),
                HasUnifiedMemory = _capabilities.SupportsUnifiedMemory,
                Capabilities = _capabilities
            };
        }

        private bool CheckMPSSupport()
        {
            // Check if Metal Performance Shaders is available
            // This would query the Metal framework
            return true; // Assume MPS is available on iOS 11+
        }

        private bool CheckUnifiedMemorySupport()
        {
            // Check if unified memory is supported
            return true; // Apple Silicon always has unified memory
        }

        private string GetDeviceName()
        {
            // Get the device name from Metal
            return "Apple GPU"; // Placeholder
        }

        private uint GetDeviceFamilyId()
        {
            // Get the device family ID
            return 6; // Placeholder for Apple A-series
        }

        private uint GetRecommendedWorkingSetSize()
        {
            // Get the recommended working set size
            return 512 * 1024 * 1024; // 512 MB default
        }

        private int[] CalculateOutputShape(OperatorType opType, ITensor[] inputs, Dictionary<string, object> parameters)
        {
            // Calculate output shape based on operator type and inputs
            // This is a simplified implementation
            switch (opType)
            {
                case OperatorType.Conv2D:
                    int outChannels = parameters.TryGetValue("out_channels", out var oc) ? Convert.ToInt32(oc) : 64;
                    return new[] { inputs[0].Shape[0], outChannels, inputs[0].Shape[2], inputs[0].Shape[3] };

                case OperatorType.MaxPool2D:
                case OperatorType.Relu:
                case OperatorType.Sigmoid:
                case OperatorType.Tanh:
                    return (int[])inputs[0].Shape.Clone();

                case OperatorType.FullyConnected:
                    int outputSize = parameters.TryGetValue("output_size", out var os) ? Convert.ToInt32(os) : 1024;
                    return new[] { inputs[0].Shape[0], outputSize };

                case OperatorType.Softmax:
                    return (int[])inputs[0].Shape.Clone();

                case OperatorType.Add:
                case OperatorType.Sub:
                case OperatorType.Mul:
                case OperatorType.Div:
                    return (int[])inputs[0].Shape.Clone();

                default:
                    return (int[])inputs[0].Shape.Clone();
            }
        }
    }
}
