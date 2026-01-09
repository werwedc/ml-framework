using System;

namespace MobileRuntime
{
    /// <summary>
    /// Mobile runtime implementation
    /// </summary>
    public class RuntimeMobileRuntime : MobileRuntime
    {
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the RuntimeMobileRuntime class
        /// </summary>
        public RuntimeMobileRuntime() : base()
        {
        }

        /// <summary>
        /// Loads a model from a file path
        /// </summary>
        public override IModel LoadModel(string modelPath)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(RuntimeMobileRuntime));

            if (string.IsNullOrEmpty(modelPath))
                throw new ArgumentException("Model path cannot be null or empty", nameof(modelPath));

            return new ConcreteModel(modelPath);
        }

        /// <summary>
        /// Loads a model from a byte array
        /// </summary>
        public override IModel LoadModel(byte[] modelBytes)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(RuntimeMobileRuntime));

            if (modelBytes == null || modelBytes.Length == 0)
                throw new ArgumentException("Model bytes cannot be null or empty", nameof(modelBytes));

            return new ConcreteModel(modelBytes);
        }

        /// <summary>
        /// Gets runtime information
        /// </summary>
        public override RuntimeInfo GetRuntimeInfo()
        {
            return new RuntimeInfo
            {
                Version = "1.0.0",
                Platform = Environment.OSVersion.Platform.ToString(),
                DeviceInfo = Environment.MachineName,
                SupportedBackends = new[] { BackendType.Cpu }
            };
        }

        /// <summary>
        /// Disposes resources
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Cleanup managed resources
                }
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Concrete model implementation
    /// </summary>
    internal class ConcreteModel : Model
    {
        private readonly string? _modelPath;
        private readonly byte[]? _modelBytes;
        private bool _disposed;

        public ConcreteModel(string modelPath)
        {
            _modelPath = modelPath ?? throw new ArgumentNullException(nameof(modelPath));
            _modelBytes = Array.Empty<byte>();
        }

        public ConcreteModel(byte[] modelBytes)
        {
            _modelBytes = modelBytes ?? throw new ArgumentNullException(nameof(modelBytes));
            _modelPath = null; // Set to null so Name property returns default
        }

        public override string Name => string.IsNullOrEmpty(_modelPath) ? "LoadedModel" : _modelPath;

        public override InputInfo[] Inputs => new[]
        {
            new InputInfo { Name = "input", Shape = new[] { 1, 28, 28 }, DataType = DataType.Float32 }
        };

        public override OutputInfo[] Outputs => new[]
        {
            new OutputInfo { Name = "output", Shape = new[] { 10 }, DataType = DataType.Float32 }
        };

        public override long MemoryFootprint => _modelBytes?.Length ?? 0;

        public override ModelFormat Format => ModelFormat.MobileBinary;

        public override ITensor[] Predict(ITensor[] inputs)
        {
            ValidateInputs(inputs);

            // Placeholder - return simple echo for now
            var outputs = new ITensor[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                var inputData = inputs[i].Data;
                var tensor = Tensor.FromArray(inputData, inputs[i].Shape);
                outputs[i] = tensor;
            }
            return outputs;
        }

        public override System.Threading.Tasks.Task<ITensor[]> PredictAsync(ITensor[] inputs)
        {
            return System.Threading.Tasks.Task.FromResult(Predict(inputs));
        }

        public override void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
            }
        }
    }
}
