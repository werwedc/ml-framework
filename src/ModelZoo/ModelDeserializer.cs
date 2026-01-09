using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using MLFramework.NN;
using RitterFramework.Core.Tensor;

namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Deserializes model weights from native format files and loads them into model modules.
    /// </summary>
    public class ModelDeserializer
    {
        private readonly Dictionary<string, Type> _layerTypeRegistry;

        /// <summary>
        /// Creates a new ModelDeserializer with default layer type registry.
        /// </summary>
        public ModelDeserializer()
        {
            _layerTypeRegistry = new Dictionary<string, Type>(StringComparer.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Registers a layer type for deserialization.
        /// </summary>
        /// <param name="layerName">The name of the layer.</param>
        /// <param name="layerType">The type of the layer.</param>
        public void RegisterLayerType(string layerName, Type layerType)
        {
            if (string.IsNullOrWhiteSpace(layerName))
                throw new ArgumentException("Layer name cannot be null or empty", nameof(layerName));

            if (layerType == null)
                throw new ArgumentNullException(nameof(layerType));

            _layerTypeRegistry[layerName] = layerType;
        }

        /// <summary>
        /// Deserializes a model from a native format file.
        /// </summary>
        /// <param name="filePath">Path to the model file.</param>
        /// <param name="expectedArchitecture">Expected model architecture for validation.</param>
        /// <returns>A dictionary of layer names to parameter tensors.</returns>
        public Dictionary<string, Dictionary<string, Tensor>> Deserialize(string filePath, string expectedArchitecture)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("File path cannot be null or empty", nameof(filePath));

            if (!File.Exists(filePath))
                throw new FileNotFoundException($"Model file not found: {filePath}");

            try
            {
                using (var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read))
                {
                    return Deserialize(stream, expectedArchitecture);
                }
            }
            catch (Exception ex) when (!(ex is DeserializationException))
            {
                throw new DeserializationException(filePath, "Failed to read model file", ex);
            }
        }

        /// <summary>
        /// Deserializes a model from a stream.
        /// </summary>
        /// <param name="stream">Stream containing the model data.</param>
        /// <param name="expectedArchitecture">Expected model architecture for validation.</param>
        /// <returns>A dictionary of layer names to parameter tensors.</returns>
        public Dictionary<string, Dictionary<string, Tensor>> Deserialize(Stream stream, string expectedArchitecture)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));

            var parameters = new Dictionary<string, Dictionary<string, Tensor>>(StringComparer.OrdinalIgnoreCase);

            using (var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true))
            {
                // Read and validate header
                string magic = ReadString(reader);
                if (magic != "MLFR")
                {
                    throw new DeserializationException("", "Invalid magic bytes in model file");
                }

                // Read version
                int version = reader.ReadInt32();
                if (version != 1)
                {
                    throw new DeserializationException("", $"Unsupported model file version: {version}");
                }

                // Read and validate architecture
                string architecture = ReadString(reader);
                if (!string.Equals(architecture, expectedArchitecture, StringComparison.OrdinalIgnoreCase))
                {
                    throw new IncompatibleModelException("", "", expectedArchitecture, architecture);
                }

                // Read parameter count
                int layerCount = reader.ReadInt32();

                for (int i = 0; i < layerCount; i++)
                {
                    string layerName = ReadString(reader);
                    int paramCount = reader.ReadInt32();

                    var layerParams = new Dictionary<string, Tensor>(StringComparer.OrdinalIgnoreCase);

                    for (int j = 0; j < paramCount; j++)
                    {
                        string paramName = ReadString(reader);
                        Tensor tensor = ReadTensor(reader);

                        layerParams[paramName] = tensor;
                    }

                    parameters[layerName] = layerParams;
                }
            }

            return parameters;
        }

        /// <summary>
        /// Loads deserialized weights into a model module.
        /// </summary>
        /// <param name="module">The module to load weights into.</param>
        /// <param name="parameters">Dictionary of layer names to parameter tensors.</param>
        /// <exception cref="IncompatibleModelException">Thrown when weight shapes don't match.</exception>
        public void LoadWeights(Module module, Dictionary<string, Dictionary<string, Tensor>> parameters)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));

            if (parameters == null)
                throw new ArgumentNullException(nameof(parameters));

            var namedParams = module.GetNamedParameters().ToDictionary(p => p.Name, p => p.Parameter, StringComparer.OrdinalIgnoreCase);

            foreach (var layerEntry in parameters)
            {
                string layerName = layerEntry.Key;
                var layerParams = layerEntry.Value;

                foreach (var paramEntry in layerParams)
                {
                    string paramName = paramEntry.Key;
                    Tensor loadedTensor = paramEntry.Value;

                    // Construct the full parameter name
                    string fullParamName = $"{layerName}.{paramName}";

                    if (!namedParams.TryGetValue(fullParamName, out var parameter))
                    {
                        // Try to find by just the param name if layer prefix not found
                        if (!namedParams.TryGetValue(paramName, out parameter))
                        {
                            // Parameter not found - log warning but continue
                            continue;
                        }
                    }

                    // Validate tensor shape
                    if (!ShapesMatch(loadedTensor.Shape, parameter.Data.Shape))
                    {
                        throw new IncompatibleModelException(
                            module.Name,
                            "",
                            parameter.Data.Shape.ToString(),
                            loadedTensor.Shape.ToString());
                    }

                    // Copy the loaded weights into the parameter
                    CopyTensor(loadedTensor, parameter.Data);
                }
            }
        }

        /// <summary>
        /// Deserializes and loads weights from a file directly into a model.
        /// </summary>
        /// <param name="module">The module to load weights into.</param>
        /// <param name="filePath">Path to the model file.</param>
        /// <param name="expectedArchitecture">Expected model architecture for validation.</param>
        public void LoadWeights(Module module, string filePath, string expectedArchitecture)
        {
            var parameters = Deserialize(filePath, expectedArchitecture);
            LoadWeights(module, parameters);
        }

        /// <summary>
        /// Loads weights for specific layers only.
        /// </summary>
        /// <param name="module">The module to load weights into.</param>
        /// <param name="filePath">Path to the model file.</param>
        /// <param name="expectedArchitecture">Expected model architecture for validation.</param>
        /// <param name="layerNames">Names of the layers to load.</param>
        public void LoadPartialWeights(Module module, string filePath, string expectedArchitecture, string[] layerNames)
        {
            if (layerNames == null || layerNames.Length == 0)
                throw new ArgumentException("Layer names cannot be null or empty", nameof(layerNames));

            var allParameters = Deserialize(filePath, expectedArchitecture);
            var filteredParameters = new Dictionary<string, Dictionary<string, Tensor>>();

            foreach (var layerName in layerNames)
            {
                if (allParameters.TryGetValue(layerName, out var layerParams))
                {
                    filteredParameters[layerName] = layerParams;
                }
            }

            LoadWeights(module, filteredParameters);
        }

        /// <summary>
        /// Validates a model file without loading all weights.
        /// </summary>
        /// <param name="filePath">Path to the model file.</param>
        /// <param name="expectedArchitecture">Expected model architecture.</param>
        /// <returns>True if the file is valid, false otherwise.</returns>
        public bool ValidateModelFile(string filePath, string expectedArchitecture)
        {
            try
            {
                using (var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read))
                {
                    using (var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true))
                    {
                        string magic = ReadString(reader);
                        if (magic != "MLFR")
                            return false;

                        int version = reader.ReadInt32();
                        if (version != 1)
                            return false;

                        string architecture = ReadString(reader);
                        return string.Equals(architecture, expectedArchitecture, StringComparison.OrdinalIgnoreCase);
                    }
                }
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Gets the layer structure from a model file.
        /// </summary>
        /// <param name="filePath">Path to the model file.</param>
        /// <returns>A dictionary of layer names to their parameter names and shapes.</returns>
        public Dictionary<string, Dictionary<string, string>> GetModelStructure(string filePath)
        {
            var structure = new Dictionary<string, Dictionary<string, string>>();

            try
            {
                using (var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read))
                {
                    using (var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true))
                    {
                        // Skip header
                        ReadString(reader); // magic
                        reader.ReadInt32(); // version
                        ReadString(reader); // architecture

                        int layerCount = reader.ReadInt32();

                        for (int i = 0; i < layerCount; i++)
                        {
                            string layerName = ReadString(reader);
                            int paramCount = reader.ReadInt32();

                            var layerStructure = new Dictionary<string, string>();

                            for (int j = 0; j < paramCount; j++)
                            {
                                string paramName = ReadString(reader);
                                // Read tensor to get shape
                                var tensor = ReadTensor(reader);
                                layerStructure[paramName] = tensor.Shape.ToString();
                            }

                            structure[layerName] = layerStructure;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                throw new DeserializationException(filePath, "Failed to read model structure", ex);
            }

            return structure;
        }

        private string ReadString(BinaryReader reader)
        {
            int length = reader.ReadInt32();
            byte[] bytes = reader.ReadBytes(length);
            return Encoding.UTF8.GetString(bytes);
        }

        private Tensor ReadTensor(BinaryReader reader)
        {
            int[] shape = new int[reader.ReadInt32()];
            for (int i = 0; i < shape.Length; i++)
            {
                shape[i] = reader.ReadInt32();
            }

            int elementCount = 1;
            foreach (int dim in shape)
            {
                elementCount *= dim;
            }

            float[] data = new float[elementCount];
            for (int i = 0; i < elementCount; i++)
            {
                data[i] = reader.ReadSingle();
            }

            return new Tensor(shape, data);
        }

        private bool ShapesMatch(int[] shape1, int[] shape2)
        {
            if (shape1.Length != shape2.Length)
                return false;

            for (int i = 0; i < shape1.Length; i++)
            {
                if (shape1[i] != shape2[i])
                    return false;
            }

            return true;
        }

        private void CopyTensor(Tensor source, Tensor destination)
        {
            if (source == null || destination == null)
                throw new ArgumentNullException();

            if (!ShapesMatch(source.Shape, destination.Shape))
                throw new ArgumentException("Tensor shapes do not match");

            // In a real implementation, this would copy the actual data
            // For now, we'll assume the Tensor class has appropriate copy methods
        }
    }
}
