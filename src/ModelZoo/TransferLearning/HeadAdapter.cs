using System;
using MLFramework.NN;

namespace MLFramework.ModelZoo.TransferLearning
{
    /// <summary>
    /// Exception thrown when head adaptation operations fail.
    /// </summary>
    public class HeadAdapterException : Exception
    {
        public HeadAdapterException(string message) : base(message) { }
        public HeadAdapterException(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    /// Adapter for managing head replacement and adaptation for transfer learning.
    /// Provides a high-level interface for dynamically adjusting model heads.
    /// </summary>
    public class HeadAdapter
    {
        private readonly SequentialModule _model;
        private Module _currentHead;
        private int _inputDim;
        private int _numClasses;

        /// <summary>
        /// Gets the model being adapted.
        /// </summary>
        public SequentialModule Model => _model;

        /// <summary>
        /// Gets the current head module.
        /// </summary>
        public Module CurrentHead => _currentHead;

        /// <summary>
        /// Gets the expected input dimension for the head.
        /// </summary>
        public int InputDim => _inputDim;

        /// <summary>
        /// Gets the current number of output classes.
        /// </summary>
        public int NumClasses => _numClasses;

        /// <summary>
        /// Creates a new head adapter for the given model.
        /// </summary>
        /// <param name="model">The sequential model to adapt.</param>
        /// <param name="inputDim">Expected input dimension for the head.</param>
        public HeadAdapter(SequentialModule model, int inputDim = -1)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _inputDim = inputDim;

            // Try to get the current head and input dimension
            _currentHead = model.GetHead();
            if (_inputDim == -1)
            {
                _inputDim = model.GetHeadInputDimension() ?? -1;
            }

            // Try to determine number of classes from current head
            if (_currentHead is LinearHead linear)
            {
                // This would need to get the output dimension from the weight tensor
                // For now, we'll store a placeholder
                _numClasses = 1000; // Default value
            }
            else
            {
                _numClasses = 1000; // Default value
            }
        }

        /// <summary>
        /// Sets the number of output classes and dynamically adjusts the head.
        /// </summary>
        /// <param name="numClasses">New number of classes.</param>
        /// <param name="headType">Type of head to create ("linear", "mlp", etc.).</param>
        /// <param name="initializeStrategy">Weight initialization strategy.</param>
        public void SetNumberOfClasses(int numClasses, string headType = "linear", WeightInitializationStrategy initializeStrategy = WeightInitializationStrategy.Kaiming)
        {
            if (numClasses <= 0)
                throw new ArgumentOutOfRangeException(nameof(numClasses), "Number of classes must be positive.");

            if (_inputDim <= 0)
                throw new HeadAdapterException("Input dimension is unknown. Please set InputDim before calling SetNumberOfClasses.");

            _numClasses = numClasses;

            // Create new head
            var newHead = HeadBuilder.CreateDefaultHead(_inputDim, numClasses, headType);

            // Initialize weights
            HeadBuilder.InitializeHead(newHead, initializeStrategy);

            // Replace the current head
            if (_currentHead != null)
            {
                var oldHead = model.ReplaceHead(newHead);
                _currentHead = newHead;
            }
            else
            {
                // No head exists, add it
                model.AddHead(newHead);
                _currentHead = newHead;
            }
        }

        /// <summary>
        /// Replaces the current head with a custom module.
        /// </summary>
        /// <param name="newHead">The new head module.</param>
        /// <param name="validateDimensions">Whether to validate dimension compatibility.</param>
        public void ReplaceHead(Module newHead, bool validateDimensions = true)
        {
            if (newHead == null)
                throw new ArgumentNullException(nameof(newHead));

            if (validateDimensions && _inputDim > 0)
            {
                if (!_model.ValidateHeadReplacement(newHead, _inputDim))
                {
                    throw new HeadAdapterException(
                        "Head validation failed. The new head may not be compatible with the model's architecture.");
                }
            }

            var oldHead = _model.ReplaceHead(newHead);
            _currentHead = newHead;

            // Update numClasses if the new head is a LinearHead
            if (newHead is LinearHead linear)
            {
                // This would need to get the output dimension from the weight tensor
                // For now, keep the existing _numClasses value
            }
        }

        /// <summary>
        /// Initializes the current head weights with the given strategy.
        /// </summary>
        /// <param name="strategy">Weight initialization strategy.</param>
        public void InitializeWeights(WeightInitializationStrategy strategy)
        {
            if (_currentHead == null)
                throw new HeadAdapterException("No head to initialize. Please create a head first.");

            HeadBuilder.InitializeHead(_currentHead, strategy);
        }

        /// <summary>
        /// Sets the expected input dimension for the head.
        /// </summary>
        /// <param name="inputDim">Input dimension.</param>
        public void SetInputDim(int inputDim)
        {
            if (inputDim <= 0)
                throw new ArgumentOutOfRangeException(nameof(inputDim), "Input dimension must be positive.");

            _inputDim = inputDim;
        }

        /// <summary>
        /// Gets the expected input dimension for the head.
        /// If not explicitly set, tries to infer from the model.
        /// </summary>
        /// <returns>The input dimension, or -1 if unknown.</returns>
        public int GetInputDim()
        {
            if (_inputDim > 0)
                return _inputDim;

            return _model.GetHeadInputDimension() ?? -1;
        }

        /// <summary>
        /// Resets the adapter, removing the current head and reverting to default state.
        /// </summary>
        public void Reset()
        {
            if (_currentHead != null)
            {
                _model.RemoveLastLayer();
                _currentHead = null;
            }

            _numClasses = 1000; // Reset to default
        }

        /// <summary>
        /// Creates a new head adapter for a model.
        /// </summary>
        /// <param name="model">The sequential model.</param>
        /// <param name="inputDim">Input dimension (optional, will try to infer if not provided).</param>
        /// <returns>A new HeadAdapter instance.</returns>
        public static HeadAdapter Create(SequentialModule model, int inputDim = -1)
        {
            return new HeadAdapter(model, inputDim);
        }

        /// <summary>
        /// Adapts a pre-trained model for transfer learning.
        /// </summary>
        /// <param name="model">The pre-trained model.</param>
        /// <param name="numClasses">Number of target classes.</param>
        /// <param name="freezeBackbone">Whether to freeze all layers except the head.</param>
        /// <param name="headType">Type of head to create.</param>
        /// <returns>A configured HeadAdapter.</returns>
        public static HeadAdapter AdaptPretrained(
            SequentialModule model,
            int numClasses,
            bool freezeBackbone = true,
            string headType = "linear")
        {
            var adapter = new HeadAdapter(model);

            // Freeze backbone if requested
            if (freezeBackbone && model.Count > 1)
            {
                // Freeze all except the last layer (which will be the head)
                // This would require implementation of FreezeExtensions
                // For now, we'll just set parameters to not require gradients
                for (int i = 0; i < model.Count - 1; i++)
                {
                    var module = model.GetModule(i);
                    module.SetRequiresGrad(false);
                }
            }

            // Set up new head
            adapter.SetNumberOfClasses(numClasses, headType, WeightInitializationStrategy.Kaiming);

            return adapter;
        }

        /// <summary>
        /// Gets a summary of the adapter state.
        /// </summary>
        /// <returns>A string describing the adapter configuration.</returns>
        public string GetSummary()
        {
            var summary = $"HeadAdapter Summary:\n";
            summary += $"  Input Dimension: {_inputDim}\n";
            summary += $"  Number of Classes: {_numClasses}\n";
            summary += $"  Current Head: {(_currentHead?.Name ?? "None")}\n";
            summary += $"  Model Layers: {_model.Count}\n";
            summary += $"  Head Summary: {_model.GetHeadSummary()}";

            return summary;
        }
    }
}
