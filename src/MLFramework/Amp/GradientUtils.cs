using System.Collections.Generic;
using RitterFramework.Core.Tensor;

namespace MLFramework.Amp
{
    /// <summary>
    /// Utility functions for gradient manipulation in AMP
    /// </summary>
    public static class GradientUtils
    {
        /// <summary>
        /// Unscales a gradient tensor by dividing by the scale factor
        /// </summary>
        /// <param name="gradient">The gradient tensor to unscale</param>
        /// <param name="scale">The scale factor to divide by</param>
        /// <returns>Unscaled gradient tensor</returns>
        public static Tensor Unscale(Tensor gradient, float scale)
        {
            if (gradient == null)
                throw new System.ArgumentNullException(nameof(gradient));

            if (scale <= 0)
                throw new System.ArgumentException("Scale must be positive", nameof(scale));

            if (scale == 1.0f)
                return gradient;

            return gradient * (1.0f / scale);
        }

        /// <summary>
        /// Unscales multiple gradient tensors
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <param name="scale">The scale factor to divide by</param>
        /// <returns>Dictionary of unscaled gradient tensors</returns>
        public static Dictionary<string, Tensor> Unscale(
            Dictionary<string, Tensor> gradients,
            float scale)
        {
            if (gradients == null)
                throw new System.ArgumentNullException(nameof(gradients));

            if (scale <= 0)
                throw new System.ArgumentException("Scale must be positive", nameof(scale));

            if (scale == 1.0f)
                return gradients;

            var result = new Dictionary<string, Tensor>();
            foreach (var (name, grad) in gradients)
            {
                result[name] = Unscale(grad, scale);
            }
            return result;
        }

        /// <summary>
        /// In-place unscale of a gradient tensor
        /// </summary>
        /// <param name="gradient">The gradient tensor to unscale in-place</param>
        /// <param name="scale">The scale factor to divide by</param>
        public static void UnscaleInPlace(Tensor gradient, float scale)
        {
            if (gradient == null)
                throw new System.ArgumentNullException(nameof(gradient));

            if (scale <= 0)
                throw new System.ArgumentException("Scale must be positive", nameof(scale));

            if (scale == 1.0f)
                return;

            float inverseScale = 1.0f / scale;
            int totalElements = gradient.Size;

            // Modify tensor data directly
            var data = gradient.Data;
            for (int i = 0; i < totalElements; i++)
            {
                data[i] *= inverseScale;
            }
        }

        /// <summary>
        /// Checks for overflow (Inf/NaN) in a gradient tensor
        /// </summary>
        /// <param name="gradient">The gradient tensor to check</param>
        /// <returns>True if overflow detected, false otherwise</returns>
        public static bool CheckOverflow(Tensor gradient)
        {
            if (gradient == null)
                throw new System.ArgumentNullException(nameof(gradient));

            return IsInf(gradient) || IsNaN(gradient);
        }

        /// <summary>
        /// Checks for overflow (Inf/NaN) in multiple gradient tensors
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <returns>True if any overflow detected, false otherwise</returns>
        public static bool CheckOverflow(Dictionary<string, Tensor> gradients)
        {
            if (gradients == null)
                throw new System.ArgumentNullException(nameof(gradients));

            // Early exit on first overflow
            foreach (var grad in gradients.Values)
            {
                if (CheckOverflow(grad))
                {
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Checks for overflow (Inf/NaN) in an array of tensors
        /// </summary>
        /// <param name="tensors">Array of tensors to check</param>
        /// <returns>True if any overflow detected, false otherwise</returns>
        public static bool CheckOverflow(Tensor[] tensors)
        {
            if (tensors == null)
                throw new System.ArgumentNullException(nameof(tensors));

            // Early exit on first overflow
            foreach (var tensor in tensors)
            {
                if (tensor != null && CheckOverflow(tensor))
                {
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Checks if a tensor contains Inf values
        /// </summary>
        /// <param name="tensor">The tensor to check</param>
        /// <returns>True if any Inf values, false otherwise</returns>
        public static bool IsInf(Tensor tensor)
        {
            if (tensor == null)
                throw new System.ArgumentNullException(nameof(tensor));

            var data = tensor.Data;
            for (int i = 0; i < data.Length; i++)
            {
                if (float.IsInfinity(data[i]))
                {
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Checks if a tensor contains NaN values
        /// </summary>
        /// <param name="tensor">The tensor to check</param>
        /// <returns>True if any NaN values, false otherwise</returns>
        public static bool IsNaN(Tensor tensor)
        {
            if (tensor == null)
                throw new System.ArgumentNullException(nameof(tensor));

            var data = tensor.Data;
            for (int i = 0; i < data.Length; i++)
            {
                if (float.IsNaN(data[i]))
                {
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Checks if a tensor contains any Inf or NaN values
        /// </summary>
        /// <param name="tensor">The tensor to check</param>
        /// <returns>True if any Inf or NaN values, false otherwise</returns>
        public static bool IsInfOrNaN(Tensor tensor)
        {
            return IsInf(tensor) || IsNaN(tensor);
        }

        /// <summary>
        /// Finds tensors with overflow in a dictionary
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <returns>List of parameter names with overflow</returns>
        public static System.Collections.Generic.List<string> FindOverflowGradients(
            Dictionary<string, Tensor> gradients)
        {
            if (gradients == null)
                throw new System.ArgumentNullException(nameof(gradients));

            var overflowParams = new System.Collections.Generic.List<string>();

            foreach (var (name, grad) in gradients)
            {
                if (CheckOverflow(grad))
                {
                    overflowParams.Add(name);
                }
            }

            return overflowParams;
        }

        /// <summary>
        /// Gets statistics about gradient overflow
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <returns>Overflow statistics</returns>
        public static OverflowStats GetOverflowStats(Dictionary<string, Tensor> gradients)
        {
            if (gradients == null)
                throw new System.ArgumentNullException(nameof(gradients));

            int totalCount = gradients.Count;
            int overflowCount = 0;
            var overflowParams = new System.Collections.Generic.List<string>();

            foreach (var (name, grad) in gradients)
            {
                if (CheckOverflow(grad))
                {
                    overflowCount++;
                    overflowParams.Add(name);
                }
            }

            return new OverflowStats(totalCount, overflowCount, overflowParams);
        }
    }
}
