using MLFramework.Core;
using System;

namespace MLFramework.Amp
{
    /// <summary>
    /// Attribute to mark methods as AMP-aware
    /// </summary>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Class)]
    public class AmpEnabledAttribute : Attribute
    {
        /// <summary>
        /// Gets or sets the default forward precision for this method
        /// </summary>
        public DataType? DefaultForwardDtype { get; set; }

        /// <summary>
        /// Gets or sets the default backward precision for this method
        /// </summary>
        public DataType? DefaultBackwardDtype { get; set; }

        /// <summary>
        /// Creates a new AmpEnabledAttribute
        /// </summary>
        public AmpEnabledAttribute()
        {
        }

        /// <summary>
        /// Creates a new AmpEnabledAttribute with specified precision
        /// </summary>
        /// <param name="forwardDtype">The default forward precision</param>
        /// <param name="backwardDtype">The default backward precision</param>
        public AmpEnabledAttribute(DataType forwardDtype, DataType backwardDtype)
        {
            DefaultForwardDtype = forwardDtype;
            DefaultBackwardDtype = backwardDtype;
        }
    }
}
