using System;

namespace MLFramework.IR.Transformations
{
    /// <summary>
    /// Instrumentation for tracking IR transformation pass execution
    /// </summary>
    public class PassInstrumentation
    {
        /// <summary>
        /// Event fired before a pass runs
        /// </summary>
        public event Action<IRTransformation, HLIRModule> BeforePass;

        /// <summary>
        /// Event fired after a pass runs successfully
        /// </summary>
        public event Action<IRTransformation, HLIRModule> AfterPass;

        /// <summary>
        /// Event fired when a pass encounters an error
        /// </summary>
        public event Action<IRTransformation, HLIRModule, Exception> OnPassError;

        /// <summary>
        /// Notifies listeners that a pass is about to run
        /// </summary>
        /// <param name="pass">The pass that is about to run</param>
        /// <param name="module">The module the pass will run on</param>
        public void NotifyBeforePass(IRTransformation pass, HLIRModule module)
        {
            if (pass == null)
                throw new ArgumentNullException(nameof(pass));
            if (module == null)
                throw new ArgumentNullException(nameof(module));

            BeforePass?.Invoke(pass, module);
        }

        /// <summary>
        /// Notifies listeners that a pass has completed successfully
        /// </summary>
        /// <param name="pass">The pass that completed</param>
        /// <param name="module">The module the pass ran on</param>
        public void NotifyAfterPass(IRTransformation pass, HLIRModule module)
        {
            if (pass == null)
                throw new ArgumentNullException(nameof(pass));
            if (module == null)
                throw new ArgumentNullException(nameof(module));

            AfterPass?.Invoke(pass, module);
        }

        /// <summary>
        /// Notifies listeners that a pass encountered an error
        /// </summary>
        /// <param name="pass">The pass that encountered an error</param>
        /// <param name="module">The module the pass was running on</param>
        /// <param name="ex">The exception that was thrown</param>
        public void NotifyPassError(IRTransformation pass, HLIRModule module, Exception ex)
        {
            if (pass == null)
                throw new ArgumentNullException(nameof(pass));
            if (module == null)
                throw new ArgumentNullException(nameof(module));
            if (ex == null)
                throw new ArgumentNullException(nameof(ex));

            OnPassError?.Invoke(pass, module, ex);
        }

        /// <summary>
        /// Checks if any listeners are registered
        /// </summary>
        public bool HasListeners =>
            BeforePass != null || AfterPass != null || OnPassError != null;

        /// <summary>
        /// Clears all event listeners
        /// </summary>
        public void ClearListeners()
        {
            BeforePass = null;
            AfterPass = null;
            OnPassError = null;
        }
    }
}
