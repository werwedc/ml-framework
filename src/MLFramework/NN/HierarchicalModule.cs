using System;
using System.Collections.Generic;
using RitterFramework.Core.Tensor;
using MLFramework.Modules;

namespace MLFramework.NN
{
    /// <summary>
    /// Interface for modules that support hierarchical structure
    /// </summary>
    public interface IHierarchicalModule : IModule
    {
        /// <summary>
        /// Gets the module name
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Gets or sets the parent module (null if root)
        /// </summary>
        IHierarchicalModule Parent { get; set; }

        /// <summary>
        /// Gets all child modules
        /// </summary>
        IEnumerable<IHierarchicalModule> Children();

        /// <summary>
        /// Replaces a child module with a new module
        /// </summary>
        void ReplaceChild(string childName, IHierarchicalModule newModule);

        /// <summary>
        /// Adds a child module
        /// </summary>
        void AddChild(IHierarchicalModule child);

        /// <summary>
        /// Gets a child module by name
        /// </summary>
        IHierarchicalModule GetChild(string name);

        /// <summary>
        /// Gets all parameters from this module
        /// </summary>
        IEnumerable<Parameter> GetParameters();

        /// <summary>
        /// Gets all named parameters from this module
        /// </summary>
        IEnumerable<(string Name, Parameter Parameter)> GetNamedParameters();
    }

    /// <summary>
    /// Base class for hierarchical modules with child management
    /// </summary>
    public abstract class HierarchicalModule : Module, IHierarchicalModule
    {
        private readonly Dictionary<string, IHierarchicalModule> _children = new Dictionary<string, IHierarchicalModule>();
        private IHierarchicalModule _parent;

        /// <summary>
        /// Gets the parent module (null if root)
        /// </summary>
        public IHierarchicalModule Parent
        {
            get => _parent;
            set => _parent = value;
        }

        /// <summary>
        /// Gets the module type
        /// </summary>
        public string ModuleType => "HierarchicalModule";

        /// <summary>
        /// Creates a new hierarchical module
        /// </summary>
        protected HierarchicalModule(string name) : base(name)
        {
        }

        /// <summary>
        /// Gets all child modules
        /// </summary>
        public virtual IEnumerable<IHierarchicalModule> Children()
        {
            return _children.Values;
        }

        /// <summary>
        /// Replaces a child module with a new module
        /// </summary>
        public virtual void ReplaceChild(string childName, IHierarchicalModule newModule)
        {
            if (!_children.ContainsKey(childName))
                throw new ArgumentException($"Child module '{childName}' not found", nameof(childName));

            var oldChild = _children[childName];
            oldChild.Parent = null;

            _children[childName] = newModule;
            newModule.Parent = this;
        }

        /// <summary>
        /// Adds a child module
        /// </summary>
        public virtual void AddChild(IHierarchicalModule child)
        {
            if (child == null)
                throw new ArgumentNullException(nameof(child));

            _children[child.Name] = child;
            child.Parent = this;
        }

        /// <summary>
        /// Gets a child module by name
        /// </summary>
        public virtual IHierarchicalModule GetChild(string name)
        {
            _children.TryGetValue(name, out var child);
            return child;
        }

        /// <summary>
        /// Gets all parameters from this module and all children recursively
        /// </summary>
        public override IEnumerable<Parameter> GetParameters()
        {
            foreach (var child in Children())
            {
                foreach (var param in child.GetParameters())
                {
                    yield return param;
                }
            }
        }

        /// <summary>
        /// Gets all named parameters from this module and all children recursively
        /// </summary>
        public override IEnumerable<(string Name, Parameter Parameter)> GetNamedParameters()
        {
            foreach (var child in Children())
            {
                foreach (var (name, param) in child.GetNamedParameters())
                {
                    yield return ($"{Name}.{name}", param);
                }
            }
        }
    }

    /// <summary>
    /// Simple container module that holds child modules
    /// </summary>
    public class ModuleContainer : HierarchicalModule
    {
        /// <summary>
        /// Gets the module type
        /// </summary>
        public new string ModuleType => "ModuleContainer";

        /// <summary>
        /// Creates a new module container
        /// </summary>
        public ModuleContainer(string name) : base(name)
        {
        }

        /// <summary>
        /// Forward pass - containers typically don't have forward logic
        /// </summary>
        public override Tensor Forward(Tensor input)
        {
            // Containers don't perform forward operations directly
            throw new InvalidOperationException("ModuleContainer does not support forward operations. Use child modules directly.");
        }
    }
}
