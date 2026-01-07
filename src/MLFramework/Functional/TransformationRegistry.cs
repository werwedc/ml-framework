using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;

namespace MLFramework.Functional
{
    /// <summary>
    /// Registry for tracking functional transformations applied to delegates.
    /// Maintains a thread-safe mapping of delegates to their applied transformations.
    /// </summary>
    public class TransformationRegistry
    {
        private static readonly ConcurrentDictionary<Delegate, List<IFunctionalTransformation>> _transformations =
            new ConcurrentDictionary<Delegate, List<IFunctionalTransformation>>();

        /// <summary>
        /// Registers a transformation for a given delegate.
        /// Multiple transformations can be registered for the same delegate.
        /// </summary>
        /// <param name="original">The original delegate.</param>
        /// <param name="transform">The transformation to register.</param>
        /// <exception cref="ArgumentNullException">Thrown when original or transform is null.</exception>
        public static void Register(Delegate original, IFunctionalTransformation transform)
        {
            if (original == null)
                throw new ArgumentNullException(nameof(original));

            if (transform == null)
                throw new ArgumentNullException(nameof(transform));

            _transformations.AddOrUpdate(
                original,
                _ => new List<IFunctionalTransformation> { transform },
                (_, list) =>
                {
                    lock (list)
                    {
                        // Avoid duplicate registrations
                        if (!list.Any(t => t.Name == transform.Name && t.Type == transform.Type))
                        {
                            list.Add(transform);
                        }
                    }
                    return list;
                });
        }

        /// <summary>
        /// Gets all transformations registered for a given delegate.
        /// </summary>
        /// <param name="original">The original delegate.</param>
        /// <returns>A list of transformations, or an empty list if none are registered.</returns>
        public static List<IFunctionalTransformation> GetTransformations(Delegate original)
        {
            if (_transformations.TryGetValue(original, out var transforms))
            {
                return new List<IFunctionalTransformation>(transforms);
            }
            return new List<IFunctionalTransformation>();
        }

        /// <summary>
        /// Gets transformations of a specific type for a given delegate.
        /// </summary>
        /// <param name="original">The original delegate.</param>
        /// <param name="type">The type of transformations to retrieve.</param>
        /// <returns>A list of transformations of the specified type.</returns>
        public static List<IFunctionalTransformation> GetTransformationsByType(Delegate original, TransformationType type)
        {
            return GetTransformations(original)
                .Where(t => t.Type == type)
                .ToList();
        }

        /// <summary>
        /// Checks if a delegate has any registered transformations.
        /// </summary>
        /// <param name="original">The original delegate.</param>
        /// <returns>True if transformations are registered, false otherwise.</returns>
        public static bool HasTransformations(Delegate original)
        {
            return _transformations.ContainsKey(original);
        }

        /// <summary>
        /// Checks if a delegate has any transformations of a specific type.
        /// </summary>
        /// <param name="original">The original delegate.</param>
        /// <param name="type">The type to check for.</param>
        /// <returns>True if transformations of the specified type are registered, false otherwise.</returns>
        public static bool HasTransformationOfType(Delegate original, TransformationType type)
        {
            return GetTransformationsByType(original, type).Any();
        }

        /// <summary>
        /// Removes all transformations for a specific delegate.
        /// </summary>
        /// <param name="original">The original delegate.</param>
        /// <returns>True if transformations were removed, false otherwise.</returns>
        public static bool Unregister(Delegate original)
        {
            return _transformations.TryRemove(original, out _);
        }

        /// <summary>
        /// Clears all registered transformations.
        /// </summary>
        public static void Clear()
        {
            _transformations.Clear();
        }

        /// <summary>
        /// Gets the total number of registered delegates.
        /// </summary>
        /// <returns>The count of registered delegates.</returns>
        public static int Count()
        {
            return _transformations.Count;
        }

        /// <summary>
        /// Gets the total number of registered transformations across all delegates.
        /// </summary>
        /// <returns>The total count of transformations.</returns>
        public static int TotalTransformationCount()
        {
            return _transformations.Values.Sum(list => list.Count);
        }
    }
}
