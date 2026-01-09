#!/bin/bash

set -e

echo "Building Mobile Runtime..."

# Build all targets
dotnet build src/MobileRuntime/MobileRuntime.csproj --configuration Release

# Build tests
dotnet build tests/MobileRuntime.Tests/MobileRuntime.Tests.csproj --configuration Release

# Build converter tool
dotnet build tools/MobileModelConverter/MobileModelConverter.csproj --configuration Release

# Run tests
echo "Running tests..."
dotnet test tests/MobileRuntime.Tests/MobileRuntime.Tests.csproj --configuration Release --no-build

# Create NuGet package
echo "Creating NuGet package..."
dotnet pack src/MobileRuntime/MobileRuntime.csproj --configuration Release --no-build

echo "Build completed successfully!"
