$ErrorActionPreference = "Stop"

Write-Host "Building Mobile Runtime..." -ForegroundColor Green

# Build all targets
dotnet build src/MobileRuntime/MobileRuntime.csproj --configuration Release

# Build tests
dotnet build tests/MobileRuntime.Tests/MobileRuntime.Tests.csproj --configuration Release

# Build converter tool
dotnet build tools/MobileModelConverter/MobileModelConverter.csproj --configuration Release

# Run tests
Write-Host "Running tests..." -ForegroundColor Yellow
dotnet test tests/MobileRuntime.Tests/MobileRuntime.Tests.csproj --configuration Release --no-build

# Create NuGet package
Write-Host "Creating NuGet package..." -ForegroundColor Yellow
dotnet pack src/MobileRuntime/MobileRuntime.csproj --configuration Release --no-build

Write-Host "Build completed successfully!" -ForegroundColor Green
