# Build and bring up services
cd $PSScriptRoot/..
docker-compose up -d --build
Write-Host "Waiting 10s for services to start..."
Start-Sleep -s 10

# Run DB init inside container or locally
if (Get-Command docker -ErrorAction SilentlyContinue) {
    Write-Host "Initializing DB and MinIO bucket (container)..."
    docker run --rm -v ${PWD}:/app -w /app python:3.11-slim pwsh -Command "python migrations/init_db.py"
} else {
    Write-Host "Docker not found; please run migrations/init_db.py locally in Python environment"
}

Write-Host "To test the app upload endpoint (replace sample.jpg with your file):"
Write-Host "curl -F \"image=@tests/sample.jpg\" http://localhost:5088/api/segment -v"

Write-Host "To call model worker directly (sample):"
Write-Host "curl -F \"image=@tests/sample.jpg\" -F \"variants=4\" http://localhost:7860/infer_enhanced -v"
