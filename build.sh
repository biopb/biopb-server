#!/bin/bash
# Local build script with CI-consistent tagging behavior
#
# Usage:
#   ./build.sh [service]           # Build specific service
#   ./build.sh all                  # Build all services
#   ./build.sh --no-push cellpose   # Build without tagging for push

set -e

REGISTRY="${REGISTRY:-docker.io}"
IMAGE_PREFIX="${IMAGE_PREFIX:-jiyuuchc}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

get_version() {
    local dir="$1"
    if [ -f "$dir/VERSION" ]; then
        cat "$dir/VERSION" | tr -d '[:space:]'
    else
        echo "latest"
    fi
}

get_sha() {
    git rev-parse --short HEAD 2>/dev/null || echo "local"
}

discover_services() {
    ls -1 */Dockerfile 2>/dev/null | cut -d'/' -f1 | sort -u
}

build_service() {
    local dir="$1"
    local push="$2"

    if [ ! -f "$dir/Dockerfile" ]; then
        log_error "$dir/Dockerfile not found"
        return 1
    fi

    local version=$(get_version "$dir")
    local sha=$(get_sha)
    local image_base="$REGISTRY/$IMAGE_PREFIX/$dir"

    log_info "Building $dir"
    echo "  Version: $version"
    echo "  SHA: $sha"
    echo "  Image: $image_base"

    # Build tags (consistent with CI)
    local tags="--tag $image_base:$sha --tag $image_base:$version --tag $image_base:latest"

    # Build args
    local build_args=""
    if [ -n "$version" ] && [ "$version" != "latest" ]; then
        build_args="--build-arg VERSION=$version"
    fi

    docker buildx build \
        --platform linux/amd64 \
        $tags \
        $build_args \
        --load \
        "$dir"

    if [ "$push" = "true" ]; then
        log_info "Pushing $dir..."
        docker push "$image_base:$version"
        docker push "$image_base:latest"
        docker push "$image_base:$sha"
    fi

    log_info "Built $image_base:$sha"
}

# Parse arguments
PUSH="true"
SERVICES=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-push)
            PUSH="false"
            shift
            ;;
        --push)
            PUSH="true"
            shift
            ;;
        all)
            SERVICES=$(discover_services)
            shift
            ;;
        -*)
            log_error "Unknown option: $1"
            echo "Usage: $0 [--no-push] [service|all]"
            exit 1
            ;;
        *)
            SERVICES+=("$1")
            shift
            ;;
    esac
done

# Default to all if no service specified
if [ ${#SERVICES[@]} -eq 0 ]; then
    log_info "No service specified, building all..."
    SERVICES=$(discover_services)
fi

# Build each service
for service in $SERVICES; do
    build_service "$service" "$PUSH"
done

log_info "Done!"