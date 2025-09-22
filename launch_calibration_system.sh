#!/bin/bash
# Launch script for Online Calibration Support Vector System
# Integrated with URD (Universal Robots Daemon)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/online_calibration_support_vector.py"
CONFIG_FILE="$SCRIPT_DIR/config/calibration_config.yaml"
URD_CONFIG="$SCRIPT_DIR/config/hw_config.yaml"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"
LOG_DIR="$SCRIPT_DIR/logs"
DATA_DIR="$SCRIPT_DIR/calibration_data"

# Default parameters
CAMERA_ID=0
ROBOT_IP="localhost"
SIMULATION_MODE=false
SKIP_CALIBRATION=false
VERBOSE=false

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Launch the Online Calibration Support Vector System for Universal Robots.

OPTIONS:
    -c, --camera-id ID       Camera device ID (default: 0)
    -r, --robot-ip IP        Robot IP address (default: localhost)
    -s, --simulation         Run in simulation mode
    -n, --no-calibration     Skip initial calibration
    -v, --verbose            Enable verbose logging
    -h, --help              Show this help message

EXAMPLES:
    $0                                    # Run with default settings
    $0 -c 1 -r 192.168.1.100             # Use camera 1 and real robot
    $0 -s -n                             # Simulation mode, skip calibration
    $0 --verbose --camera-id 0            # Verbose mode with camera 0

REQUIREMENTS:
    - Python 3.7+
    - OpenCV
    - scikit-learn
    - Universal Robots URD daemon
    - Camera connected (unless simulation mode)

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--camera-id)
                CAMERA_ID="$2"
                shift 2
                ;;
            -r|--robot-ip)
                ROBOT_IP="$2"
                shift 2
                ;;
            -s|--simulation)
                SIMULATION_MODE=true
                shift
                ;;
            -n|--no-calibration)
                SKIP_CALIBRATION=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 7 ]]; then
        print_error "Python 3.7+ is required (found: $PYTHON_VERSION)"
        exit 1
    fi
    
    print_success "Python $PYTHON_VERSION found"
    
    # Check if URD daemon executable exists
    if [[ ! -f "$SCRIPT_DIR/target/release/urd" ]]; then
        print_warning "URD daemon not found at target/release/urd"
        print_status "Building URD daemon..."
        cd "$SCRIPT_DIR"
        cargo build --release || {
            print_error "Failed to build URD daemon"
            exit 1
        }
    fi
    
    print_success "URD daemon found"
    
    # Check camera (unless simulation mode)
    if [[ "$SIMULATION_MODE" == false ]]; then
        print_status "Checking camera access..."
        if python3 -c "import cv2; cap = cv2.VideoCapture($CAMERA_ID); ret, frame = cap.read(); cap.release(); exit(0 if ret else 1)" 2>/dev/null; then
            print_success "Camera $CAMERA_ID accessible"
        else
            print_warning "Camera $CAMERA_ID not accessible - continuing anyway"
        fi
    fi
}

# Setup Python environment
setup_python_env() {
    print_status "Setting up Python environment..."
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "$SCRIPT_DIR/venv" ]]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv "$SCRIPT_DIR/venv"
    fi
    
    # Activate virtual environment
    source "$SCRIPT_DIR/venv/bin/activate"
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    # Install requirements
    if [[ -f "$REQUIREMENTS_FILE" ]]; then
        print_status "Installing Python dependencies..."
        pip install -r "$REQUIREMENTS_FILE"
        print_success "Dependencies installed"
    else
        print_warning "Requirements file not found: $REQUIREMENTS_FILE"
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p "$LOG_DIR"
    mkdir -p "$DATA_DIR"
    mkdir -p "$SCRIPT_DIR/config"
    
    print_success "Directories created"
}

# Update configuration files
update_config() {
    print_status "Updating configuration..."
    
    # Update robot IP in URD config if different from localhost
    if [[ "$ROBOT_IP" != "localhost" ]]; then
        if [[ -f "$URD_CONFIG" ]]; then
            # Create backup
            cp "$URD_CONFIG" "$URD_CONFIG.backup"
            
            # Update IP address
            sed -i.tmp "s/host: \".*\"/host: \"$ROBOT_IP\"/" "$URD_CONFIG"
            rm "$URD_CONFIG.tmp"
            
            print_success "Updated robot IP to $ROBOT_IP in URD config"
        else
            print_warning "URD config file not found: $URD_CONFIG"
        fi
    fi
    
    # Update calibration config
    if [[ -f "$CONFIG_FILE" ]]; then
        # Create backup
        cp "$CONFIG_FILE" "$CONFIG_FILE.backup"
        
        # Update camera ID
        sed -i.tmp "s/device_id: [0-9]*/device_id: $CAMERA_ID/" "$CONFIG_FILE"
        rm "$CONFIG_FILE.tmp"
        
        print_success "Updated camera ID to $CAMERA_ID in calibration config"
    else
        print_warning "Calibration config file not found: $CONFIG_FILE"
    fi
}

# Start the system
start_system() {
    print_status "Starting Online Calibration Support Vector System..."
    
    # Setup log file
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$LOG_DIR/calibration_${TIMESTAMP}.log"
    
    # Activate Python environment
    source "$SCRIPT_DIR/venv/bin/activate"
    
    # Set environment variables
    export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
    
    # Set log level
    if [[ "$VERBOSE" == true ]]; then
        export LOG_LEVEL="DEBUG"
    else
        export LOG_LEVEL="INFO"
    fi
    
    # Set simulation mode
    if [[ "$SIMULATION_MODE" == true ]]; then
        export SIMULATION_MODE="true"
    fi
    
    # Set skip calibration
    if [[ "$SKIP_CALIBRATION" == true ]]; then
        export SKIP_CALIBRATION="true"
    fi
    
    print_success "Environment configured"
    print_status "Log file: $LOG_FILE"
    print_status "Camera ID: $CAMERA_ID"
    print_status "Robot IP: $ROBOT_IP"
    print_status "Simulation mode: $SIMULATION_MODE"
    print_status "Skip calibration: $SKIP_CALIBRATION"
    
    echo
    print_status "Starting system... (Press Ctrl+C to stop)"
    echo
    
    # Start Python script
    cd "$SCRIPT_DIR"
    python3 "$PYTHON_SCRIPT" 2>&1 | tee "$LOG_FILE"
}

# Cleanup function
cleanup() {
    print_status "Cleaning up..."
    
    # Kill any background processes
    pkill -f "urd" 2>/dev/null || true
    pkill -f "online_calibration_support_vector.py" 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Main function
main() {
    # Set up signal handling
    trap cleanup EXIT INT TERM
    
    echo "=========================================="
    echo "Online Calibration Support Vector System"
    echo "Universal Robots Integration"
    echo "=========================================="
    echo
    
    # Parse arguments
    parse_arguments "$@"
    
    # Check requirements
    check_requirements
    
    # Setup environment
    setup_python_env
    
    # Create directories
    create_directories
    
    # Update configuration
    update_config
    
    # Start system
    start_system
}

# Run main function
main "$@"