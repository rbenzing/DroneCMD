# DroneCmd Enhanced Library - Installation Guide

This guide covers different installation options for the DroneCmd Enhanced Library, from minimal installations to full development setups.

## Quick Start

For most users who want the full functionality:

```bash
pip install -r requirements.txt
```

## Installation Options

### 1. Minimal Installation (Core Features Only)

**Use case**: Basic SDR capture, signal processing, and protocol classification with RTL-SDR only.

```bash
pip install -r requirements-minimal.txt
```

**Includes**:
- RTL-SDR support
- Signal processing and demodulation
- Protocol classification
- FHSS functionality
- Basic CLI tools

**Size**: ~200MB of dependencies

### 2. Standard Installation (Recommended)

**Use case**: Full functionality including extended SDR support, visualization, and optional features.

```bash
pip install -r requirements.txt
```

**Includes**:
- All minimal features
- Extended SDR hardware support (HackRF, etc.)
- Visualization tools (matplotlib)
- Cryptography features
- Advanced signal analysis
- Documentation tools

**Size**: ~500MB of dependencies

### 3. Development Installation

**Use case**: Contributors and developers who need testing, linting, and documentation tools.

```bash
pip install -r requirements-dev.txt
```

**Includes**:
- All standard features
- Testing frameworks (pytest, coverage)
- Code quality tools (black, flake8, mypy)
- Documentation generation (sphinx)
- Debugging and profiling tools
- Jupyter notebook support

**Size**: ~800MB of dependencies

### 4. Custom Hardware Support

**Use case**: Installing support for specific SDR hardware only.

```bash
# Base installation
pip install -r requirements-minimal.txt

# Add specific SDR support
pip install pyhackrf                    # HackRF support
pip install SoapySDR                    # Universal SDR support
```

See `requirements-sdr.txt` for detailed hardware support options.

## Python Environment Setup

### Using Virtual Environments (Recommended)

```bash
# Create virtual environment
python -m venv dronecmd-env

# Activate (Linux/macOS)
source dronecmd-env/bin/activate

# Activate (Windows)
dronecmd-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Deactivate when done
deactivate
```

### Using Conda

```bash
# Create conda environment
conda create -n dronecmd python=3.11

# Activate environment
conda activate dronecmd

# Install dependencies
pip install -r requirements.txt
```

### Using Poetry (Alternative)

```bash
# Install poetry if not already installed
pip install poetry

# Install dependencies from pyproject.toml
poetry install

# Activate shell
poetry shell
```

## System Dependencies

Some SDR hardware requires system-level drivers and libraries.

### Ubuntu/Debian

```bash
# Update package list
sudo apt update

# RTL-SDR support
sudo apt install librtlsdr-dev rtl-sdr

# HackRF support  
sudo apt install libhackrf-dev hackrf

# Airspy support
sudo apt install libairspy-dev airspy

# USRP support
sudo apt install libuhd-dev uhd-host

# PlutoSDR support
sudo apt install libiio-dev libad9361-dev
```

### macOS (via Homebrew)

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install SDR libraries
brew install librtlsdr hackrf airspy

# USRP support
brew install uhd
```

### Windows

1. Install [Zadig](https://zadig.akeo.ie/) for USB driver management
2. Download vendor-specific drivers:
   - [RTL-SDR drivers](https://www.rtl-sdr.com/rtl-sdr-quick-start-guide/)
   - [HackRF drivers](https://github.com/greatscottgadgets/hackrf/releases)
   - [Airspy drivers](https://airspy.com/download/)

## USB Permissions (Linux)

Linux users need proper USB permissions for SDR devices:

```bash
# Download and install udev rules
sudo wget -O /etc/udev/rules.d/20-rtlsdr.rules \
  https://raw.githubusercontent.com/osmocom/rtl-sdr/master/rtl-sdr.rules

sudo wget -O /etc/udev/rules.d/53-hackrf.rules \
  https://raw.githubusercontent.com/greatscottgadgets/hackrf/master/host/libhackrf/53-hackrf.rules

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Add user to plugdev group (logout/login required)
sudo usermod -a -G plugdev $USER
```

## Verification

Test your installation:

```python
# Test core functionality
python -c "import dronecmd; print('DroneCmd:', dronecmd.__version__)"

# Test SDR support
python -c "from rtlsdr import RtlSdr; print('RTL-SDR: OK')"

# Test signal processing
python -c "from dronecmd.core.signal_processing import SignalProcessor; print('Signal Processing: OK')"

# Test FHSS
python -c "from dronecmd.core.fhss import SimpleFHSS; print('FHSS: OK')"
```

## Installation Troubleshooting

### Common Issues

**1. "No module named 'rtlsdr'"**
```bash
# Install RTL-SDR system libraries first
sudo apt install librtlsdr-dev  # Ubuntu/Debian
brew install librtlsdr          # macOS

# Then reinstall Python package
pip uninstall pyrtlsdr
pip install pyrtlsdr
```

**2. "Microsoft Visual C++ 14.0 is required" (Windows)**
```bash
# Install Microsoft C++ Build Tools
# Or install Visual Studio Community with C++ workload
# Or use conda-forge packages:
conda install -c conda-forge pyrtlsdr
```

**3. "Permission denied" for USB devices (Linux)**
```bash
# Check device permissions
lsusb
ls -l /dev/bus/usb/

# Install udev rules (see USB Permissions section above)
# Or run with sudo (not recommended for development)
```

**4. "ImportError: No module named '_ctypes'" (Some Linux)**
```bash
# Install development libraries
sudo apt install libffi-dev python3-dev
```

### Performance Optimization

**For better performance**:

```bash
# Install optimized BLAS libraries
pip install numpy[blas]

# Or use conda with MKL
conda install numpy scipy scikit-learn intel-openmp mkl
```

**For memory-constrained systems**:

```bash
# Use minimal installation
pip install -r requirements-minimal.txt

# Disable optional features in code:
# Set enable_enhanced_features=False in configurations
```

## Feature-Specific Installation

### Jupyter Notebook Support

```bash
pip install "dronecmd[jupyter]"
# or
pip install jupyter jupyterlab ipywidgets
```

### GUI Components

```bash
pip install "dronecmd[gui]"
# or  
pip install PyQt5 pyqtgraph
```

### Visualization Tools

```bash
pip install "dronecmd[viz]"
# or
pip install matplotlib seaborn plotly pandas
```

### All Optional Features

```bash
pip install "dronecmd[all]"
```

## Docker Installation (Alternative)

For a containerized environment:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    librtlsdr-dev \
    libhackrf-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY dronecmd/ /app/dronecmd/
WORKDIR /app

CMD ["python", "-m", "dronecmd.cli"]
```

```bash
# Build and run
docker build -t dronecmd .
docker run --privileged -v /dev/bus/usb:/dev/bus/usb dronecmd
```

## Update Instructions

### Updating Dependencies

```bash
# Update to latest compatible versions
pip install -r requirements.txt --upgrade

# Update specific package
pip install --upgrade numpy scipy scikit-learn

# Check for outdated packages
pip list --outdated
```

### Version Compatibility

- **Python**: 3.8+ (3.11+ recommended)
- **NumPy**: 1.24+ for performance improvements
- **SciPy**: 1.10+ for enhanced signal processing
- **scikit-learn**: 1.3+ for latest ML algorithms

## Getting Help

If you encounter issues:

1. Check the [troubleshooting section](#installation-troubleshooting)
2. Review system dependencies for your platform
3. Test with minimal installation first
4. Check [GitHub issues](https://github.com/your-org/dronecmd/issues)
5. Join our [community discussions](https://github.com/your-org/dronecmd/discussions)

## Next Steps

After successful installation:

1. Read the [Quick Start Guide](README.md#quick-start)
2. Try the [examples](examples/)
3. Review the [API documentation](docs/)
4. Join the [community](CONTRIBUTING.md)