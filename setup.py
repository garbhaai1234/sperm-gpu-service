"""
Setup script for the Sperm Quality Analysis System
Handles installation, configuration, and initial testing
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path

def setup_logging():
    """Setup logging for the setup script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def check_python_version():
    """Check if Python version is compatible"""
    logger = logging.getLogger(__name__)
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    
    logger.info(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_system_requirements():
    """Check system requirements"""
    logger = logging.getLogger(__name__)
    
    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 8:
            logger.warning(f"Low memory detected: {memory_gb:.1f}GB (8GB+ recommended)")
        else:
            logger.info(f"✓ Available memory: {memory_gb:.1f}GB")
    except ImportError:
        logger.warning("psutil not available, cannot check memory")
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"✓ CUDA available: {gpu_count} GPU(s) - {gpu_name}")
        else:
            logger.info("ℹ CUDA not available, will use CPU")
    except ImportError:
        logger.warning("PyTorch not installed, cannot check CUDA")
    
    return True

def install_dependencies():
    """Install required dependencies"""
    logger = logging.getLogger(__name__)
    
    logger.info("Installing dependencies...")
    
    try:
        # Install from requirements file
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "enhanced_requirements.txt"
        ])
        logger.info("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def check_model_files():
    """Check if model files are present"""
    logger = logging.getLogger(__name__)
    
    model_files = {
        'detection_best.pt': 'YOLOv8 detection model',
        'best_resnet50_transfer_from_101.pth': 'Mask R-CNN segmentation model',
        'HNK_best.pt': 'Head-Neck-Tail segmentation model'
    }
    
    missing_files = []
    for filename, description in model_files.items():
        if os.path.exists(filename):
            file_size = os.path.getsize(filename) / (1024**2)  # MB
            logger.info(f"✓ {description}: {filename} ({file_size:.1f} MB)")
        else:
            missing_files.append((filename, description))
            logger.error(f"✗ Missing: {filename} ({description})")
    
    if missing_files:
        logger.error("Required model files are missing!")
        logger.error("Please ensure all model files are in the current directory:")
        for filename, description in missing_files:
            logger.error(f"  - {filename} ({description})")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    logger = logging.getLogger(__name__)
    
    directories = ['uploads', 'outputs', 'logs']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"✓ Created directory: {directory}")
    
    return True

def run_tests():
    """Run basic tests to verify installation"""
    logger = logging.getLogger(__name__)
    
    logger.info("Running installation tests...")
    
    try:
        # Test imports
        from sperm_pipeline import SpermAnalysisPipeline
        logger.info("✓ Pipeline imports successful")
        
        # Test individual components
        from sperm_pipeline import SpermDetector, SpermTracker, SpermSegmentation
        from sperm_pipeline import MorphologyAnalyzer, MotilityAnalyzer
        logger.info("✓ Component imports successful")
        
        # Run comprehensive tests if model files exist
        if check_model_files():
            logger.info("Running comprehensive tests...")
            result = subprocess.run([sys.executable, "test_pipeline.py"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✓ All tests passed!")
                return True
            else:
                logger.error("✗ Some tests failed:")
                logger.error(result.stdout)
                logger.error(result.stderr)
                return False
        else:
            logger.warning("Skipping comprehensive tests due to missing model files")
            return True
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

def create_example_script():
    """Create an example script for users"""
    logger = logging.getLogger(__name__)
    
    example_content = '''"""
Example script for Sperm Quality Analysis
Run this script to test the system with your video
"""

import os
from sperm_pipeline import SpermAnalysisPipeline

def main():
    # Model paths (update these if your models are in different locations)
    model_paths = {
        'detection': 'detection_best.pt',
        'maskrcnn': 'best_resnet50_transfer_from_101.pth',
        'hnk': 'HNK_best.pt'
    }
    
    # Check if models exist
    for name, path in model_paths.items():
        if not os.path.exists(path):
            print(f"Error: Model file not found: {path}")
            return
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = SpermAnalysisPipeline(
        detection_model_path=model_paths['detection'],
        maskrcnn_model_path=model_paths['maskrcnn'],
        hnk_model_path=model_paths['hnk'],
        device="cpu"  # Change to "cuda" if you have a GPU
    )
    
    # Process video (replace with your video path)
    video_path = "your_video.mp4"  # Update this path
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        print("Please update the video_path variable with your video file")
        return
    
    print(f"Processing video: {video_path}")
    
    def progress_callback(message, percent):
        print(f"Progress {percent}%: {message}")
    
    # Run analysis
    results = pipeline.process_video(
        video_path=video_path,
        output_dir="example_output",
        progress_callback=progress_callback
    )
    
    # Print results
    analysis_results = results['analysis_results']
    total_sperm = len(analysis_results)
    good_sperm = len([r for r in analysis_results if r['status'] == 'good'])
    
    print(f"\\nAnalysis complete!")
    print(f"Total sperm detected: {total_sperm}")
    print(f"Good quality sperm: {good_sperm} ({good_sperm/total_sperm*100:.1f}%)")
    print(f"Results saved to: example_output/")

if __name__ == "__main__":
    main()
'''
    
    with open("example_analysis.py", "w") as f:
        f.write(example_content)
    
    logger.info("✓ Created example_analysis.py")

def print_usage_instructions():
    """Print usage instructions"""
    logger = logging.getLogger(__name__)
    
    logger.info("\\n" + "="*60)
    logger.info("SETUP COMPLETE!")
    logger.info("="*60)
    logger.info("\\nTo get started:")
    logger.info("1. Web Interface:")
    logger.info("   python enhanced_flask_app.py")
    logger.info("   Then open http://localhost:5002 in your browser")
    logger.info("\\n2. Command Line:")
    logger.info("   python run_analysis.py your_video.mp4")
    logger.info("\\n3. Programmatic:")
    logger.info("   python example_analysis.py")
    logger.info("\\n4. Test the system:")
    logger.info("   python test_pipeline.py")
    logger.info("\\nFor more information, see README.md")
    logger.info("="*60)

def main():
    """Main setup function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Sperm Quality Analysis System - Setup")
    logger.info("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check system requirements
    check_system_requirements()
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Setup failed during dependency installation")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check model files
    models_ok = check_model_files()
    
    # Run tests
    if not run_tests():
        logger.warning("Some tests failed, but setup may still work")
    
    # Create example script
    create_example_script()
    
    # Print usage instructions
    print_usage_instructions()
    
    if models_ok:
        logger.info("\\n🎉 Setup completed successfully!")
        logger.info("You can now use the sperm quality analysis system.")
    else:
        logger.warning("\\n⚠️  Setup completed with warnings.")
        logger.warning("Please ensure all model files are present before using the system.")

if __name__ == "__main__":
    main()
