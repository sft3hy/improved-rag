#!/usr/bin/env python3
"""
Setup script for Advanced RAG Pipeline
This script helps install dependencies and check system requirements
"""

import subprocess
import sys
import os
import importlib.util

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True

def check_ollama():
    """Check if Ollama is installed and running"""
    print("\n🤖 Checking Ollama installation...")
    
    # Check if ollama command exists
    if not run_command("ollama --version", "Checking Ollama installation"):
        print("❌ Ollama is not installed. Please install from https://ollama.ai")
        return False
    
    # Check if Ollama is running
    if not run_command("ollama list", "Checking if Ollama is running"):
        print("❌ Ollama is not running. Please start Ollama service")
        return False
    
    return True

def install_required_models():
    """Install required Ollama models"""
    models = ["nomic-embed-text", "llama3.1:8b"]
    
    print("\n📥 Installing required Ollama models...")
    for model in models:
        if not run_command(f"ollama pull {model}", f"Installing {model}"):
            print(f"❌ Failed to install {model}")
            return False
    
    return True

def install_python_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    # Core dependencies
    core_deps = [
        "streamlit>=1.28.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.20",
        "langchain-core>=0.1.0",
        "langchain-ollama>=0.1.0",
        "langchain-chroma>=0.1.0",
        "langchain-text-splitters>=0.0.1",
        "chromadb>=0.4.0",
        "pandas>=1.5.0"
    ]
    
    # Document processing dependencies
    doc_deps = [
        "PyPDF2>=3.0.0",
        "python-docx>=0.8.11",
        "python-pptx>=0.6.21",
        "openpyxl>=3.0.10"
    ]
    
    # Install core dependencies
    for dep in core_deps:
        if not run_command(f"pip install '{dep}'", f"Installing {dep.split('>=')[0]}"):
            print(f"❌ Failed to install {dep}")
            return False
    
    # Install document processing dependencies (these are optional)
    print("\n📄 Installing document processing dependencies...")
    for dep in doc_deps:
        if not run_command(f"pip install '{dep}'", f"Installing {dep.split('>=')[0]}"):
            print(f"⚠️  Warning: Failed to install {dep} (optional)")
    
    return True

def check_installed_packages():
    """Check which packages are successfully installed"""
    packages = {
        'streamlit': 'Streamlit',
        'langchain': 'LangChain',
        'langchain_ollama': 'LangChain Ollama',
        'chromadb': 'ChromaDB',
        'pandas': 'Pandas',
        'PyPDF2': 'PyPDF2 (PDF support)',
        'docx': 'python-docx (Word support)',
        'pptx': 'python-pptx (PowerPoint support)',
        'openpyxl': 'openpyxl (Excel support)'
    }
    
    print("\n📋 Checking installed packages...")
    installed = []
    missing = []
    
    for package, description in packages.items():
        try:
            importlib.import_module(package)
            installed.append(description)
            print(f"✅ {description}")
        except ImportError:
            missing.append(description)
            print(f"❌ {description}")
    
    return installed, missing

def create_test_files():
    """Create test files for the application"""
    print("\n📝 Creating test files...")
    
    # Create a sample text file
    test_content = """# Sample Document

This is a sample document for testing the RAG pipeline.

## Introduction
This document contains information about various topics to test the retrieval system.

## Technology
Artificial Intelligence and Machine Learning are transforming how we process information.

## Science
The scientific method involves observation, hypothesis formation, and testing.
"""
    
    try:
        with open("sample_document.md", "w") as f:
            f.write(test_content)
        print("✅ Created sample_document.md")
        return True
    except Exception as e:
        print(f"❌ Failed to create test files: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Advanced RAG Pipeline Setup")
    print("=" * 40)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Check Ollama
    if not check_ollama():
        success = False
    
    if success:
        # Install Python dependencies
        if not install_python_dependencies():
            success = False
        
        # Install Ollama models
        if not install_required_models():
            success = False
        
        # Check what's installed
        installed, missing = check_installed_packages()
        
        # Create test files
        create_test_files()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run the application: streamlit run main.py")
        print("2. Upload documents and start asking questions!")
        
        if missing:
            print(f"\n⚠️  Optional packages not installed: {', '.join(missing)}")
            print("Some file types may not be supported.")
    else:
        print("❌ Setup failed. Please resolve the issues above and try again.")
        print("\nCommon solutions:")
        print("- Install Ollama from https://ollama.ai")
        print("- Make sure Ollama service is running")
        print("- Check your internet connection")
        print("- Try running: pip install --upgrade pip")
    
    return success

if __name__ == "__main__":
    sys.exit(0 if main() else 1)