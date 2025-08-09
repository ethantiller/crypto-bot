#!/usr/bin/env python3
"""
Test Runner for Crypto Trading Bot
Simple script to run all tests and show results
"""

import subprocess
import sys
import os

def run_tests():
    """Run the trading bot tests"""
    print("🧪 Running Crypto Trading Bot Tests...")
    print("=" * 50)
    
    # Make sure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        # Run the tests
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 'test_trading_bot.py', 
            '-v', '--tb=short', '--color=yes'
        ], capture_output=True, text=True, timeout=300)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        if result.returncode == 0:
            print("\n🎉 All tests passed!")
            return True
        else:
            print(f"\n❌ Tests failed with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Tests timed out after 5 minutes")
        return False
    except FileNotFoundError:
        print("⚠️  pytest not found, trying unittest...")
        return run_unittest()
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def run_unittest():
    """Fallback to unittest if pytest is not available"""
    try:
        result = subprocess.run([
            sys.executable, 'test_trading_bot.py'
        ], capture_output=True, text=True, timeout=300)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        if result.returncode == 0:
            print("\n🎉 All tests passed!")
            return True
        else:
            print(f"\n❌ Tests failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running unittest: {e}")
        return False

def install_test_dependencies():
    """Install required test dependencies"""
    print("📦 Installing test dependencies...")
    
    dependencies = ['pytest', 'pytest-mock']
    
    for dep in dependencies:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                         check=True, capture_output=True)
            print(f"✅ Installed {dep}")
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to install {dep}")

if __name__ == "__main__":
    print("🚀 Crypto Trading Bot Test Suite")
    print("=" * 40)
    
    # Check if pytest is available, if not install it
    try:
        import pytest
        print("✅ pytest found")
    except ImportError:
        print("⚠️  pytest not found, installing...")
        install_test_dependencies()
    
    # Run the tests
    success = run_tests()
    
    if success:
        print("\n✨ Test suite completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Test suite failed!")
        sys.exit(1)
