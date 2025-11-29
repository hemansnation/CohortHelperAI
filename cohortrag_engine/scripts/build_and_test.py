#!/usr/bin/env python3
"""
Build and Test Script for CohortRAG Engine
==========================================

This script builds the package, runs tests, and prepares for deployment.
"""

import subprocess
import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


class PackageBuilder:
    """Build and test CohortRAG Engine package"""

    def __init__(self, root_dir: Path = None):
        self.root_dir = root_dir or Path(__file__).parent.parent
        self.dist_dir = self.root_dir / "dist"
        self.build_dir = self.root_dir / "build"

    def clean_build_artifacts(self) -> bool:
        """Clean previous build artifacts"""
        print("🧹 Cleaning build artifacts...")

        try:
            # Remove dist and build directories
            for dir_path in [self.dist_dir, self.build_dir]:
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                    print(f"   Removed {dir_path}")

            # Remove egg-info directories
            for egg_info in self.root_dir.glob("*.egg-info"):
                if egg_info.is_dir():
                    shutil.rmtree(egg_info)
                    print(f"   Removed {egg_info}")

            # Remove __pycache__ directories
            for pycache in self.root_dir.rglob("__pycache__"):
                if pycache.is_dir():
                    shutil.rmtree(pycache)

            print("✅ Build artifacts cleaned")
            return True

        except Exception as e:
            print(f"❌ Error cleaning artifacts: {e}")
            return False

    def run_code_quality_checks(self) -> bool:
        """Run code quality checks"""
        print("🔍 Running code quality checks...")

        checks = [
            ("Black formatting", ["black", "--check", "."]),
            ("Import sorting", ["isort", "--check-only", "."]),
            ("Flake8 linting", ["flake8", "--max-line-length=88", "--extend-ignore=E203,W503", "."]),
        ]

        all_passed = True

        for name, command in checks:
            try:
                print(f"   Running {name}...")
                result = subprocess.run(
                    command,
                    cwd=self.root_dir,
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if result.returncode == 0:
                    print(f"   ✅ {name} passed")
                else:
                    print(f"   ❌ {name} failed:")
                    print(f"      {result.stdout}")
                    print(f"      {result.stderr}")
                    all_passed = False

            except subprocess.TimeoutExpired:
                print(f"   ❌ {name} timed out")
                all_passed = False
            except FileNotFoundError:
                print(f"   ⚠️  {name} skipped (tool not installed)")
            except Exception as e:
                print(f"   ❌ {name} error: {e}")
                all_passed = False

        if all_passed:
            print("✅ All code quality checks passed")
        else:
            print("❌ Some code quality checks failed")

        return all_passed

    def run_tests(self) -> bool:
        """Run test suite"""
        print("🧪 Running test suite...")

        try:
            # Run pytest with coverage
            result = subprocess.run(
                [
                    "python", "-m", "pytest",
                    "tests/",
                    "-v",
                    "--tb=short",
                    "--cov=cohortrag_engine",
                    "--cov-report=term-missing",
                    "--cov-report=xml",
                    "--cov-fail-under=80"
                ],
                cwd=self.root_dir,
                timeout=600  # 10 minutes max
            )

            if result.returncode == 0:
                print("✅ All tests passed")
                return True
            else:
                print("❌ Some tests failed")
                return False

        except subprocess.TimeoutExpired:
            print("❌ Tests timed out")
            return False
        except FileNotFoundError:
            print("⚠️  Pytest not found, skipping tests")
            return True
        except Exception as e:
            print(f"❌ Test error: {e}")
            return False

    def build_package(self) -> bool:
        """Build the package"""
        print("📦 Building package...")

        try:
            # Build source distribution and wheel
            result = subprocess.run(
                [sys.executable, "-m", "build"],
                cwd=self.root_dir,
                timeout=300
            )

            if result.returncode == 0:
                print("✅ Package built successfully")

                # List built packages
                if self.dist_dir.exists():
                    packages = list(self.dist_dir.glob("*"))
                    print(f"   Built packages ({len(packages)}):")
                    for package in packages:
                        size = package.stat().st_size / 1024 / 1024  # MB
                        print(f"      {package.name} ({size:.1f}MB)")

                return True
            else:
                print("❌ Package build failed")
                return False

        except subprocess.TimeoutExpired:
            print("❌ Package build timed out")
            return False
        except FileNotFoundError:
            print("❌ Build tool not found. Install with: pip install build")
            return False
        except Exception as e:
            print(f"❌ Build error: {e}")
            return False

    def test_installation(self) -> bool:
        """Test package installation in clean environment"""
        print("🔧 Testing package installation...")

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Find the wheel file
                wheel_files = list(self.dist_dir.glob("*.whl"))
                if not wheel_files:
                    print("❌ No wheel file found")
                    return False

                wheel_file = wheel_files[0]
                print(f"   Testing wheel: {wheel_file.name}")

                # Create virtual environment
                venv_dir = Path(temp_dir) / "test_venv"
                subprocess.run([
                    sys.executable, "-m", "venv", str(venv_dir)
                ], check=True, timeout=60)

                # Determine pip path
                if os.name == 'nt':  # Windows
                    pip_path = venv_dir / "Scripts" / "pip"
                    python_path = venv_dir / "Scripts" / "python"
                else:  # Unix
                    pip_path = venv_dir / "bin" / "pip"
                    python_path = venv_dir / "bin" / "python"

                # Install the package
                subprocess.run([
                    str(pip_path), "install", str(wheel_file)
                ], check=True, timeout=120)

                # Test import
                test_script = '''
import cohortrag_engine
print(f"CohortRAG Engine v{cohortrag_engine.__version__} installed successfully")

# Test high-level API
from cohortrag_engine import CohortRAGEngine
engine = CohortRAGEngine()
print("✅ High-level API working")

# Test CLI availability
import subprocess
result = subprocess.run(["cohortrag", "--help"], capture_output=True, text=True)
if result.returncode == 0:
    print("✅ CLI working")
else:
    print("⚠️  CLI may need additional setup")
'''

                result = subprocess.run([
                    str(python_path), "-c", test_script
                ], capture_output=True, text=True, timeout=60)

                if result.returncode == 0:
                    print("✅ Installation test passed")
                    print(f"   Output: {result.stdout.strip()}")
                    return True
                else:
                    print("❌ Installation test failed")
                    print(f"   Error: {result.stderr}")
                    return False

            except subprocess.CalledProcessError as e:
                print(f"❌ Installation test failed: {e}")
                return False
            except subprocess.TimeoutExpired:
                print("❌ Installation test timed out")
                return False
            except Exception as e:
                print(f"❌ Installation test error: {e}")
                return False

    def validate_package_metadata(self) -> bool:
        """Validate package metadata"""
        print("📋 Validating package metadata...")

        try:
            # Check pyproject.toml
            pyproject_file = self.root_dir / "pyproject.toml"
            if not pyproject_file.exists():
                print("❌ pyproject.toml not found")
                return False

            # Check __init__.py
            init_file = self.root_dir / "cohortrag_engine" / "__init__.py"
            if not init_file.exists():
                print("❌ __init__.py not found")
                return False

            # Read version from __init__.py
            with open(init_file, 'r') as f:
                init_content = f.read()
                if '__version__' not in init_content:
                    print("❌ __version__ not found in __init__.py")
                    return False

            print("✅ Package metadata validation passed")
            return True

        except Exception as e:
            print(f"❌ Metadata validation error: {e}")
            return False

    def check_security(self) -> bool:
        """Basic security checks"""
        print("🔒 Running security checks...")

        try:
            # Check for hardcoded secrets
            secret_patterns = [
                "password",
                "secret",
                "key",
                "token",
                "api_key"
            ]

            issues_found = []

            for py_file in self.root_dir.rglob("*.py"):
                if "venv" in str(py_file) or "__pycache__" in str(py_file):
                    continue

                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()

                        for pattern in secret_patterns:
                            if f"{pattern} = " in content and "getenv" not in content:
                                # Check if it's not in getenv or config
                                lines = content.split('\n')
                                for i, line in enumerate(lines):
                                    if f"{pattern} = " in line and "getenv" not in line and "config" not in line:
                                        if not line.strip().startswith('#'):  # Not a comment
                                            issues_found.append(f"{py_file}:{i+1} - Potential hardcoded {pattern}")

                except UnicodeDecodeError:
                    continue

            if issues_found:
                print("⚠️  Potential security issues found:")
                for issue in issues_found[:5]:  # Show first 5
                    print(f"   {issue}")
                if len(issues_found) > 5:
                    print(f"   ... and {len(issues_found) - 5} more")
                print("   Please verify these are not actual secrets")
            else:
                print("✅ No obvious security issues found")

            return len(issues_found) == 0

        except Exception as e:
            print(f"❌ Security check error: {e}")
            return False

    def generate_build_report(self, results: Dict[str, bool]) -> None:
        """Generate build report"""
        print("\n📊 Build Report")
        print("=" * 50)

        total_checks = len(results)
        passed_checks = sum(results.values())

        print(f"Overall Status: {'✅ PASSED' if passed_checks == total_checks else '❌ FAILED'}")
        print(f"Checks Passed: {passed_checks}/{total_checks}")
        print()

        for check, status in results.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {check}")

        if passed_checks == total_checks:
            print(f"\n🎉 Package is ready for deployment!")
            print(f"📦 Distribution files: {self.dist_dir}")
            print(f"🚀 Ready to upload to PyPI")
        else:
            print(f"\n⚠️  Please fix issues before deployment")

    def run_full_pipeline(self) -> bool:
        """Run the complete build and test pipeline"""
        print("🚀 CohortRAG Engine - Build and Test Pipeline")
        print("=" * 60)

        # Define all checks
        checks = [
            ("Clean Build Artifacts", self.clean_build_artifacts),
            ("Code Quality Checks", self.run_code_quality_checks),
            ("Test Suite", self.run_tests),
            ("Package Metadata", self.validate_package_metadata),
            ("Security Checks", self.check_security),
            ("Build Package", self.build_package),
            ("Installation Test", self.test_installation),
        ]

        results = {}

        # Run all checks
        for name, check_func in checks:
            print(f"\n{'='*20} {name} {'='*20}")
            try:
                results[name] = check_func()
            except Exception as e:
                print(f"❌ {name} failed with error: {e}")
                results[name] = False

        # Generate report
        self.generate_build_report(results)

        # Return overall success
        return all(results.values())


def main():
    """Main function"""
    builder = PackageBuilder()
    success = builder.run_full_pipeline()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()