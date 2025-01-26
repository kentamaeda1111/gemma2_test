import sys
import pkg_resources

def check_python_version():
    current_version = sys.version_info
    min_version = (3, 10)
    
    if current_version >= min_version:
        print(f"✓ Python version {sys.version} meets minimum requirement of 3.10")
    else:
        print(f"✗ Python version {sys.version} does not meet minimum requirement of 3.10")

def check_required_packages():
    with open('requirements.txt') as f:
        requirements = pkg_resources.parse_requirements(f)
        for req in requirements:
            try:
                pkg_resources.require(str(req))
                print(f"✓ {req} is installed correctly")
            except pkg_resources.DistributionNotFound:
                print(f"✗ {req} is not installed")
            except pkg_resources.VersionConflict:
                print(f"✗ {req} version conflict")

def main():
    print("Checking Basic Requirements...")
    print("\nPython Version Check:")
    check_python_version()
    
    print("\nRequired Packages Check:")
    check_required_packages()

if __name__ == "__main__":
    main() 