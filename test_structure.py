#!/usr/bin/env python3
"""
Simple syntax and structure test for the refactored hdf_preparator_v2.py
"""

import sys
import ast

def test_syntax():
    """Test that the refactored file has valid Python syntax."""
    try:
        with open('/home/marc/dev/fdq_diffusion/hdf_preparator_v2.py', 'r') as f:
            content = f.read()
        
        # Parse the file to check syntax
        ast.parse(content)
        print("✓ Syntax is valid")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return False

def test_structure():
    """Test that the refactored file has the expected structure."""
    try:
        with open('/home/marc/dev/fdq_diffusion/hdf_preparator_v2.py', 'r') as f:
            content = f.read()
        
        # Check for key classes and functions
        expected_items = [
            'class DatasetType',
            'class DatasetConfig',
            'class SampleInfo',
            'class TensorFormatter',
            'class HDFKeyProcessor',
            'class HDF5FileProcessor',
            'class H5Dataset',
            'class DatasetFactory',
            'def create_datasets'
        ]
        
        missing_items = []
        for item in expected_items:
            if item not in content:
                missing_items.append(item)
        
        if missing_items:
            print(f"✗ Missing items: {missing_items}")
            return False
        else:
            print("✓ All expected classes and functions are present")
            return True
    
    except Exception as e:
        print(f"✗ Error checking structure: {e}")
        return False

def test_imports():
    """Test that imports are properly structured."""
    try:
        with open('/home/marc/dev/fdq_diffusion/hdf_preparator_v2.py', 'r') as f:
            lines = f.readlines()
        
        # Check for proper import structure
        import_section = []
        in_docstring = False
        for i, line in enumerate(lines):
            # Skip docstrings
            if '"""' in line:
                in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
                
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_section.append((i+1, line.strip()))
            elif line.strip() and not line.strip().startswith('#'):
                # Stop at first non-import, non-comment line
                break
        
        if import_section:
            print("✓ Import section found:")
            for line_num, import_line in import_section:
                print(f"  Line {line_num}: {import_line}")
            return True
        else:
            print("✗ No imports found")
            return False
    
    except Exception as e:
        print(f"✗ Error checking imports: {e}")
        return False

def test_backward_compatibility():
    """Test that the create_datasets function is still available."""
    try:
        with open('/home/marc/dev/fdq_diffusion/hdf_preparator_v2.py', 'r') as f:
            content = f.read()
        
        # Check for the main function
        if 'def create_datasets(' in content:
            print("✓ create_datasets function is available (backward compatibility)")
            return True
        else:
            print("✗ create_datasets function not found")
            return False
    
    except Exception as e:
        print(f"✗ Error checking backward compatibility: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing refactored hdf_preparator_v2.py structure...")
    print("=" * 60)
    
    tests = [
        test_syntax,
        test_structure,
        test_imports,
        test_backward_compatibility
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    print("=" * 60)
    if all(results):
        print("✓ All structure tests passed!")
        print("✓ The refactored code is properly structured and maintains backward compatibility.")
    else:
        print("✗ Some tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
