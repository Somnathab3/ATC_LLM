#!/usr/bin/env python3
"""Analyze test files to identify duplications and structure."""

import os
import ast
from collections import defaultdict

def analyze_tests():
    """Analyze all test files."""
    test_files = []
    for root, dirs, files in os.walk('tests'):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                test_files.append(os.path.join(root, file))

    test_summary = {}
    for file_path in test_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            classes = []
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    functions.append(node.name)
            
            test_summary[file_path] = {
                'classes': classes,
                'functions': functions,
                'line_count': len(content.splitlines())
            }
        except Exception as e:
            test_summary[file_path] = {'error': str(e)}

    # Print summary
    print("TEST FILE ANALYSIS")
    print("=" * 50)
    
    for file_path, info in sorted(test_summary.items()):
        print(f"\n{file_path}:")
        if 'error' in info:
            print(f"  Error: {info['error']}")
        else:
            print(f"  Lines: {info['line_count']}")
            if info['classes']:
                print(f"  Classes: {', '.join(info['classes'])}")
            if info['functions']:
                print(f"  Functions: {', '.join(info['functions'])}")

    # Analyze duplications
    print("\n\nDUPLICATION ANALYSIS")
    print("=" * 50)
    
    # Group by similar function names
    function_groups = defaultdict(list)
    for file_path, info in test_summary.items():
        if 'functions' in info:
            for func in info['functions']:
                function_groups[func].append(file_path)
    
    duplicates = {func: files for func, files in function_groups.items() if len(files) > 1}
    
    if duplicates:
        print("\nDuplicate function names across files:")
        for func, files in duplicates.items():
            print(f"  {func}: {', '.join(files)}")
    
    # Group by module being tested
    module_groups = defaultdict(list)
    for file_path in test_summary.keys():
        if 'error' not in test_summary[file_path]:
            base_name = os.path.basename(file_path)
            if base_name.startswith('test_'):
                module_name = base_name[5:-3]  # Remove 'test_' and '.py'
                module_groups[module_name].append(file_path)
    
    print("\nFiles testing similar modules:")
    for module, files in module_groups.items():
        if len(files) > 1:
            print(f"  {module}: {', '.join(files)}")

if __name__ == "__main__":
    analyze_tests()
