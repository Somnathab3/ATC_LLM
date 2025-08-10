"""
Repository cleanup utility for ATC LLM project.

This module provides functionality to clean up old files, obsolete notebooks,
demos, and stale output directories while preserving essential project files.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Set, Dict, Any
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RepositoryCleanup:
    """Repository cleanup utility with dry-run and selective cleaning."""
    
    def __init__(self, project_root: Path):
        """
        Initialize cleanup utility.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.cleanup_stats = {
            'files_removed': 0,
            'dirs_removed': 0,
            'bytes_freed': 0,
            'items_preserved': 0
        }
        
        # Files and directories to preserve (whitelist)
        self.preserve_patterns = {
            # Essential project files
            'README.md', 'pyproject.toml', 'requirements.txt', 'pytest.ini',
            '.gitignore', '.github', 'LICENSE',
            
            # Source code
            'src/', 'tests/', 'cli.py',
            
            # Documentation that's referenced
            'DEPENDENCY_MATRIX.md', 'FUNCTION_DEPENDENCY_TREE.md', 
            'IMPLEMENTATION_SUMMARY.md', 'WORKFLOW_RESULTS_SUMMARY.md',
            
            # Important scripts
            'atc-llm.sh', 'atc-llm.bat',
            
            # Current sample data
            'sample_data/',
            
            # Version control
            '.git/'
        }
        
        # Patterns for files/directories to remove
        self.remove_patterns = {
            # Old demo files
            'demo_*.py', 'test_*.py', 'debug_*.py', 'validate_*.py',
            'enhanced_*.py', 'workflow_*.py', 'create_*.py', 'fix_*.py',
            
            # Old notebooks and demos
            '*.ipynb', 'notebooks/', 'demos/',
            
            # Stale outputs
            'Output/', 'outputs/', 'results/', 'reports/',
            
            # Backup and temporary files
            'backup/', '*.bak', '*.tmp', '*~', '*.swp',
            
            # Old data exports
            'scat_export_test/', 'test_scat_data_demo/',
            
            # Compressed archives
            '*.zip', '*.tar.gz', '*.tar.bz2',
            
            # Python cache
            '__pycache__/', '*.pyc', '*.pyo', '.pytest_cache/',
            
            # IDE files
            '.vscode/', '.idea/', '*.code-workspace',
            
            # OS files
            '.DS_Store', 'Thumbs.db',
            
            # Log files
            '*.log', 'logs/'
        }
        
        # Files referenced by CLI and function tree (preserve these)
        self.referenced_files = self._get_referenced_files()
    
    def _get_referenced_files(self) -> Set[str]:
        """Get files referenced by CLI and function dependency tree."""
        referenced = set()
        
        # Add files from CLI imports and usage
        cli_referenced = [
            'src/cdr/pipeline.py',
            'src/cdr/scat_adapter.py', 
            'src/cdr/bluesky_io.py',
            'src/cdr/llm_client.py',
            'src/cdr/metrics.py',
            'src/cdr/reporting.py',
            'src/cdr/nav_utils.py',
            'src/cdr/schemas.py',
            'src/cdr/detect.py',
            'src/cdr/resolve.py',
            'src/cdr/geodesy.py',
            'src/cdr/enhanced_cpa.py',
            'src/cdr/dual_verification.py',
            'src/cdr/memory.py',
            'src/cdr/intruders_openap.py',
            'src/cdr/visualization.py',
            'src/atc_llm_cli.py',
            'test_run_e2e.py'
        ]
        
        for file_path in cli_referenced:
            referenced.add(file_path)
        
        return referenced
    
    def scan_for_cleanup(self) -> Dict[str, List[Path]]:
        """Scan project for files/directories that can be cleaned up."""
        cleanup_candidates = {
            'old_demos': [],
            'stale_outputs': [],
            'backup_files': [],
            'temp_files': [],
            'cache_files': [],
            'archive_files': []
        }
        
        def should_preserve(path: Path) -> bool:
            """Check if a path should be preserved."""
            path_str = str(path.relative_to(self.project_root))
            
            # Check if it's in preserve patterns
            for pattern in self.preserve_patterns:
                if pattern.endswith('/'):
                    if path_str.startswith(pattern):
                        return True
                else:
                    if path_str == pattern or path.name == pattern:
                        return True
            
            # Check if it's referenced by CLI/function tree
            if path_str in self.referenced_files:
                return True
            
            return False
        
        def categorize_file(path: Path) -> str:
            """Categorize file for cleanup."""
            name = path.name.lower()
            path_str = str(path.relative_to(self.project_root)).lower()
            
            if any(pattern in name for pattern in ['demo_', 'test_', 'debug_', 'validate_', 'create_']):
                return 'old_demos'
            elif any(pattern in path_str for pattern in ['output', 'result', 'report']):
                return 'stale_outputs'
            elif any(pattern in name for pattern in ['.bak', '.tmp', '~', '.swp']):
                return 'temp_files'
            elif any(pattern in name for pattern in ['.zip', '.tar.gz', '.tar.bz2']):
                return 'archive_files'
            elif any(pattern in path_str for pattern in ['__pycache__', '.pytest_cache']):
                return 'cache_files'
            elif 'backup' in path_str:
                return 'backup_files'
            else:
                return 'temp_files'
        
        # Walk through project directory
        for root, dirs, files in os.walk(self.project_root):
            root_path = Path(root)
            
            # Skip .git directory entirely
            if '.git' in str(root_path):
                continue
            
            # Check directories
            for dir_name in dirs.copy():
                dir_path = root_path / dir_name
                if not should_preserve(dir_path):
                    category = categorize_file(dir_path)
                    cleanup_candidates[category].append(dir_path)
                    dirs.remove(dir_name)  # Don't recurse into this directory
            
            # Check files
            for file_name in files:
                file_path = root_path / file_name
                if not should_preserve(file_path):
                    category = categorize_file(file_path)
                    cleanup_candidates[category].append(file_path)
        
        return cleanup_candidates
    
    def dry_run(self) -> Dict[str, Any]:
        """Perform dry run to show what would be cleaned up."""
        logger.info("Performing dry run cleanup scan...")
        
        candidates = self.scan_for_cleanup()
        
        total_files = sum(len(files) for files in candidates.values())
        total_size = 0
        
        summary = {
            'total_items': total_files,
            'categories': {},
            'total_size_mb': 0,
            'sample_files': {}
        }
        
        for category, paths in candidates.items():
            if not paths:
                continue
            
            category_size = 0
            sample_files = []
            
            for path in paths[:10]:  # Sample first 10 files
                try:
                    if path.is_file():
                        size = path.stat().st_size
                        category_size += size
                        sample_files.append(f"{path.name} ({size/1024:.1f} KB)")
                    elif path.is_dir():
                        # Estimate directory size
                        dir_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                        category_size += dir_size
                        sample_files.append(f"{path.name}/ ({dir_size/1024:.1f} KB)")
                except (OSError, PermissionError):
                    sample_files.append(f"{path.name} (size unknown)")
            
            summary['categories'][category] = {
                'count': len(paths),
                'size_mb': category_size / (1024 * 1024),
                'sample_files': sample_files
            }
            total_size += category_size
        
        summary['total_size_mb'] = total_size / (1024 * 1024)
        
        # Log summary
        logger.info(f"Dry run complete: {total_files} items, {summary['total_size_mb']:.1f} MB")
        for category, info in summary['categories'].items():
            logger.info(f"  {category}: {info['count']} items, {info['size_mb']:.1f} MB")
        
        return summary
    
    def cleanup(self, categories: List[str] = None, confirm: bool = False) -> Dict[str, Any]:
        """Perform actual cleanup."""
        if not confirm:
            logger.error("Cleanup requires explicit confirmation with confirm=True")
            return {'error': 'Confirmation required'}
        
        logger.info("Starting repository cleanup...")
        
        candidates = self.scan_for_cleanup()
        
        if categories:
            # Filter to only specified categories
            filtered_candidates = {cat: paths for cat, paths in candidates.items() if cat in categories}
            candidates = filtered_candidates
        
        removed_items = []
        errors = []
        
        for category, paths in candidates.items():
            for path in paths:
                try:
                    if path.is_file():
                        size = path.stat().st_size
                        path.unlink()
                        self.cleanup_stats['files_removed'] += 1
                        self.cleanup_stats['bytes_freed'] += size
                        removed_items.append(str(path))
                        logger.debug(f"Removed file: {path}")
                    
                    elif path.is_dir():
                        # Calculate directory size before removal
                        dir_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                        shutil.rmtree(path)
                        self.cleanup_stats['dirs_removed'] += 1
                        self.cleanup_stats['bytes_freed'] += dir_size
                        removed_items.append(str(path) + '/')
                        logger.debug(f"Removed directory: {path}")
                
                except (OSError, PermissionError) as e:
                    error_msg = f"Failed to remove {path}: {e}"
                    errors.append(error_msg)
                    logger.warning(error_msg)
        
        # Summary
        summary = {
            'files_removed': self.cleanup_stats['files_removed'],
            'dirs_removed': self.cleanup_stats['dirs_removed'],
            'total_items_removed': len(removed_items),
            'mb_freed': self.cleanup_stats['bytes_freed'] / (1024 * 1024),
            'errors': errors,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Cleanup complete: {summary['total_items_removed']} items removed, "
                   f"{summary['mb_freed']:.1f} MB freed")
        
        if errors:
            logger.warning(f"{len(errors)} errors occurred during cleanup")
        
        return summary
    
    def clean_specific_pattern(self, pattern: str, confirm: bool = False) -> Dict[str, Any]:
        """Clean files matching a specific pattern."""
        if not confirm:
            logger.error("Cleanup requires explicit confirmation with confirm=True")
            return {'error': 'Confirmation required'}
        
        matching_files = list(self.project_root.rglob(pattern))
        removed = []
        errors = []
        
        for file_path in matching_files:
            # Don't remove if it's in preserve list
            if any(preserve in str(file_path) for preserve in self.preserve_patterns):
                continue
            
            try:
                if file_path.is_file():
                    file_path.unlink()
                    removed.append(str(file_path))
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
                    removed.append(str(file_path) + '/')
            except (OSError, PermissionError) as e:
                errors.append(f"Failed to remove {file_path}: {e}")
        
        return {
            'pattern': pattern,
            'removed': removed,
            'errors': errors,
            'count': len(removed)
        }
    
    def export_cleanup_report(self, output_file: Path) -> None:
        """Export cleanup report to JSON."""
        report = {
            'project_root': str(self.project_root),
            'scan_timestamp': datetime.now().isoformat(),
            'preserve_patterns': list(self.preserve_patterns),
            'remove_patterns': list(self.remove_patterns),
            'referenced_files': list(self.referenced_files),
            'cleanup_stats': self.cleanup_stats
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Cleanup report exported to {output_file}")


def main():
    """Command-line interface for repository cleanup."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ATC LLM Repository Cleanup Utility')
    parser.add_argument('--project-root', type=Path, default=Path.cwd(),
                       help='Project root directory')
    parser.add_argument('--dry-run', action='store_true',
                       help='Perform dry run without actual cleanup')
    parser.add_argument('--confirm', action='store_true',
                       help='Confirm actual cleanup')
    parser.add_argument('--categories', nargs='+',
                       choices=['old_demos', 'stale_outputs', 'backup_files', 
                               'temp_files', 'cache_files', 'archive_files'],
                       help='Specific categories to clean')
    parser.add_argument('--pattern', type=str,
                       help='Clean files matching specific pattern')
    parser.add_argument('--report', type=Path,
                       help='Export cleanup report to file')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    cleanup = RepositoryCleanup(args.project_root)
    
    if args.dry_run:
        summary = cleanup.dry_run()
        print(f"\nDry run results:")
        print(f"Total items to clean: {summary['total_items']}")
        print(f"Total size: {summary['total_size_mb']:.1f} MB")
        for category, info in summary['categories'].items():
            print(f"  {category}: {info['count']} items ({info['size_mb']:.1f} MB)")
            if info['sample_files']:
                print(f"    Sample files: {', '.join(info['sample_files'][:3])}")
    
    elif args.pattern:
        result = cleanup.clean_specific_pattern(args.pattern, args.confirm)
        print(f"Pattern '{args.pattern}': {result['count']} items removed")
        if result['errors']:
            print(f"Errors: {len(result['errors'])}")
    
    elif args.confirm:
        result = cleanup.cleanup(args.categories, confirm=True)
        print(f"Cleanup complete: {result['total_items_removed']} items removed")
        print(f"Space freed: {result['mb_freed']:.1f} MB")
        if result['errors']:
            print(f"Errors: {len(result['errors'])}")
    
    else:
        print("Use --dry-run to see what would be cleaned, or --confirm to perform cleanup")
    
    if args.report:
        cleanup.export_cleanup_report(args.report)


if __name__ == '__main__':
    main()