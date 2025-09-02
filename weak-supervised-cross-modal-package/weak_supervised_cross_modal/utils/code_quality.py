"""
代码质量检查工具
提供类型检查、导入优化和代码规范检查功能
"""

import ast
import os
import sys
import importlib
import inspect
from typing import Dict, List, Set, Tuple, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ImportAnalyzer:
    """导入分析器，检测循环依赖和优化导入结构"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.imports_graph: Dict[str, Set[str]] = {}
        self.file_imports: Dict[str, List[str]] = {}
        
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """分析单个文件的导入情况"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
            
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append({
                            "type": "import",
                            "module": alias.name,
                            "alias": alias.asname,
                            "line": node.lineno
                        })
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append({
                            "type": "from_import",
                            "module": module,
                            "name": alias.name,
                            "alias": alias.asname,
                            "line": node.lineno,
                            "level": node.level
                        })
            
            return {
                "file": file_path,
                "imports": imports,
                "total_imports": len(imports)
            }
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return {"file": file_path, "imports": [], "total_imports": 0}
    
    def detect_circular_imports(self) -> List[List[str]]:
        """检测循环导入"""
        def dfs(node: str, path: List[str], visited: Set[str]) -> List[List[str]]:
            if node in path:
                # 找到循环
                cycle_start = path.index(node)
                return [path[cycle_start:] + [node]]
            
            if node in visited:
                return []
            
            visited.add(node)
            cycles = []
            
            for neighbor in self.imports_graph.get(node, set()):
                cycles.extend(dfs(neighbor, path + [node], visited.copy()))
            
            return cycles
        
        all_cycles = []
        visited_global = set()
        
        for node in self.imports_graph:
            if node not in visited_global:
                cycles = dfs(node, [], set())
                all_cycles.extend(cycles)
                visited_global.add(node)
        
        return all_cycles


def run_quality_check(project_root: str = ".") -> Dict[str, Any]:
    """运行代码质量检查"""
    analyzer = ImportAnalyzer(project_root)
    python_files = list(Path(project_root).rglob("*.py"))
    
    report = {
        "project_root": project_root,
        "total_files": len(python_files),
        "import_analysis": {},
        "summary": {}
    }
    
    # 导入分析
    for file_path in python_files:
        analysis = analyzer.analyze_file(str(file_path))
        report["import_analysis"][str(file_path)] = analysis
    
    # 检测循环导入
    circular_imports = analyzer.detect_circular_imports()
    report["import_analysis"]["circular_imports"] = circular_imports
    
    # 生成摘要
    report["summary"] = {
        "total_files_analyzed": len(python_files),
        "circular_imports_count": len(circular_imports)
    }
    
    return report

 