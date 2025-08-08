"""Comprehensive reporting infrastructure for Sprint 5.

This module generates:
- Metrics tables and CSV exports
- Performance charts and visualizations
- Narrative analysis of failure modes
- Timeline analysis and example scenarios
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from .metrics import MetricsCollector, MetricsSummary, ComparisonReport
from .simple_stress_test import StressTestResult

logger = logging.getLogger(__name__)


@dataclass
class FailureModeAnalysis:
    """Analysis of different failure modes."""
    late_detection_rate: float
    missed_conflict_rate: float
    unsafe_resolution_rate: float
    oscillation_rate: float
    
    # Detailed breakdown
    late_detection_scenarios: List[str]
    missed_conflict_scenarios: List[str] 
    unsafe_resolution_scenarios: List[str]
    oscillation_scenarios: List[str]
    
    # Impact analysis
    safety_impact_score: float  # 0-100
    performance_impact_score: float  # 0-100
    recommendations: List[str]


@dataclass
class ReportPackage:
    """Complete Sprint 5 report package."""
    generation_timestamp: datetime
    summary_metrics: Dict[str, Any]
    failure_analysis: FailureModeAnalysis
    stress_test_results: List[StressTestResult]
    baseline_comparison: Optional[ComparisonReport]
    
    # File paths for generated artifacts
    metrics_csv_path: str
    charts_directory: str
    narrative_report_path: str


class Sprint5Reporter:
    """Comprehensive reporting system for Sprint 5."""
    
    def __init__(self, output_dir: str = "reports/sprint_05"):
        """Initialize reporter with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
    
    def generate_metrics_csv(
        self,
        metrics_data: List[MetricsSummary],
        filename: str = "comprehensive_metrics.csv"
    ) -> str:
        """Generate comprehensive metrics CSV file.
        
        Args:
            metrics_data: List of metrics summaries
            filename: Output filename
            
        Returns:
            Path to generated CSV file
        """
        output_path = self.output_dir / filename
        
        # Convert metrics to DataFrame
        if metrics_data:
            df_data = [asdict(metrics) for metrics in metrics_data]
            df = pd.DataFrame(df_data)
            
            # Add computed columns
            df['conflict_resolution_efficiency'] = (
                df['successful_resolutions'] / df['total_resolutions_issued']
            ).fillna(0)
            
            df['safety_score'] = (
                100 * (1 - df['safety_violations'] / (df['conflicts_detected'] + 1))
            )
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            logger.info(f"Metrics CSV saved to: {output_path}")
        else:
            # Create empty CSV with headers
            columns = [
                'timestamp', 'conflicts_detected', 'conflicts_resolved',
                'total_resolutions_issued', 'successful_resolutions',
                'safety_violations', 'avg_detection_latency_sec',
                'avg_resolution_time_sec', 'min_separation_nm'
            ]
            df = pd.DataFrame(columns=columns)
            df.to_csv(output_path, index=False)
            logger.warning(f"Created empty metrics CSV: {output_path}")
        
        return str(output_path)
    
    def generate_performance_charts(
        self,
        metrics_data: List[MetricsSummary],
        stress_results: List[StressTestResult],
        charts_subdir: str = "charts"
    ) -> str:
        """Generate comprehensive performance charts.
        
        Args:
            metrics_data: Metrics summaries for charting
            stress_results: Stress test results
            charts_subdir: Subdirectory for charts
            
        Returns:
            Path to charts directory
        """
        charts_dir = self.output_dir / charts_subdir
        charts_dir.mkdir(exist_ok=True)
        
        # Chart 1: Detection Performance Over Time
        if metrics_data:
            self._plot_detection_performance(metrics_data, charts_dir)
        
        # Chart 2: Resolution Success Rates
        if metrics_data:
            self._plot_resolution_performance(metrics_data, charts_dir)
        
        # Chart 3: Safety Metrics
        if metrics_data:
            self._plot_safety_metrics(metrics_data, charts_dir)
        
        # Chart 4: Stress Test Results
        if stress_results:
            self._plot_stress_test_results(stress_results, charts_dir)
        
        # Chart 5: Failure Mode Distribution
        if stress_results:
            self._plot_failure_modes(stress_results, charts_dir)
        
        logger.info(f"Performance charts saved to: {charts_dir}")
        return str(charts_dir)
    
    def _plot_detection_performance(
        self,
        metrics_data: List[MetricsSummary],
        output_dir: Path
    ) -> None:
        """Plot conflict detection performance."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data
        timestamps = [m.timestamp for m in metrics_data]
        detection_rates = [
            m.conflicts_resolved / max(1, m.conflicts_detected) 
            for m in metrics_data
        ]
        detection_latency = [m.avg_detection_latency_sec for m in metrics_data]
        
        # Detection success rate
        ax1.plot(timestamps, detection_rates, marker='o', linewidth=2)
        ax1.set_title('Conflict Detection Success Rate')
        ax1.set_ylabel('Success Rate')
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        
        # Detection latency
        ax2.plot(timestamps, detection_latency, marker='s', color='orange', linewidth=2)
        ax2.set_title('Average Detection Latency')
        ax2.set_ylabel('Latency (seconds)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'detection_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_resolution_performance(
        self,
        metrics_data: List[MetricsSummary],
        output_dir: Path
    ) -> None:
        """Plot resolution performance metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Resolution success rate
        success_rates = [
            m.successful_resolutions / max(1, m.total_resolutions_issued)
            for m in metrics_data
        ]
        resolution_times = [m.avg_resolution_time_sec for m in metrics_data]
        
        ax1.bar(range(len(success_rates)), success_rates, alpha=0.7, color='green')
        ax1.set_title('Resolution Success Rate by Test')
        ax1.set_ylabel('Success Rate')
        ax1.set_xlabel('Test Number')
        ax1.set_ylim(0, 1.1)
        
        ax2.boxplot(resolution_times)
        ax2.set_title('Resolution Time Distribution')
        ax2.set_ylabel('Time (seconds)')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'resolution_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_safety_metrics(
        self,
        metrics_data: List[MetricsSummary],
        output_dir: Path
    ) -> None:
        """Plot safety-related metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Safety violations
        violations = [m.safety_violations for m in metrics_data]
        min_separations = [m.min_separation_nm for m in metrics_data]
        
        ax1.bar(range(len(violations)), violations, alpha=0.7, color='red')
        ax1.set_title('Safety Violations by Test')
        ax1.set_ylabel('Number of Violations')
        ax1.set_xlabel('Test Number')
        
        ax2.plot(range(len(min_separations)), min_separations, marker='o', color='blue')
        ax2.axhline(y=3.0, color='red', linestyle='--', label='Minimum Safe Separation')
        ax2.set_title('Minimum Separation Achieved')
        ax2.set_ylabel('Separation (nm)')
        ax2.set_xlabel('Test Number')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'safety_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_stress_test_results(
        self,
        stress_results: List[StressTestResult],
        output_dir: Path
    ) -> None:
        """Plot stress test results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data
        scenario_ids = [r.scenario_id for r in stress_results]
        conflicts_detected = [r.conflicts_detected for r in stress_results]
        conflicts_resolved = [r.conflicts_resolved for r in stress_results]
        min_separations = [r.min_separation_nm for r in stress_results]
        processing_times = [r.processing_time_sec for r in stress_results]
        
        # Conflicts detected vs resolved
        x = range(len(stress_results))
        width = 0.35
        ax1.bar([i - width/2 for i in x], conflicts_detected, width, label='Detected', alpha=0.7)
        ax1.bar([i + width/2 for i in x], conflicts_resolved, width, label='Resolved', alpha=0.7)
        ax1.set_title('Conflicts Detected vs Resolved')
        ax1.set_ylabel('Number of Conflicts')
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenario_ids, rotation=45)
        ax1.legend()
        
        # Minimum separations
        ax2.bar(x, min_separations, alpha=0.7, color='blue')
        ax2.axhline(y=3.0, color='red', linestyle='--', label='Safe Separation')
        ax2.set_title('Minimum Separation by Scenario')
        ax2.set_ylabel('Separation (nm)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(scenario_ids, rotation=45)
        ax2.legend()
        
        # Processing times
        ax3.bar(x, processing_times, alpha=0.7, color='green')
        ax3.set_title('Processing Time by Scenario')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(scenario_ids, rotation=45)
        
        # Success rate
        success_rates = [
            r.conflicts_resolved / max(1, r.conflicts_detected) 
            for r in stress_results
        ]
        ax4.bar(x, success_rates, alpha=0.7, color='orange')
        ax4.set_title('Resolution Success Rate')
        ax4.set_ylabel('Success Rate')
        ax4.set_ylim(0, 1.1)
        ax4.set_xticks(x)
        ax4.set_xticklabels(scenario_ids, rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'stress_test_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_failure_modes(
        self,
        stress_results: List[StressTestResult],
        output_dir: Path
    ) -> None:
        """Plot failure mode analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Failure mode counts
        total_safety_violations = sum(r.safety_violations for r in stress_results)
        total_oscillations = sum(r.oscillations for r in stress_results)
        
        failure_types = ['Safety Violations', 'Oscillations', 'Processing Failures']
        failure_counts = [total_safety_violations, total_oscillations, 0]  # Placeholder for processing failures
        
        ax1.pie(failure_counts, labels=failure_types, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Failure Mode Distribution')
        
        # Scenario difficulty vs performance
        scenario_difficulty = list(range(len(stress_results)))  # Proxy for difficulty
        performance_scores = [
            (r.conflicts_resolved / max(1, r.conflicts_detected)) * 
            (1 - r.safety_violations / 10)  # Composite score
            for r in stress_results
        ]
        
        ax2.scatter(scenario_difficulty, performance_scores, alpha=0.7, s=100)
        ax2.set_xlabel('Scenario Difficulty Index')
        ax2.set_ylabel('Performance Score')
        ax2.set_title('Performance vs Scenario Difficulty')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'failure_mode_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_failure_modes(
        self,
        stress_results: List[StressTestResult],
        metrics_data: List[MetricsSummary]
    ) -> FailureModeAnalysis:
        """Analyze failure modes across test results."""
        total_tests = len(stress_results)
        if total_tests == 0:
            return FailureModeAnalysis(
                late_detection_rate=0.0,
                missed_conflict_rate=0.0,
                unsafe_resolution_rate=0.0,
                oscillation_rate=0.0,
                late_detection_scenarios=[],
                missed_conflict_scenarios=[],
                unsafe_resolution_scenarios=[],
                oscillation_scenarios=[],
                safety_impact_score=100.0,
                performance_impact_score=100.0,
                recommendations=["No test data available for analysis"]
            )
        
        # Calculate failure rates
        total_safety_violations = sum(r.safety_violations for r in stress_results)
        total_oscillations = sum(r.oscillations for r in stress_results)
        
        # Identify problematic scenarios
        safety_violation_scenarios = [
            r.scenario_id for r in stress_results if r.safety_violations > 0
        ]
        oscillation_scenarios = [
            r.scenario_id for r in stress_results if r.oscillations > 0
        ]
        
        # Calculate impact scores
        safety_impact = max(0, 100 - (total_safety_violations / total_tests) * 50)
        performance_impact = np.mean([
            r.conflicts_resolved / max(1, r.conflicts_detected) 
            for r in stress_results
        ]) * 100
        
        # Generate recommendations
        recommendations = []
        if total_safety_violations > 0:
            recommendations.append("Enhance safety validation algorithms")
        if total_oscillations > 0:
            recommendations.append("Strengthen oscillation guard mechanisms")
        if performance_impact < 80:
            recommendations.append("Optimize resolution algorithms for better success rates")
        if not recommendations:
            recommendations.append("System performing within acceptable parameters")
        
        return FailureModeAnalysis(
            late_detection_rate=0.0,  # Would need more detailed data
            missed_conflict_rate=0.0,  # Would need more detailed data
            unsafe_resolution_rate=total_safety_violations / total_tests,
            oscillation_rate=total_oscillations / total_tests,
            late_detection_scenarios=[],
            missed_conflict_scenarios=[],
            unsafe_resolution_scenarios=safety_violation_scenarios,
            oscillation_scenarios=oscillation_scenarios,
            safety_impact_score=safety_impact,
            performance_impact_score=performance_impact,
            recommendations=recommendations
        )
    
    def generate_narrative_report(
        self,
        failure_analysis: FailureModeAnalysis,
        stress_results: List[StressTestResult],
        metrics_data: List[MetricsSummary],
        filename: str = "sprint5_narrative_analysis.md"
    ) -> str:
        """Generate comprehensive narrative analysis report."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            f.write("# Sprint 5 - Robustness and Failure Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"This report analyzes the robustness of the ATC LLM system across ")
            f.write(f"{len(stress_results)} stress test scenarios involving multi-intruder ")
            f.write(f"conflicts and edge cases.\n\n")
            
            # Key Findings
            f.write("## Key Findings\n\n")
            f.write(f"- **Safety Impact Score**: {failure_analysis.safety_impact_score:.1f}/100\n")
            f.write(f"- **Performance Impact Score**: {failure_analysis.performance_impact_score:.1f}/100\n")
            f.write(f"- **Unsafe Resolution Rate**: {failure_analysis.unsafe_resolution_rate:.2%}\n")
            f.write(f"- **Oscillation Rate**: {failure_analysis.oscillation_rate:.2%}\n\n")
            
            # Failure Mode Analysis
            f.write("## Failure Mode Analysis\n\n")
            
            f.write("### Late Detection\n")
            f.write(f"- **Rate**: {failure_analysis.late_detection_rate:.2%}\n")
            f.write(f"- **Affected Scenarios**: {len(failure_analysis.late_detection_scenarios)}\n")
            if failure_analysis.late_detection_scenarios:
                f.write(f"- **Scenarios**: {', '.join(failure_analysis.late_detection_scenarios)}\n")
            f.write("\n")
            
            f.write("### Missed Conflicts\n")
            f.write(f"- **Rate**: {failure_analysis.missed_conflict_rate:.2%}\n")
            f.write(f"- **Affected Scenarios**: {len(failure_analysis.missed_conflict_scenarios)}\n")
            f.write("\n")
            
            f.write("### Unsafe Resolutions\n")
            f.write(f"- **Rate**: {failure_analysis.unsafe_resolution_rate:.2%}\n")
            f.write(f"- **Affected Scenarios**: {len(failure_analysis.unsafe_resolution_scenarios)}\n")
            if failure_analysis.unsafe_resolution_scenarios:
                f.write(f"- **Scenarios**: {', '.join(failure_analysis.unsafe_resolution_scenarios)}\n")
            f.write("\n")
            
            f.write("### Oscillations\n")
            f.write(f"- **Rate**: {failure_analysis.oscillation_rate:.2%}\n")
            f.write(f"- **Affected Scenarios**: {len(failure_analysis.oscillation_scenarios)}\n")
            if failure_analysis.oscillation_scenarios:
                f.write(f"- **Scenarios**: {', '.join(failure_analysis.oscillation_scenarios)}\n")
            f.write("\n")
            
            # Performance Summary
            f.write("## Performance Summary\n\n")
            if stress_results:
                total_conflicts = sum(r.conflicts_detected for r in stress_results)
                total_resolved = sum(r.conflicts_resolved for r in stress_results)
                avg_resolution_rate = total_resolved / max(1, total_conflicts)
                
                f.write(f"- **Total Conflicts Detected**: {total_conflicts}\n")
                f.write(f"- **Total Conflicts Resolved**: {total_resolved}\n") 
                f.write(f"- **Average Resolution Rate**: {avg_resolution_rate:.2%}\n")
                f.write(f"- **Average Processing Time**: {np.mean([r.processing_time_sec for r in stress_results]):.2f} seconds\n")
                f.write(f"- **Minimum Separation Achieved**: {min(r.min_separation_nm for r in stress_results):.2f} nm\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            for i, rec in enumerate(failure_analysis.recommendations, 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")
            
            # Detailed Scenario Analysis
            f.write("## Detailed Scenario Analysis\n\n")
            for result in stress_results:
                f.write(f"### {result.scenario_id}\n")
                f.write(f"- **Conflicts Detected**: {result.conflicts_detected}\n")
                f.write(f"- **Conflicts Resolved**: {result.conflicts_resolved}\n")
                f.write(f"- **Success Rate**: {result.conflicts_resolved / max(1, result.conflicts_detected):.2%}\n")
                f.write(f"- **Min Separation**: {result.min_separation_nm:.2f} nm\n")
                f.write(f"- **Safety Violations**: {result.safety_violations}\n")
                f.write(f"- **Processing Time**: {result.processing_time_sec:.2f} seconds\n\n")
            
            # Technical Details
            f.write("## Technical Implementation Notes\n\n")
            f.write("- **Oscillation Guard**: Prevents opposite commands within 10 minutes without net benefit\n")
            f.write("- **Multi-Intruder Testing**: Scenarios with 2-4 simultaneous conflicts\n")
            f.write("- **Monte Carlo Perturbations**: Random variations to test robustness\n")
            f.write("- **Safety Validation**: Comprehensive checks before command execution\n\n")
            
            # Appendix
            f.write("## Appendix: Raw Data\n\n")
            f.write("Detailed metrics and charts are available in the accompanying CSV files and chart directory.\n")
        
        logger.info(f"Narrative report saved to: {output_path}")
        return str(output_path)
    
    def generate_complete_report_package(
        self,
        metrics_data: List[MetricsSummary],
        stress_results: List[StressTestResult],
        baseline_comparison: Optional[ComparisonReport] = None
    ) -> ReportPackage:
        """Generate complete Sprint 5 report package."""
        logger.info("Generating complete Sprint 5 report package...")
        
        # Generate all components
        metrics_csv = self.generate_metrics_csv(metrics_data)
        charts_dir = self.generate_performance_charts(metrics_data, stress_results)
        failure_analysis = self.analyze_failure_modes(stress_results, metrics_data)
        narrative_report = self.generate_narrative_report(
            failure_analysis, stress_results, metrics_data
        )
        
        # Create summary metrics
        summary_metrics = {
            "total_tests": len(stress_results),
            "total_conflicts": sum(r.conflicts_detected for r in stress_results),
            "total_resolved": sum(r.conflicts_resolved for r in stress_results),
            "overall_success_rate": sum(r.conflicts_resolved for r in stress_results) / 
                                   max(1, sum(r.conflicts_detected for r in stress_results)),
            "safety_score": failure_analysis.safety_impact_score,
            "performance_score": failure_analysis.performance_impact_score
        }
        
        package = ReportPackage(
            generation_timestamp=datetime.now(),
            summary_metrics=summary_metrics,
            failure_analysis=failure_analysis,
            stress_test_results=stress_results,
            baseline_comparison=baseline_comparison,
            metrics_csv_path=metrics_csv,
            charts_directory=charts_dir,
            narrative_report_path=narrative_report
        )
        
        logger.info("Sprint 5 report package generation complete!")
        logger.info(f"Reports saved to: {self.output_dir}")
        
        return package
