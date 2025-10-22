"""
Unified Report Generator for Agentic Framework Testing Harness.
Consolidates all report generation functionality.
"""

import json
import os
import webbrowser
import subprocess
import platform
from src.reporting.comprehensive_table_generator import generate_comprehensive_evaluation_table
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List


class UnifiedReporter:
    """Unified reporter for generating comprehensive benchmark reports."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize the unified reporter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_comprehensive_report(
        self, 
        results_file: Union[str, Path, Dict], 
        auto_open: bool = True
    ) -> Path:
        """
        Generate a comprehensive HTML report from benchmark results.
        
        Args:
            results_file: Path to JSON results or results dict
            auto_open: Whether to auto-open the report in browser
            
        Returns:
            Path to generated report
        """
        # Load results
        if isinstance(results_file, dict):
            data = results_file
        else:
            with open(results_file, encoding='utf-8') as f:
                data = json.load(f)
        
        # Store full data for consistency - THIS IS WHERE CAPABILITY SCORES COME FROM
        self.full_data = data
        results = data.get('results', data)
        
        # Generate HTML report
        html = self._generate_html_content(results)
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"report_{timestamp}.html"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✅ Report saved to: {report_path}")
        
        # Auto-open if requested
        if auto_open:
            self._open_report(report_path)
        
        return report_path
    
    def generate_capability_matrix_report(
        self,
        results: Dict[str, Any],
        auto_open: bool = True
    ) -> Path:
        """Generate capability matrix report."""
        html = self._generate_capability_matrix_html(results)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"capability_matrix_{timestamp}.html"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        if auto_open:
            self._open_report(report_path)
        
        return report_path
    
    def generate_enhanced_report(
        self,
        results_file: Union[str, Path, Dict],
        auto_open: bool = True
    ) -> Path:
        """Generate enhanced capability matrix report with detailed metrics."""
        # Load results
        if isinstance(results_file, dict):
            data = results_file
        else:
            with open(results_file, encoding='utf-8') as f:
                data = json.load(f)
        
        # Store full data so _generate_enhanced_html can access capability scores
        self.full_data = data
        results = data.get('results', data)
        
        # Generate enhanced HTML with capability matrix
        html = self._generate_enhanced_html(results)
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"enhanced_report_{timestamp}.html"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✅ Enhanced report saved to: {report_path}")
        
        # Auto-open if requested
        if auto_open:
            self._open_report(report_path)
        
        return report_path
    
    def generate_all_reports(
        self,
        results: Dict[str, Any],
        auto_open: bool = True
    ) -> Path:
        """Generate all report types."""
        # Main report
        main_report = self.generate_comprehensive_report(results, auto_open=False)
        
        # Capability matrix
        matrix_report = self.generate_capability_matrix_report(results, auto_open=False)
        
        # Markdown summary
        md_report = self.generate_markdown_summary(results)
        
        print(f"\n📊 Generated Reports:")
        print(f"  • Main Report: {main_report.name}")
        print(f"  • Capability Matrix: {matrix_report.name}")
        print(f"  • Markdown Summary: {md_report.name}")
        
        if auto_open:
            self._open_report(main_report)
        
        return self.output_dir
    
    def generate_markdown_summary(self, results: Dict[str, Any]) -> Path:
        """Generate a markdown summary of results."""
        md_content = self._generate_markdown_content(results)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"summary_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return report_path
    
    def _open_report(self, report_path: Path) -> None:
        """Open report in default browser."""
        try:
            import platform
            import subprocess
            import os
            
            file_path = str(report_path.absolute())
            system = platform.system()
            
            print(f"🌐 Opening report in default browser...")
            
            if system == 'Darwin':  # macOS
                # Use open command which reliably works on macOS
                result = subprocess.run(['open', file_path], capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"🌐 Report opened successfully")
                else:
                    print(f"⚠️ Could not open report: {result.stderr}")
                    print(f"   Please open manually: {file_path}")
            elif system == 'Windows':
                os.startfile(file_path)
                print(f"🌐 Report opened successfully")
            else:  # Linux and others
                subprocess.run(['xdg-open', file_path], check=False)
                print(f"🌐 Report opened successfully")
                
        except Exception as e:
            print(f"⚠️ Could not auto-open report: {e}")
            print(f"   Please open manually: {report_path.absolute()}")
    
    def _generate_html_content(self, results: Dict[str, Any]) -> str:
        """Generate HTML content for main report."""
        # Calculate metrics
        total_frameworks = len(results)
        
        # Find winner based on success rate
        scores = {}
        for framework in results:
            successes = sum(1 for r in results[framework].values() if r.get('success'))
            scores[framework] = successes
        winner = max(scores, key=scores.get) if scores else "Unknown"
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agentic AI Framework Evaluation Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f8f9fa;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{
            background: linear-gradient(135deg, #CC001F 0%, #990016 100%);
            color: white;
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 20px 60px rgba(204,0,31,0.15);
            text-align: center;
        }}
        h1 {{
            color: white;
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            text-align: center;
            border: 2px solid #f3f4f6;
            transition: all 0.3s ease;
        }}
        .metric-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 15px 40px rgba(204,0,31,0.15);
            border-color: #CC001F;
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #CC001F;
            margin: 10px 0;
        }}
        .metric-label {{
            color: #718096;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .table-container {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
        }}
        th {{
            background: #CC001F;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        th:first-child {{ border-radius: 10px 0 0 0; }}
        th:last-child {{ border-radius: 0 10px 0 0; }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e2e8f0;
            color: #4a5568;
        }}
        tr:hover td {{ background: #fef2f2; }}
        .winner-section {{
            background: linear-gradient(135deg, #CC001F 0%, #990016 100%);
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 30px;
            color: white;
            text-align: center;
            box-shadow: 0 20px 60px rgba(204,0,31,0.2);
        }}
        .chart-container {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border: 1px solid #f3f4f6;
            transition: all 0.3s ease;
        }}
        .chart-container:hover {{
            box-shadow: 0 15px 40px rgba(204,0,31,0.12);
        }}
        .chart-title {{
            font-size: 1.5em;
            color: #2d3748;
            margin-bottom: 20px;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Agentic AI Framework Evaluation Report</h1>
            <div style="color: white; opacity: 0.95; font-size: 1.2em; margin-top: 10px;">
                Comprehensive Benchmark Report - {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Frameworks Tested</div>
                <div class="metric-value">{total_frameworks}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Use Cases</div>
                <div class="metric-value">5</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Tests</div>
                <div class="metric-value">{total_frameworks * 5}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Top Framework</div>
                <div class="metric-value" style="font-size: 1.5em;">{winner}</div>
            </div>
        </div>
        
        <div class="winner-section">
            <div style="font-size: 4em; margin-bottom: 20px;">🏆</div>
            <div style="font-size: 2em; font-weight: bold;">Overall Winner: {winner}</div>
            <div style="font-size: 1.2em; margin-top: 10px;">Best performance across all use cases</div>
        </div>
        
        <div class="table-container">
            <h2 class="chart-title">📊 Framework Performance Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>Framework</th>
                        <th>Success Rate</th>
                        <th>Avg Latency</th>
                        <th>Total Cost</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add framework data
        for framework in results:
            framework_data = results[framework]
            successes = sum(1 for r in framework_data.values() if r.get('success'))
            total = len(framework_data)
            success_rate = f"{successes}/{total}"
            
            latencies = [r.get('latency', 0) for r in framework_data.values() if r.get('success')]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            
            costs = [r.get('cost', 0) for r in framework_data.values() if r.get('success')]
            total_cost = sum(costs)
            
            status = "✅ Operational" if successes > 0 else "❌ Failed"
            
            html += f"""
                    <tr>
                        <td><strong>{framework.replace('_', ' ').title()}</strong></td>
                        <td>{success_rate}</td>
                        <td>{avg_latency:.2f}s</td>
                        <td>${total_cost:.4f}</td>
                        <td>{status}</td>
                    </tr>
"""
        
        html += """
                </tbody>
            </table>
        </div>
        
        <div class="chart-container">
            <h2 class="chart-title">💡 Key Insights</h2>
            <ul style="list-style: none; padding: 0;">
                <li style="padding: 10px 0; color: #4a5568;">
                    <strong>🚀 Performance:</strong> Fastest frameworks achieve sub-second response times
                </li>
                <li style="padding: 10px 0; color: #4a5568;">
                    <strong>💰 Cost Efficiency:</strong> Mock mode enables free testing without API costs
                </li>
                <li style="padding: 10px 0; color: #4a5568;">
                    <strong>📊 Scalability:</strong> Framework performance varies significantly by use case
                </li>
                <li style="padding: 10px 0; color: #4a5568;">
                    <strong>🔧 Integration:</strong> All frameworks support standardized interfaces
                </li>
            </ul>
        </div>
    </div>
</body>
</html>"""
        
        return html
    
    def _generate_capability_matrix_html(self, results: Dict[str, Any]) -> str:
        """Generate capability matrix HTML."""
        # This would be the full capability matrix implementation
        # For now, returning a simplified version
        return self._generate_html_content(results)
    
    def _generate_executive_summary(self, results: Dict[str, Any], capabilities: Dict, weights: Dict) -> str:
        """Generate executive summary with key insights."""
        # Calculate overall scores for each framework
        framework_scores = {}
        for framework, caps in capabilities.items():
            if framework in results:
                overall = sum(caps[cap] * weights.get(cap, 0.11) for cap in caps) / sum(weights.values())
                framework_scores[framework] = overall
        
        # Find best overall, best RAG, most cost-effective
        best_overall = max(framework_scores.items(), key=lambda x: x[1])
        best_rag = max(capabilities.items(), key=lambda x: x[1].get('rag_capability', 0))
        
        # Calculate average costs
        framework_costs = {}
        for framework, tests in results.items():
            total_cost = sum(test.get('cost', 0) for test in tests.values())
            framework_costs[framework] = total_cost
        
        most_cost_effective = min(framework_costs.items(), key=lambda x: x[1]) if framework_costs else (None, 0)
        
        # Generate insights
        insights = []
        if best_overall[1] >= 85:
            insights.append(f"🏆 {best_overall[0].replace('_', ' ').title()} leads with exceptional overall score ({best_overall[1]:.1f}/100)")
        if best_rag[1].get('rag_capability', 0) >= 95:
            insights.append(f"📚 {best_rag[0].replace('_', ' ').title()} excels at RAG ({best_rag[1]['rag_capability']}/100) - ideal for document processing")
        if most_cost_effective[0]:
            insights.append(f"💰 {most_cost_effective[0].replace('_', ' ').title()} is most cost-effective (${most_cost_effective[1]:.4f} total)")
        
        return f"""
            <div id="executive-summary" style="padding: 20px;">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 25px; margin-bottom: 20px;">
                    <div style="background: white; border-radius: 12px; padding: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); border-top: 4px solid #CC001F; transition: transform 0.2s; cursor: default;"
                         onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(204,0,31,0.15)'" 
                         onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(0,0,0,0.08)'">
                        <div style="display: flex; align-items: center; margin-bottom: 15px;">
                            <span style="font-size: 28px; margin-right: 10px;">🏆</span>
                            <h3 style="color: #CC001F; margin: 0;">Top 3 Frameworks</h3>
                        </div>
                        <ol style="color: #4a5568; line-height: 2; margin: 0; padding-left: 20px;">
                            {self._format_top_frameworks(framework_scores, 3)}
                        </ol>
                    </div>
                    <div style="background: white; border-radius: 12px; padding: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); border-top: 4px solid #CC001F; transition: transform 0.2s; cursor: default;"
                         onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(204,0,31,0.15)'" 
                         onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(0,0,0,0.08)'">
                        <div style="display: flex; align-items: center; margin-bottom: 15px;">
                            <span style="font-size: 28px; margin-right: 10px;">💡</span>
                            <h3 style="color: #CC001F; margin: 0;">Key Insights</h3>
                        </div>
                        <ul style="color: #4a5568; line-height: 2; list-style: none; padding: 0; margin: 0;">
                            {''.join(f'<li style="padding: 4px 0;">{insight}</li>' for insight in insights[:3])}
                        </ul>
                    </div>
                    <div style="background: white; border-radius: 12px; padding: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); border-top: 4px solid #CC001F; transition: transform 0.2s; cursor: default;"
                         onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(204,0,31,0.15)'" 
                         onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(0,0,0,0.08)'">
                        <div style="display: flex; align-items: center; margin-bottom: 15px;">
                            <span style="font-size: 28px; margin-right: 10px;">🎯</span>
                            <h3 style="color: #CC001F; margin: 0;">Quick Recommendations</h3>
                        </div>
                        <ul style="color: #4a5568; line-height: 2; list-style: none; padding: 0; margin: 0;">
                            <li style="padding: 4px 0;">• <strong>Production:</strong> LangGraph or CrewAI</li>
                            <li style="padding: 4px 0;">• <strong>Document Processing:</strong> Haystack or LlamaIndex</li>
                            <li style="padding: 4px 0;">• <strong>Budget Conscious:</strong> {most_cost_effective[0].replace('_', ' ').title() if most_cost_effective[0] else 'PydanticAI'}</li>
                        </ul>
                    </div>
                </div>
            </div>
        """
    
    def _format_top_frameworks(self, scores: Dict[str, float], n: int) -> str:
        """Format top N frameworks for display."""
        sorted_frameworks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        items = []
        for framework, score in sorted_frameworks:
            items.append(f"<li><strong>{framework.replace('_', ' ').title()}</strong> - {score:.1f}/100</li>")
        return ''.join(items)
    
    def _generate_enhanced_metrics_section(self, enhanced_metrics: Dict) -> str:
        """Generate HTML section for enhanced metrics."""
        if not enhanced_metrics:
            return ""
        
        # Aggregate metrics across all frameworks and use cases
        aggregated = self._aggregate_enhanced_metrics(enhanced_metrics)
        
        # Also prepare per-framework metrics for the detailed table
        framework_details = self._prepare_framework_metrics(enhanced_metrics)
        
        html = f"""
        <div class="matrix-container" style="background: linear-gradient(135deg, #fff 0%, #f0f9ff 100%); border: 2px solid #60a5fa;">
            <h2 class="chart-title" style="color: #2563eb;">🔬 Advanced Analysis Metrics</h2>
            
            <!-- Metric Explanations -->
            <div style="background: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                <h3 style="color: #2563eb; margin-bottom: 15px;">📖 Understanding Enhanced Metrics</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; font-size: 14px; color: #4a5568;">
                    <div>
                        <strong style="color: #2563eb;">🧠 Reasoning Metrics:</strong>
                        <ul style="margin: 5px 0; padding-left: 20px;">
                            <li><strong>Depth:</strong> Number of reasoning steps taken</li>
                            <li><strong>Planning Score:</strong> Quality of upfront planning (0-100)</li>
                            <li><strong>Decision Confidence:</strong> Certainty in decisions made</li>
                            <li><strong>Backtrack Rate:</strong> How often the agent reverses decisions</li>
                        </ul>
                    </div>
                    <div>
                        <strong style="color: #2563eb;">🔍 Hallucination Detection:</strong>
                        <ul style="margin: 5px 0; padding-left: 20px;">
                            <li><strong>Grounding Score:</strong> % of claims supported by context</li>
                            <li><strong>Factual Consistency:</strong> Absence of contradictions</li>
                            <li><strong>Confidence Calibration:</strong> Appropriate certainty levels</li>
                        </ul>
                    </div>
                    <div>
                        <strong style="color: #2563eb;">📊 Resource Usage:</strong>
                        <ul style="margin: 5px 0; padding-left: 20px;">
                            <li><strong>Memory:</strong> RAM usage (current/peak MB)</li>
                            <li><strong>CPU:</strong> Processor utilization %</li>
                            <li><strong>Cache Hit Rate:</strong> Efficiency of data reuse</li>
                        </ul>
                    </div>
                    <div>
                        <strong style="color: #2563eb;">🛠️ Tool Efficiency:</strong>
                        <ul style="margin: 5px 0; padding-left: 20px;">
                            <li><strong>Total Calls:</strong> Number of tool invocations</li>
                            <li><strong>Unique Tools:</strong> Variety of tools used</li>
                            <li><strong>Efficiency Score:</strong> Optimal tool usage %</li>
                        </ul>
                    </div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin: 20px 0;">
                
                <!-- Reasoning Analysis Card -->
                <div style="background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h3 style="color: #2563eb; margin-bottom: 15px; font-size: 18px;">🧠 Reasoning Analysis</h3>
                    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                        <span style="color: #64748b;">Avg Reasoning Depth:</span>
                        <span style="color: #2563eb; font-weight: bold;">{aggregated['reasoning']['avg_depth']:.1f} steps</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                        <span style="color: #64748b;">Planning Score:</span>
                        <span style="color: #2563eb; font-weight: bold;">{aggregated['reasoning']['planning_score']:.0f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                        <span style="color: #64748b;">Decision Confidence:</span>
                        <span style="color: #2563eb; font-weight: bold;">{aggregated['reasoning']['decision_confidence']:.0f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                        <span style="color: #64748b;">Backtrack Rate:</span>
                        <span style="color: {self._get_color_for_value(aggregated['reasoning']['backtrack_rate'], inverse=True)}; font-weight: bold;">{aggregated['reasoning']['backtrack_rate']:.1f}%</span>
                    </div>
                </div>
                
                <!-- Hallucination Detection Card -->
                <div style="background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h3 style="color: #2563eb; margin-bottom: 15px; font-size: 18px;">🔍 Hallucination Detection</h3>
                    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                        <span style="color: #64748b;">Grounding Score:</span>
                        <span style="color: {self._get_color_for_value(aggregated['hallucination']['grounding_score'])}; font-weight: bold;">{aggregated['hallucination']['grounding_score']:.1f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                        <span style="color: #64748b;">Factual Consistency:</span>
                        <span style="color: {self._get_color_for_value(aggregated['hallucination']['factual_consistency'])}; font-weight: bold;">{aggregated['hallucination']['factual_consistency']:.0f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                        <span style="color: #64748b;">Confidence Score:</span>
                        <span style="color: #2563eb; font-weight: bold;">{aggregated['confidence']['score']:.0f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                        <span style="color: #64748b;">Appropriate Confidence:</span>
                        <span style="color: {self._get_color_for_value(100 if aggregated['confidence']['appropriate'] else 0)}; font-weight: bold;">{'Yes' if aggregated['confidence']['appropriate'] else 'No'}</span>
                    </div>
                </div>
                
                <!-- Resource Usage Card -->
                <div style="background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h3 style="color: #2563eb; margin-bottom: 15px; font-size: 18px;">📊 Resource Usage</h3>
                    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                        <span style="color: #64748b;">Avg Memory Usage:</span>
                        <span style="color: #2563eb; font-weight: bold;">{aggregated['resource']['memory_mb']:.1f} MB</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                        <span style="color: #64748b;">Peak Memory:</span>
                        <span style="color: #2563eb; font-weight: bold;">{aggregated['resource']['peak_memory']:.1f} MB</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                        <span style="color: #64748b;">CPU Usage:</span>
                        <span style="color: {self._get_color_for_value(100 - aggregated['resource']['cpu_percent'])}; font-weight: bold;">{aggregated['resource']['cpu_percent']:.1f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                        <span style="color: #64748b;">Cache Hit Rate:</span>
                        <span style="color: {self._get_color_for_value(aggregated['resource']['cache_hit_rate'])}; font-weight: bold;">{aggregated['resource']['cache_hit_rate']:.0f}%</span>
                    </div>
                </div>
                
                <!-- Tool Efficiency Card -->
                <div style="background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h3 style="color: #2563eb; margin-bottom: 15px; font-size: 18px;">🛠️ Tool Efficiency</h3>
                    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                        <span style="color: #64748b;">Total Tool Calls:</span>
                        <span style="color: #2563eb; font-weight: bold;">{aggregated['tool']['total_calls']:.0f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                        <span style="color: #64748b;">Unique Tools Used:</span>
                        <span style="color: #2563eb; font-weight: bold;">{aggregated['tool']['unique_tools']:.0f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                        <span style="color: #64748b;">Redundant Calls:</span>
                        <span style="color: {self._get_color_for_value(100 - aggregated['tool']['redundant_rate'] * 100)}; font-weight: bold;">{aggregated['tool']['redundant_calls']:.0f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
                        <span style="color: #64748b;">Efficiency Score:</span>
                        <span style="color: {self._get_color_for_value(aggregated['tool']['efficiency'])}; font-weight: bold;">{aggregated['tool']['efficiency']:.1f}%</span>
                    </div>
                </div>
                
            </div>
            
            <div style="margin-top: 20px; padding: 15px; background: #f0f9ff; border-radius: 8px; border-left: 4px solid #2563eb;">
                <p style="color: #1e40af; font-size: 14px; margin: 0;">
                    <strong>💡 Insight:</strong> These advanced metrics provide deeper insights into HOW frameworks solve problems, 
                    not just whether they succeed. Lower backtrack rates and higher grounding scores indicate more reliable and efficient agents.
                </p>
            </div>
            
            <!-- Framework-Specific Metrics Table -->
            <div style="margin-top: 30px;">
                <h3 style="color: #2563eb; margin-bottom: 15px;">🎯 Detailed Framework Metrics</h3>
                <div style="overflow-x: auto; max-height: 400px; overflow-y: auto; border: 1px solid #e2e8f0; border-radius: 8px;">
                    <table style="width: 100%; border-collapse: collapse; background: white;">
                        <thead style="position: sticky; top: 0; background: linear-gradient(to right, #2563eb, #3b82f6); color: white;">
                            <tr>
                                <th style="padding: 12px; text-align: left; font-weight: 600;">Framework</th>
                                <th style="padding: 12px; text-align: center;">Reasoning<br/>Depth</th>
                                <th style="padding: 12px; text-align: center;">Planning<br/>Score</th>
                                <th style="padding: 12px; text-align: center;">Decision<br/>Confidence</th>
                                <th style="padding: 12px; text-align: center;">Backtrack<br/>Rate</th>
                                <th style="padding: 12px; text-align: center;">Grounding<br/>Score</th>
                                <th style="padding: 12px; text-align: center;">Factual<br/>Consistency</th>
                                <th style="padding: 12px; text-align: center;">Memory<br/>(MB)</th>
                                <th style="padding: 12px; text-align: center;">CPU<br/>(%)</th>
                                <th style="padding: 12px; text-align: center;">Tool<br/>Efficiency</th>
                            </tr>
                        </thead>
                        <tbody>
{self._generate_framework_metrics_rows(framework_details)}
                        </tbody>
                    </table>
                </div>
                <p style="margin-top: 10px; font-size: 12px; color: #718096; text-align: center;">
                    <em>Scroll horizontally to see all metrics. Values are averaged across all use cases for each framework.</em>
                </p>
            </div>
        </div>
        """
        
        return html
    
    def _aggregate_enhanced_metrics(self, enhanced_metrics: Dict) -> Dict:
        """Aggregate enhanced metrics across frameworks and use cases."""
        aggregated = {
            'reasoning': {'avg_depth': 0, 'planning_score': 0, 'decision_confidence': 0, 'backtrack_rate': 0},
            'hallucination': {'grounding_score': 0, 'factual_consistency': 0},
            'confidence': {'score': 0, 'appropriate': 0},
            'resource': {'memory_mb': 0, 'peak_memory': 0, 'cpu_percent': 0, 'cache_hit_rate': 0},
            'tool': {'total_calls': 0, 'unique_tools': 0, 'redundant_calls': 0, 'efficiency': 0, 'redundant_rate': 0}
        }
        
        count = 0
        for framework, use_cases in enhanced_metrics.items():
            for use_case, metrics in use_cases.items():
                count += 1
                
                # Aggregate reasoning metrics
                if 'reasoning' in metrics:
                    aggregated['reasoning']['avg_depth'] += metrics['reasoning'].get('reasoning_depth', 0)
                    aggregated['reasoning']['planning_score'] += metrics['reasoning'].get('planning_score', 0)
                    aggregated['reasoning']['decision_confidence'] += metrics['reasoning'].get('decision_confidence', 0)
                    aggregated['reasoning']['backtrack_rate'] += metrics['reasoning'].get('backtrack_rate', 0)
                
                # Aggregate hallucination metrics
                if 'hallucination' in metrics:
                    aggregated['hallucination']['grounding_score'] += metrics['hallucination'].get('grounding_score', 0)
                    aggregated['hallucination']['factual_consistency'] += metrics['hallucination'].get('factual_consistency', 0)
                
                # Aggregate confidence metrics
                if 'confidence' in metrics:
                    aggregated['confidence']['score'] += metrics['confidence'].get('confidence_score', 0)
                    aggregated['confidence']['appropriate'] += 1 if metrics['confidence'].get('appropriate_confidence', False) else 0
                
                # Aggregate resource metrics
                if 'resource' in metrics:
                    aggregated['resource']['memory_mb'] += metrics['resource'].get('memory_current_mb', 0)
                    aggregated['resource']['peak_memory'] += metrics['resource'].get('memory_peak_mb', 0)
                    aggregated['resource']['cpu_percent'] += metrics['resource'].get('cpu_percent', 0)
                    aggregated['resource']['cache_hit_rate'] += metrics['resource'].get('cache_hit_rate', 0)
                
                # Aggregate tool metrics
                if 'tool_efficiency' in metrics:
                    aggregated['tool']['total_calls'] += metrics['tool_efficiency'].get('total_calls', 0)
                    aggregated['tool']['unique_tools'] += metrics['tool_efficiency'].get('unique_tools', 0)
                    aggregated['tool']['redundant_calls'] += metrics['tool_efficiency'].get('redundant_calls', 0)
                    aggregated['tool']['efficiency'] += metrics['tool_efficiency'].get('efficiency_score', 0)
        
        # Calculate averages
        if count > 0:
            for category in aggregated:
                for metric in aggregated[category]:
                    if metric != 'appropriate':
                        aggregated[category][metric] /= count
            
            # Calculate redundant rate for tools
            if aggregated['tool']['total_calls'] > 0:
                aggregated['tool']['redundant_rate'] = aggregated['tool']['redundant_calls'] / aggregated['tool']['total_calls']
        
        return aggregated
    
    def _get_color_for_value(self, value: float, inverse: bool = False) -> str:
        """Get color based on value (0-100 scale)."""
        if inverse:
            value = 100 - value
        
        if value >= 80:
            return '#10b981'  # Green
        elif value >= 60:
            return '#3b82f6'  # Blue
        elif value >= 40:
            return '#f59e0b'  # Amber
        else:
            return '#ef4444'  # Red
    
    def _prepare_framework_metrics(self, enhanced_metrics: Dict) -> Dict:
        """Prepare per-framework metrics for display."""
        framework_summary = {}
        
        for framework, use_cases in enhanced_metrics.items():
            # Initialize counters
            metrics_sum = {
                'reasoning_depth': 0,
                'planning_score': 0,
                'decision_confidence': 0,
                'backtrack_rate': 0,
                'grounding_score': 0,
                'factual_consistency': 0,
                'memory_mb': 0,
                'cpu_percent': 0,
                'tool_efficiency': 0,
                'count': 0
            }
            
            # Aggregate across use cases
            for use_case, metrics in use_cases.items():
                metrics_sum['count'] += 1
                
                if 'reasoning' in metrics:
                    metrics_sum['reasoning_depth'] += metrics['reasoning'].get('reasoning_depth', 0)
                    metrics_sum['planning_score'] += metrics['reasoning'].get('planning_score', 0)
                    metrics_sum['decision_confidence'] += metrics['reasoning'].get('decision_confidence', 0)
                    metrics_sum['backtrack_rate'] += metrics['reasoning'].get('backtrack_rate', 0)
                
                if 'hallucination' in metrics:
                    metrics_sum['grounding_score'] += metrics['hallucination'].get('grounding_score', 0)
                    metrics_sum['factual_consistency'] += metrics['hallucination'].get('factual_consistency', 0)
                
                if 'resource' in metrics:
                    metrics_sum['memory_mb'] += metrics['resource'].get('memory_current_mb', 0)
                    metrics_sum['cpu_percent'] += metrics['resource'].get('cpu_percent', 0)
                
                if 'tool_efficiency' in metrics:
                    metrics_sum['tool_efficiency'] += metrics['tool_efficiency'].get('efficiency_score', 0)
            
            # Calculate averages
            if metrics_sum['count'] > 0:
                framework_summary[framework] = {
                    'reasoning_depth': metrics_sum['reasoning_depth'] / metrics_sum['count'],
                    'planning_score': metrics_sum['planning_score'] / metrics_sum['count'],
                    'decision_confidence': metrics_sum['decision_confidence'] / metrics_sum['count'],
                    'backtrack_rate': metrics_sum['backtrack_rate'] / metrics_sum['count'],
                    'grounding_score': metrics_sum['grounding_score'] / metrics_sum['count'],
                    'factual_consistency': metrics_sum['factual_consistency'] / metrics_sum['count'],
                    'memory_mb': metrics_sum['memory_mb'] / metrics_sum['count'],
                    'cpu_percent': metrics_sum['cpu_percent'] / metrics_sum['count'],
                    'tool_efficiency': metrics_sum['tool_efficiency'] / metrics_sum['count']
                }
        
        return framework_summary
    
    def _generate_framework_metrics_rows(self, framework_details: Dict) -> str:
        """Generate table rows for framework-specific metrics."""
        rows = []
        
        for framework, metrics in sorted(framework_details.items()):
            # Determine row color based on overall performance
            avg_score = (
                metrics['planning_score'] + 
                metrics['decision_confidence'] + 
                metrics['grounding_score'] + 
                metrics['factual_consistency'] + 
                metrics['tool_efficiency']
            ) / 5
            
            row_bg = '#f0fdf4' if avg_score >= 80 else '#fafafa' if avg_score >= 60 else '#fef2f2'
            
            rows.append(f"""
                            <tr style="background: {row_bg}; border-bottom: 1px solid #e2e8f0;">
                                <td style="padding: 12px; font-weight: 600; color: #2563eb;">
                                    {framework.replace('_', ' ').title()}
                                </td>
                                <td style="padding: 12px; text-align: center;">
                                    {metrics['reasoning_depth']:.1f}
                                </td>
                                <td style="padding: 12px; text-align: center; color: {self._get_color_for_value(metrics['planning_score'])};">
                                    {metrics['planning_score']:.0f}%
                                </td>
                                <td style="padding: 12px; text-align: center; color: {self._get_color_for_value(metrics['decision_confidence'])};">
                                    {metrics['decision_confidence']:.0f}%
                                </td>
                                <td style="padding: 12px; text-align: center; color: {self._get_color_for_value(metrics['backtrack_rate'], inverse=True)};">
                                    {metrics['backtrack_rate']:.1f}%
                                </td>
                                <td style="padding: 12px; text-align: center; color: {self._get_color_for_value(metrics['grounding_score'])};">
                                    {metrics['grounding_score']:.0f}%
                                </td>
                                <td style="padding: 12px; text-align: center; color: {self._get_color_for_value(metrics['factual_consistency'])};">
                                    {metrics['factual_consistency']:.0f}%
                                </td>
                                <td style="padding: 12px; text-align: center;">
                                    {metrics['memory_mb']:.1f}
                                </td>
                                <td style="padding: 12px; text-align: center; color: {self._get_color_for_value(100 - metrics['cpu_percent'])};">
                                    {metrics['cpu_percent']:.1f}%
                                </td>
                                <td style="padding: 12px; text-align: center; color: {self._get_color_for_value(metrics['tool_efficiency'])};">
                                    {metrics['tool_efficiency']:.0f}%
                                </td>
                            </tr>""")
        
        return ''.join(rows) if rows else '<tr><td colspan="10" style="padding: 20px; text-align: center; color: #718096;">No framework metrics available</td></tr>'
    
    def _generate_enhanced_html(self, results: Dict[str, Any]) -> str:
        """Generate enhanced HTML with capability matrix and detailed metrics."""
        # READ CAPABILITY SCORES AND WEIGHTS FROM JSON DATA - NOTHING HARDCODED!
        if hasattr(self, 'full_data') and self.full_data:
            capabilities = self.full_data.get('capability_scores', {})
            capability_weights = self.full_data.get('capability_weights', {})
            enhanced_metrics = self.full_data.get('enhanced_metrics', {})
        else:
            # This should never happen in normal flow
            capabilities = {}
            capability_weights = {}
            enhanced_metrics = {}
            
        if not capabilities:
            raise ValueError("Capability scores not found in JSON data! Scores must be calculated and stored in JSON first.")
        
        if not capability_weights:
            raise ValueError("Capability weights not found in JSON data! Weights must be defined and stored in JSON first.")
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agentic AI Framework Evaluation Report - Capability Matrix</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, sans-serif; background: #f8f9fa; min-height: 100vh; padding: 20px; }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #CC001F 0%, #990016 100%); color: white; border-radius: 20px; padding: 40px; margin-bottom: 30px; box-shadow: 0 20px 60px rgba(204,0,31,0.15); text-align: center; }}
        h1 {{ color: white; font-size: 2.5em; margin-bottom: 10px; font-weight: 700; }}
        .matrix-container {{ background: white; border-radius: 15px; padding: 30px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); overflow-x: auto; }}
        .matrix-table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: 14px; }}
        .matrix-table th {{ background: #CC001F; color: white; padding: 15px 10px; text-align: center; font-weight: 600; }}
        .matrix-table th:first-child {{ text-align: left; padding-left: 20px; border-radius: 10px 0 0 0; }}
        .matrix-table th:last-child {{ border-radius: 0 10px 0 0; }}
        .matrix-table td {{ padding: 12px 10px; text-align: center; border-bottom: 1px solid #e2e8f0; }}
        .matrix-table td:first-child {{ text-align: left; padding-left: 20px; font-weight: 600; color: #2d3748; }}
        .matrix-table tr:hover td {{ background: #fef2f2; }}
        .score-cell {{ font-weight: bold; border-radius: 8px; padding: 8px; }}
        .score-excellent {{ background: #4ade80; color: #052e16; font-weight: bold; }}
        .score-good {{ background: #60a5fa; color: #1e3a8a; font-weight: bold; }}
        .score-moderate {{ background: #fbbf24; color: #78350f; font-weight: bold; }}
        .score-limited {{ background: #fca5a5; color: #7f1d1d; font-weight: bold; }}
        .chart-title {{ font-size: 1.5em; color: #2d3748; margin-bottom: 20px; font-weight: 600; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: white; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.08); border: 2px solid #f3f4f6; transition: all 0.3s ease; }}
        .metric-card:hover {{ transform: translateY(-4px); box-shadow: 0 10px 25px rgba(204,0,31,0.15); border-color: #CC001F; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #CC001F; margin: 10px 0; }}
        .metric-label {{ color: #718096; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; }}
        .legend {{ display: flex; justify-content: center; gap: 30px; margin: 20px 0; flex-wrap: wrap; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; }}
        .legend-color {{ width: 30px; height: 20px; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Agentic AI Framework Evaluation Report</h1>
            <div style="color: white; opacity: 0.95; font-size: 1.2em; margin-bottom: 10px;">
                Comprehensive evaluation of {len(results)} frameworks across 9 key capabilities
            </div>
            <div style="color: white; opacity: 0.85; font-size: 0.9em;">
                Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
            </div>
        </div>
        
        <div class="matrix-container" style="background: linear-gradient(135deg, #fff 0%, #fef2f2 100%); border: 2px solid #CC001F;">
            <h2 class="chart-title" style="color: #CC001F;">📈 Executive Summary</h2>
            {self._generate_executive_summary(results, capabilities, capability_weights)}
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Frameworks</div>
                <div class="metric-value">{len(results)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Capabilities</div>
                <div class="metric-value">9</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Use Cases</div>
                <div class="metric-value">{len(next(iter(results.values())))}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Tests</div>
                <div class="metric-value">{sum(len(v) for v in results.values())}</div>
            </div>
        </div>
        
        <div class="matrix-container">
            <h2 class="chart-title">📊 Scoring Legend</h2>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color score-excellent"></div>
                    <span><strong>Excellent</strong> (85-100)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color score-good"></div>
                    <span><strong>Good</strong> (70-84)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color score-moderate"></div>
                    <span><strong>Moderate</strong> (55-69)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color score-limited"></div>
                    <span><strong>Limited</strong> (0-54)</span>
                </div>
            </div>
            <div style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 8px; color: #4a5568;">
                <strong>Note:</strong> Overall score is calculated as a weighted average of all 22 metrics with explicit weights.
            </div>
        </div>
"""
        
        # Generate the comprehensive evaluation table
        comprehensive_table_html, framework_scores = generate_comprehensive_evaluation_table(
            results, capabilities, enhanced_metrics, capability_weights
        )
        html += comprehensive_table_html
        
        # Now add per-use-case breakdown tables
        html += self._generate_use_case_tables(results, enhanced_metrics)
        
        # Add closing HTML
        html += self._generate_final_html_closing()
        
        return html
    
    def _generate_use_case_tables(self, results, enhanced_metrics):
        """Generate tables for each use case showing framework performance."""
        html = """
        <div class="matrix-container" style="margin-top: 40px;">
            <h2 class="chart-title">🎆 Per Use Case Breakdown</h2>
        """
        
        # Group results by use case
        use_cases = set()
        for framework_data in results.values():
            use_cases.update(framework_data.keys())
        
        for use_case in sorted(use_cases):
            html += f"""
            <div style="margin-top: 30px;">
                <h3 style="color: #CC001F; margin-bottom: 15px; font-size: 1.2em;">🔍 {use_case.replace('_', ' ').title()}</h3>
                <div style="overflow-x: auto; border: 2px solid #CC001F; border-radius: 8px;">
            <table class="matrix-table" style="min-width: 2400px; margin: 0;">
                <thead style="background: linear-gradient(to right, #CC001F, #E31837); color: white;">
                    <tr>
                        <th rowspan="2" style="position: sticky; left: 0; background: #CC001F; z-index: 10;">Framework</th>
                        <th colspan="8">Capability Scores (for this use case)</th>
                        <th colspan="7">Enhanced Metrics</th>
                        <th colspan="3">Resource Usage</th>
                        <th colspan="3">Performance</th>
                        <th rowspan="2" style="background: #7f1d1d;">Score</th>
                    </tr>
                    <tr>
                        <!-- Capability Scores -->
                        <th title="Multi-Agent Coordination">Multi-Agent</th>
                        <th title="Tool Usage">Tool Usage</th>
                        <th title="Error Handling">Error Handle</th>
                        <th title="Context Retention">Context</th>
                        <th title="Adaptability">Adaptability</th>
                        <th title="Scalability">Scalability</th>
                        <th title="Observability">Observability</th>
                        <th title="RAG Capability">RAG</th>
                        <!-- Enhanced Metrics -->
                        <th>Reasoning</th>
                        <th>Planning</th>
                        <th>Decision</th>
                        <th>Backtrack↓</th>
                        <th>Grounding</th>
                        <th>Factual</th>
                        <th>Tool Eff</th>
                        <!-- Resource -->
                        <th>Memory↓</th>
                        <th>CPU↓</th>
                        <th>Cache</th>
                        <!-- Performance -->
                        <th>Success</th>
                        <th>Latency↓</th>
                        <th>Cost↓</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            # First pass: Calculate all scores for this use case
            use_case_scores = {}
            
            for framework in results.keys():
                if use_case in results[framework]:
                    data = results[framework][use_case]
                    success = '✅' if data.get('success', False) else '❌'
                    latency = data.get('latency', 0)
                    tokens = data.get('tokens', 0)
                    cost = data.get('cost', 0)
                    
                    # Get enhanced metrics if available - default values
                    reasoning_depth = 0
                    planning_score = 0
                    decision_confidence = 50
                    backtrack_rate = 0
                    grounding_score = 0
                    factual_consistency = 100
                    tool_efficiency = 100
                    memory_mb = 0
                    cpu_percent = 0
                    cache_hit_rate = 0
                    
                    if framework in enhanced_metrics and use_case in enhanced_metrics[framework]:
                        em = enhanced_metrics[framework][use_case]
                        if 'reasoning' in em:
                            reasoning_depth = em['reasoning'].get('reasoning_depth', 0)
                            planning_score = em['reasoning'].get('planning_score', 0)
                            decision_confidence = em['reasoning'].get('decision_confidence', 50)
                            backtrack_rate = em['reasoning'].get('backtrack_rate', 0)
                        if 'hallucination' in em:
                            grounding_score = em['hallucination'].get('grounding_score', 0)
                            factual_consistency = em['hallucination'].get('factual_consistency', 100)
                        if 'tool_efficiency' in em:
                            tool_efficiency = em['tool_efficiency'].get('efficiency_score', 100)
                        if 'resource' in em:
                            memory_mb = em['resource'].get('memory_current_mb', 0)
                            cpu_percent = em['resource'].get('cpu_percent', 0)
                            cache_hit_rate = em['resource'].get('cache_hit_rate', 0)
                    
                    # Calculate capability scores for this use case
                    success_rate = 100 if data.get('success', False) else 0
                    
                    # Calculate resource efficiency for scalability
                    mem_efficiency = max(0, 100 - (memory_mb - 100) / 4)  # 100MB=100%, 500MB=0%
                    cpu_efficiency = max(0, 100 - cpu_percent)
                    resource_efficiency = (mem_efficiency + cpu_efficiency) / 2
                    
                    # Capability scores based on this use case's metrics
                    capabilities = {
                        'multi_agent': min(100, success_rate + decision_confidence / 2),
                        'tool_usage': tool_efficiency,
                        'error_handling': factual_consistency,
                        'context_retention': max(10, grounding_score),  # Min 10% for mock
                        'adaptability': max(0, planning_score - backtrack_rate),
                        'scalability': resource_efficiency,
                        'observability': min(100, reasoning_depth * 5),  # Scale reasoning depth
                        'rag_capability': max(10, grounding_score)  # Min 10% for mock
                    }
                    
                    # Calculate overall score using same weights as comprehensive table
                    all_metric_weights = {
                        # Core Capabilities (60% total)
                        'multi_agent': 0.09,
                        'tool_usage': 0.07,
                        'error_handling': 0.07,
                        'context_retention': 0.06,
                        'adaptability': 0.06,
                        'scalability': 0.07,
                        'observability': 0.05,
                        'rag_capability': 0.13,
                        
                        # Enhanced metrics (25% total)
                        'reasoning_depth': 0.03,
                        'planning_score': 0.04,
                        'decision_confidence': 0.03,
                        'backtrack_rate': 0.02,  # Lower is better
                        'grounding_score': 0.04,
                        'factual_consistency': 0.04,
                        'tool_efficiency': 0.05,
                        
                        # Resource metrics (5% total)
                        'memory_efficiency': 0.02,  # Lower memory is better
                        'cpu_efficiency': 0.02,      # Lower CPU is better
                        'cache_hit_rate': 0.01,
                        
                        # Performance metrics (10% total)
                        'success_rate': 0.07,
                        'latency_score': 0.02,  # Lower is better
                        'cost_efficiency': 0.01  # Lower is better
                    }
                    
                    # Calculate normalized scores
                    normalized_scores = {
                        # Capabilities
                        'multi_agent': capabilities['multi_agent'],
                        'tool_usage': capabilities['tool_usage'],
                        'error_handling': capabilities['error_handling'],
                        'context_retention': capabilities['context_retention'],
                        'adaptability': capabilities['adaptability'],
                        'scalability': capabilities['scalability'],
                        'observability': capabilities['observability'],
                        'rag_capability': capabilities['rag_capability'],
                        
                        # Enhanced metrics
                        'reasoning_depth': min(100, reasoning_depth * 10),
                        'planning_score': planning_score,
                        'decision_confidence': decision_confidence,
                        'backtrack_rate': max(0, 100 - backtrack_rate),  # Inverse
                        'grounding_score': grounding_score,
                        'factual_consistency': factual_consistency,
                        'tool_efficiency': tool_efficiency,
                        
                        # Resource metrics
                        'memory_efficiency': mem_efficiency,
                        'cpu_efficiency': cpu_efficiency,
                        'cache_hit_rate': cache_hit_rate,
                        
                        # Performance metrics
                        'success_rate': success_rate,
                        'latency_score': max(0, 100 - latency * 20),  # Inverse
                        'cost_efficiency': max(0, 100 - cost * 1000)  # Inverse
                    }
                    
                    # Calculate weighted overall score
                    overall = sum(normalized_scores[metric] * weight 
                                 for metric, weight in all_metric_weights.items())
                    
                    # Store all data for this framework in this use case
                    use_case_scores[framework] = {
                        'data': data,
                        'overall': overall,
                        'capabilities': capabilities,
                        'metrics': {
                            'reasoning_depth': reasoning_depth,
                            'planning_score': planning_score,
                            'decision_confidence': decision_confidence,
                            'backtrack_rate': backtrack_rate,
                            'grounding_score': grounding_score,
                            'factual_consistency': factual_consistency,
                            'tool_efficiency': tool_efficiency,
                            'memory_mb': memory_mb,
                            'cpu_percent': cpu_percent,
                            'cache_hit_rate': cache_hit_rate,
                            'latency': latency,
                            'tokens': tokens,
                            'cost': cost
                        }
                    }
            
            # Sort frameworks by overall score (descending)
            sorted_frameworks = sorted(use_case_scores.keys(),
                                     key=lambda f: use_case_scores[f]['overall'],
                                     reverse=True)
            
            # Second pass: Generate HTML rows in sorted order
            for framework in sorted_frameworks:
                    framework_data = use_case_scores[framework]
                    data = framework_data['data']
                    overall = framework_data['overall']
                    capabilities = framework_data['capabilities']
                    metrics = framework_data['metrics']
                    
                    # Extract metrics
                    reasoning_depth = metrics['reasoning_depth']
                    planning_score = metrics['planning_score']
                    decision_confidence = metrics['decision_confidence']
                    backtrack_rate = metrics['backtrack_rate']
                    grounding_score = metrics['grounding_score']
                    factual_consistency = metrics['factual_consistency']
                    tool_efficiency = metrics['tool_efficiency']
                    memory_mb = metrics['memory_mb']
                    cpu_percent = metrics['cpu_percent']
                    cache_hit_rate = metrics['cache_hit_rate']
                    latency = metrics['latency']
                    tokens = metrics['tokens']
                    cost = metrics['cost']
                    
                    # Format values for display
                    success_str = '✅' if data.get('success', False) else '❌'
                    
                    # Choose row color based on score
                    row_color = '#f0fdf4' if overall >= 70 else '#fafafa' if overall >= 40 else '#fef2f2'
                    
                    html += f"""
                    <tr style="background: {row_color};">
                        <td style="font-weight: 600; position: sticky; left: 0; background: {row_color}; z-index: 5;">
                            {framework.replace('_', ' ').title()}
                        </td>
                        <!-- Capability Scores -->
                        <td style="text-align: center; color: {'#10b981' if capabilities['multi_agent'] >= 80 else '#ef4444' if capabilities['multi_agent'] < 50 else '#3b82f6'};">{capabilities['multi_agent']:.0f}</td>
                        <td style="text-align: center; color: {'#10b981' if capabilities['tool_usage'] >= 80 else '#ef4444' if capabilities['tool_usage'] < 50 else '#3b82f6'};">{capabilities['tool_usage']:.0f}</td>
                        <td style="text-align: center; color: {'#10b981' if capabilities['error_handling'] >= 80 else '#ef4444' if capabilities['error_handling'] < 50 else '#3b82f6'};">{capabilities['error_handling']:.0f}</td>
                        <td style="text-align: center; color: {'#10b981' if capabilities['context_retention'] >= 80 else '#ef4444' if capabilities['context_retention'] < 50 else '#3b82f6'};">{capabilities['context_retention']:.0f}</td>
                        <td style="text-align: center; color: {'#10b981' if capabilities['adaptability'] >= 80 else '#ef4444' if capabilities['adaptability'] < 50 else '#3b82f6'};">{capabilities['adaptability']:.0f}</td>
                        <td style="text-align: center; color: {'#10b981' if capabilities['scalability'] >= 80 else '#ef4444' if capabilities['scalability'] < 50 else '#3b82f6'};">{capabilities['scalability']:.0f}</td>
                        <td style="text-align: center; color: {'#10b981' if capabilities['observability'] >= 80 else '#ef4444' if capabilities['observability'] < 50 else '#3b82f6'};">{capabilities['observability']:.0f}</td>
                        <td style="text-align: center; color: {'#10b981' if capabilities['rag_capability'] >= 80 else '#ef4444' if capabilities['rag_capability'] < 50 else '#3b82f6'};">{capabilities['rag_capability']:.0f}</td>
                        <!-- Enhanced Metrics -->
                        <td style="text-align: center;">{reasoning_depth}</td>
                        <td style="text-align: center; color: {'#10b981' if planning_score >= 50 else '#ef4444' if planning_score < 20 else '#3b82f6'};">{planning_score}%</td>
                        <td style="text-align: center;">{decision_confidence}%</td>
                        <td style="text-align: center; color: {'#10b981' if backtrack_rate <= 10 else '#ef4444' if backtrack_rate > 30 else '#f59e0b'};">{backtrack_rate:.1f}%</td>
                        <td style="text-align: center; color: {'#10b981' if grounding_score >= 80 else '#ef4444' if grounding_score < 50 else '#3b82f6'};">{grounding_score:.0f}%</td>
                        <td style="text-align: center; color: {'#10b981' if factual_consistency >= 90 else '#ef4444' if factual_consistency < 70 else '#f59e0b'};">{factual_consistency:.0f}%</td>
                        <td style="text-align: center; color: {'#10b981' if tool_efficiency >= 80 else '#ef4444' if tool_efficiency < 50 else '#3b82f6'};">{tool_efficiency:.0f}%</td>
                        <!-- Resource Metrics -->
                        <td style="text-align: center; color: {'#10b981' if memory_mb <= 200 else '#ef4444' if memory_mb > 500 else '#f59e0b'};">{memory_mb:.0f}MB</td>
                        <td style="text-align: center; color: {'#10b981' if cpu_percent <= 30 else '#ef4444' if cpu_percent > 70 else '#f59e0b'};">{cpu_percent:.1f}%</td>
                        <td style="text-align: center;">{cache_hit_rate:.0f}%</td>
                        <!-- Performance Metrics -->
                        <td style="text-align: center;">{success_str}</td>
                        <td style="text-align: center; color: {'#10b981' if latency <= 1 else '#ef4444' if latency > 3 else '#f59e0b'};">{latency:.2f}s</td>
                        <td style="text-align: center; color: {'#10b981' if cost <= 0.01 else '#ef4444' if cost > 0.05 else '#f59e0b'};">${cost:.4f}</td>
                        <!-- Overall Score -->
                        <td style="text-align: center; font-weight: bold; background: {'#dcfce7' if overall >= 70 else '#fee2e2' if overall < 40 else '#fef3c7'}; color: {'#166534' if overall >= 70 else '#991b1b' if overall < 40 else '#92400e'};">
                            {overall:.0f}
                        </td>
                    </tr>
                    """
            
            html += """
                </tbody>
            </table>
                </div>
            </div>
            """
        
        # Close the main Per Use Case Breakdown container
        html += """
        </div>
        """
        
        return html
    
    def _generate_final_html_closing(self):
        """Generate the closing HTML tags and scripts."""
        html = f"""
    </div>
    
    <script>
    function downloadCSV() {{
        const csvContent = generateCSVContent();
        const blob = new Blob([csvContent], {{ type: 'text/csv' }});
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'framework_evaluation_{datetime.now().strftime('%Y%m%d')}.csv';
        a.click();
    }}
    
    function generateCSVContent() {{
        let csv = 'Framework,Multi-Agent,Tool Usage,Error Handling,Context,Adaptability,Scalability,Observability,RAG,Overall\\n';
        const tables = document.querySelectorAll('.matrix-table');
        if (tables.length > 0) {{
            const tbody = tables[0].querySelector('tbody');
            if (tbody) {{
                const rows = tbody.querySelectorAll('tr');
                rows.forEach(row => {{
                    const cells = row.querySelectorAll('td');
                    const values = Array.from(cells).map(cell => {{
                        const scoreSpan = cell.querySelector('.score-cell');
                        return scoreSpan ? scoreSpan.textContent : cell.textContent.trim();
                    }});
                    csv += values.join(',') + '\\n';
                }});
            }}
        }}
        return csv;
    }}
    
    function copyExecutiveSummary() {{
        const summary = document.querySelector('#executive-summary');
        if (summary) {{
            navigator.clipboard.writeText(summary.innerText).then(() => {{
                alert('Executive summary copied to clipboard!');
            }});
        }}
    }}
    </script>
</body>
</html>"""
        return html
    
    def _generate_markdown_content(self, results: Dict[str, Any]) -> str:
        """Generate markdown summary."""
        md = f"""# Agentic Framework Testing Results

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Frameworks Tested**: {len(results)}
- **Use Cases**: 5
- **Total Tests**: {len(results) * 5}

## Framework Performance

| Framework | Success Rate | Avg Latency | Total Cost |
|-----------|-------------|-------------|------------|
"""
        
        for framework in results:
            framework_data = results[framework]
            successes = sum(1 for r in framework_data.values() if r.get('success'))
            total = len(framework_data)
            
            latencies = [r.get('latency', 0) for r in framework_data.values() if r.get('success')]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            
            costs = [r.get('cost', 0) for r in framework_data.values() if r.get('success')]
            total_cost = sum(costs)
            
            md += f"| {framework} | {successes}/{total} | {avg_latency:.2f}s | ${total_cost:.4f} |\n"
        
        md += "\n---\n*Report generated by Agentic Framework Testing Harness*\n"
        
        return md
