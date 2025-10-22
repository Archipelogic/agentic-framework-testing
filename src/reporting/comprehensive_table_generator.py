"""
Comprehensive table generator for framework evaluation.
ALL metrics in ONE table with proper weights.
"""

def generate_comprehensive_evaluation_table(results, capabilities, enhanced_metrics, capability_weights):
    """
    Generate ONE comprehensive table with ALL metrics that matter.
    Every metric has a weight. Everything factors into the overall score.
    """
    
    # Define weights for ALL metrics (must sum to 1.0)
    all_metric_weights = {
        # Original capability scores (60% total)
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
    
    # Verify weights sum to 1.0
    total_weight = sum(all_metric_weights.values())
    assert abs(total_weight - 1.0) < 0.01, f"Weights must sum to 1.0, got {total_weight}"
    
    html = """
    <div class="matrix-container">
        <h2 class="chart-title">🎯 Overall Framework Evaluation</h2>
        
        <!-- Metric Explanations -->
        <details style="margin: 20px 0; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
            <summary style="cursor: pointer; color: #2563eb; font-weight: bold; font-size: 18px;">📖 Click to View Metric Descriptions</summary>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; margin-top: 20px; font-size: 14px; color: #4a5568;">
                <div>
                    <strong style="color: #CC001F;">🎯 Core Capabilities (60% total):</strong>
                    <ul style="margin: 10px 0; padding-left: 20px; list-style: disc;">
                        <li><strong>Multi-Agent (9%):</strong> Orchestrating multiple AI agents</li>
                        <li><strong>Tool Usage (7%):</strong> External tool integration</li>
                        <li><strong>Error Handling (7%):</strong> Robustness & recovery</li>
                        <li><strong>Context (6%):</strong> Memory & state management</li>
                        <li><strong>Adaptability (6%):</strong> Flexibility to requirements</li>
                        <li><strong>Scalability (7%):</strong> Performance under load</li>
                        <li><strong>Observability (5%):</strong> Logging & debugging</li>
                        <li><strong>RAG (13%):</strong> Document retrieval & grounding</li>
                    </ul>
                </div>
                <div>
                    <strong style="color: #CC001F;">🧠 Enhanced Metrics (25% total):</strong>
                    <ul style="margin: 10px 0; padding-left: 20px; list-style: disc;">
                        <li><strong>Reasoning (3%):</strong> Depth of analysis steps</li>
                        <li><strong>Planning (4%):</strong> Quality of upfront planning</li>
                        <li><strong>Decision (3%):</strong> Confidence in choices</li>
                        <li><strong>Backtrack↓ (2%):</strong> How often decisions reversed</li>
                        <li><strong>Grounding (4%):</strong> Claims supported by context</li>
                        <li><strong>Factual (4%):</strong> Consistency without contradictions</li>
                        <li><strong>Tool Eff (5%):</strong> Optimal tool usage</li>
                    </ul>
                </div>
                <div>
                    <strong style="color: #CC001F;">📊 Performance & Resources (15% total):</strong>
                    <ul style="margin: 10px 0; padding-left: 20px; list-style: disc;">
                        <li><strong>Success (7%):</strong> % of tests passing</li>
                        <li><strong>Latency↓ (2%):</strong> Speed (lower is better)</li>
                        <li><strong>Cost↓ (1%):</strong> Token costs (lower is better)</li>
                        <li><strong>Memory↓ (2%):</strong> RAM usage (lower is better)</li>
                        <li><strong>CPU↓ (2%):</strong> Processor usage (lower is better)</li>
                        <li><strong>Cache (1%):</strong> Data reuse efficiency</li>
                    </ul>
                </div>
            </div>
            <p style="margin-top: 15px; padding: 10px; background: #fef2f2; border-left: 4px solid #CC001F; color: #991b1b;">
                <strong>📊 Note:</strong> Metrics marked with ↓ are inverse - lower values result in higher scores.
                All metrics are normalized to 0-100 scale before weighting.
            </p>
        </details>
        
        <p style="text-align: center; color: #718096; margin: 15px 0;">
            Every metric below contributes to the final score. Weights are shown in headers. Scroll to see all metrics.
        </p>
        <div style="overflow-x: auto; border: 2px solid #CC001F; border-radius: 8px;">
        <table class="matrix-table" style="min-width: 2400px; margin: 0;">
            <thead style="background: linear-gradient(to right, #CC001F, #E31837); color: white;">
                <tr>
                    <th rowspan="2" style="position: sticky; left: 0; background: #CC001F; z-index: 10;">Framework</th>
                    <th colspan="8">Core Capabilities</th>
                    <th colspan="7">Reasoning & Accuracy</th>
                    <th colspan="3">Resource Usage</th>
                    <th colspan="3">Performance</th>
                    <th rowspan="2" style="background: #7f1d1d; font-size: 16px; font-weight: bold;">OVERALL<br>SCORE</th>
                </tr>
                <tr>
                    <!-- Core Capabilities -->
                    <th title="Weight: 9%">Multi-Agent<br><small>(9%)</small></th>
                    <th title="Weight: 7%">Tool Usage<br><small>(7%)</small></th>
                    <th title="Weight: 7%">Error Handle<br><small>(7%)</small></th>
                    <th title="Weight: 6%">Context<br><small>(6%)</small></th>
                    <th title="Weight: 6%">Adaptability<br><small>(6%)</small></th>
                    <th title="Weight: 7%">Scalability<br><small>(7%)</small></th>
                    <th title="Weight: 5%">Observability<br><small>(5%)</small></th>
                    <th title="Weight: 13%">RAG<br><small>(13%)</small></th>
                    
                    <!-- Enhanced Metrics -->
                    <th title="Weight: 3%">Reasoning<br><small>(3%)</small></th>
                    <th title="Weight: 4%">Planning<br><small>(4%)</small></th>
                    <th title="Weight: 3%">Decision<br><small>(3%)</small></th>
                    <th title="Weight: 2% - Lower is better">Backtrack↓<br><small>(2%)</small></th>
                    <th title="Weight: 4%">Grounding<br><small>(4%)</small></th>
                    <th title="Weight: 4%">Factual<br><small>(4%)</small></th>
                    <th title="Weight: 5%">Tool Eff<br><small>(5%)</small></th>
                    
                    <!-- Resource Metrics -->
                    <th title="Weight: 2% - Lower is better">Memory↓<br><small>(2%)</small></th>
                    <th title="Weight: 2% - Lower is better">CPU↓<br><small>(2%)</small></th>
                    <th title="Weight: 1%">Cache<br><small>(1%)</small></th>
                    
                    <!-- Performance -->
                    <th title="Weight: 7%">Success<br><small>(7%)</small></th>
                    <th title="Weight: 2% - Lower is better">Latency↓<br><small>(2%)</small></th>
                    <th title="Weight: 1% - Lower is better">Cost↓<br><small>(1%)</small></th>
                </tr>
            </thead>
            <tbody>
    """
    
    # Calculate comprehensive scores for each framework
    framework_scores = {}
    
    # First pass: Calculate all scores
    for framework in results:
        scores = {}
        
        # Get capability scores
        caps = capabilities.get(framework, {})
        for metric in ['multi_agent', 'tool_usage', 'error_handling', 'context_retention', 
                      'adaptability', 'scalability', 'observability', 'rag_capability']:
            scores[metric] = caps.get(metric, 50)  # Default to 50 if missing
        
        # Get enhanced metrics (averaged across use cases)
        if framework in enhanced_metrics:
            fw_metrics = enhanced_metrics[framework]
            
            # Initialize aggregates
            metric_sums = {
                'reasoning_depth': 0, 'planning_score': 0, 'decision_confidence': 0,
                'backtrack_rate': 0, 'grounding_score': 0, 'factual_consistency': 0,
                'tool_efficiency': 0, 'memory_mb': 0, 'cpu_percent': 0, 'cache_hit_rate': 0
            }
            count = 0
            
            for use_case, metrics in fw_metrics.items():
                count += 1
                if 'reasoning' in metrics:
                    metric_sums['reasoning_depth'] += metrics['reasoning'].get('reasoning_depth', 0)
                    metric_sums['planning_score'] += metrics['reasoning'].get('planning_score', 0)
                    metric_sums['decision_confidence'] += metrics['reasoning'].get('decision_confidence', 0)
                    metric_sums['backtrack_rate'] += metrics['reasoning'].get('backtrack_rate', 0)
                if 'hallucination' in metrics:
                    metric_sums['grounding_score'] += metrics['hallucination'].get('grounding_score', 0)
                    metric_sums['factual_consistency'] += metrics['hallucination'].get('factual_consistency', 0)
                if 'tool_efficiency' in metrics:
                    metric_sums['tool_efficiency'] += metrics['tool_efficiency'].get('efficiency_score', 0)
                if 'resource' in metrics:
                    metric_sums['memory_mb'] += metrics['resource'].get('memory_current_mb', 0)
                    metric_sums['cpu_percent'] += metrics['resource'].get('cpu_percent', 0)
                    metric_sums['cache_hit_rate'] += metrics['resource'].get('cache_hit_rate', 0)
            
            # Calculate averages
            if count > 0:
                scores['reasoning_depth'] = min(100, metric_sums['reasoning_depth'] / count * 10)  # Scale up
                scores['planning_score'] = metric_sums['planning_score'] / count
                scores['decision_confidence'] = metric_sums['decision_confidence'] / count
                scores['backtrack_rate'] = metric_sums['backtrack_rate'] / count
                scores['grounding_score'] = metric_sums['grounding_score'] / count
                scores['factual_consistency'] = metric_sums['factual_consistency'] / count
                scores['tool_efficiency'] = metric_sums['tool_efficiency'] / count
                # Convert resource metrics to efficiency scores
                # For memory: assume 500MB is bad (0%), 100MB is good (100%)
                avg_memory = metric_sums['memory_mb'] / count
                scores['memory_efficiency'] = max(0, min(100, 100 * (1 - (avg_memory - 100) / 400)))
                
                # For CPU: 100% is bad (0%), 0% is good (100%)
                scores['cpu_efficiency'] = max(0, 100 - metric_sums['cpu_percent'] / count)
                scores['cache_hit_rate'] = metric_sums['cache_hit_rate'] / count
        else:
            # Default enhanced metrics
            scores['reasoning_depth'] = 50
            scores['planning_score'] = 50
            scores['decision_confidence'] = 50
            scores['backtrack_rate'] = 20
            scores['grounding_score'] = 50
            scores['factual_consistency'] = 50
            scores['tool_efficiency'] = 50
            scores['memory_efficiency'] = 80
            scores['cpu_efficiency'] = 80
            scores['cache_hit_rate'] = 50
        
        # Get performance metrics
        framework_data = results[framework]
        successes = sum(1 for r in framework_data.values() if r.get('success'))
        total_tests = len(framework_data)
        scores['success_rate'] = (successes / total_tests * 100) if total_tests > 0 else 0
        
        # Latency (convert to efficiency score - lower is better)
        latencies = [r.get('latency', 0) for r in framework_data.values() if r.get('success')]
        avg_latency = sum(latencies) / len(latencies) if latencies else 1.0
        scores['latency_score'] = max(0, 100 - min(100, avg_latency * 20))  # Scale latency
        
        # Cost (convert to efficiency score - lower is better)
        total_cost = sum(r.get('cost', 0) for r in framework_data.values())
        scores['cost_efficiency'] = max(0, 100 - min(100, total_cost * 1000))  # Scale cost
        
        # Calculate OVERALL WEIGHTED SCORE
        overall_score = 0
        for metric, weight in all_metric_weights.items():
            metric_score = scores.get(metric, 50)
            
            # Handle inverse metrics (lower is better)
            if metric in ['backtrack_rate']:
                metric_score = 100 - metric_score
            
            overall_score += metric_score * weight
        
        scores['overall'] = overall_score
        framework_scores[framework] = scores
    
    # Sort frameworks by overall score (descending)
    sorted_frameworks = sorted(framework_scores.keys(), 
                              key=lambda f: framework_scores[f]['overall'], 
                              reverse=True)
    
    # Second pass: Generate HTML rows in sorted order
    for framework in sorted_frameworks:
        scores = framework_scores[framework]
        overall_score = scores['overall']
        
        # Generate table row
        row_color = '#f0fdf4' if overall_score >= 80 else '#fafafa' if overall_score >= 60 else '#fef2f2'
        
        html += f'''
                <tr style="background: {row_color};">
                    <td style="position: sticky; left: 0; background: {row_color}; font-weight: bold; border-right: 2px solid #e5e7eb;">
                        {framework.replace('_', ' ').title()}
                    </td>
        '''
        
        # Add all metric cells
        for metric in ['multi_agent', 'tool_usage', 'error_handling', 'context_retention',
                      'adaptability', 'scalability', 'observability', 'rag_capability',
                      'reasoning_depth', 'planning_score', 'decision_confidence', 'backtrack_rate',
                      'grounding_score', 'factual_consistency', 'tool_efficiency',
                      'memory_efficiency', 'cpu_efficiency', 'cache_hit_rate',
                      'success_rate', 'latency_score', 'cost_efficiency']:
            
            value = scores.get(metric, 0)
            
            # Color coding
            if metric in ['backtrack_rate']:  # Lower is better
                color = '#10b981' if value <= 20 else '#3b82f6' if value <= 40 else '#ef4444'
            elif metric in ['memory_efficiency', 'cpu_efficiency', 'latency_score', 'cost_efficiency']:
                # These are already inverted (higher = better efficiency)
                color = '#10b981' if value >= 80 else '#3b82f6' if value >= 60 else '#ef4444'
            else:  # Normal metrics (higher is better)
                color = '#10b981' if value >= 80 else '#3b82f6' if value >= 60 else '#ef4444'
            
            # Format value
            if metric in ['memory_efficiency']:
                display = f"{100-value:.0f}MB"
            elif metric in ['cpu_efficiency']:
                display = f"{100-value:.0f}%"
            elif metric == 'latency_score':
                display = f"{(100-value)/20:.1f}s"
            elif metric == 'cost_efficiency':
                display = f"${(100-value)/1000:.3f}"
            else:
                display = f"{value:.0f}"
            
            html += f'<td style="text-align: center; color: {color}; font-weight: 600;">{display}</td>'
        
        # Add overall score
        overall_color = '#10b981' if overall_score >= 80 else '#3b82f6' if overall_score >= 60 else '#ef4444'
        html += f'''
                    <td style="text-align: center; background: #fee2e2; font-size: 18px; font-weight: bold; color: {overall_color};">
                        {overall_score:.1f}
                    </td>
                </tr>
        '''
    
    html += '''
            </tbody>
        </table>
        </div>
    </div>
    '''
    
    return html, framework_scores
