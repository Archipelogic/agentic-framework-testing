"""
Resource tracking for framework evaluation.
Monitors memory, CPU, and cache efficiency.
"""

import psutil
import time
from typing import Dict, Any, Optional
from collections import defaultdict


class ResourceTracker:
    """Track resource usage during framework execution."""
    
    def __init__(self):
        """Initialize resource tracker."""
        self.process = psutil.Process()
        self.cache = defaultdict(lambda: {'hits': 0, 'misses': 0, 'first_seen': time.time()})
        self.start_memory = None
        self.peak_memory = 0
        
    def start_tracking(self):
        """Start resource tracking."""
        self.start_memory = self.get_memory_mb()
        self.peak_memory = self.start_memory
        
    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def track_cache_access(self, key: str) -> bool:
        """Track cache hit/miss for a key."""
        entry = self.cache[key]
        
        if entry['hits'] > 0 or entry['misses'] > 0:
            # Already seen this key
            entry['hits'] += 1
            return True
        else:
            # First time seeing this key
            entry['misses'] += 1
            return False
    
    def update_peak_memory(self):
        """Update peak memory usage."""
        current = self.get_memory_mb()
        if current > self.peak_memory:
            self.peak_memory = current
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive resource metrics."""
        total_hits = sum(v['hits'] for v in self.cache.values())
        total_accesses = sum(v['hits'] + v['misses'] for v in self.cache.values())
        
        current_memory = self.get_memory_mb()
        memory_delta = current_memory - self.start_memory if self.start_memory else 0
        
        return {
            'memory_current_mb': round(current_memory, 2),
            'memory_peak_mb': round(self.peak_memory, 2),
            'memory_delta_mb': round(memory_delta, 2),
            'cpu_percent': self.process.cpu_percent(interval=0.1),
            'cache_hit_rate': round(total_hits / total_accesses * 100, 1) if total_accesses > 0 else 0,
            'cache_total_keys': len(self.cache),
            'cache_total_hits': total_hits
        }
