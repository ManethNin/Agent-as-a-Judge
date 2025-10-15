"""
Performance Tracker utility for monitoring evaluation metrics.

This module provides functionality to track and analyze the performance
of Agent-as-a-Judge evaluations including timing, costs, and accuracy metrics.
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class EvaluationMetrics:
    """Data class for storing evaluation performance metrics."""
    timestamp: float
    evaluation_time: float
    llm_cost: float
    input_tokens: int
    output_tokens: int
    criteria_satisfied: bool
    confidence_score: float = 0.0
    agent_name: str = "unknown"


class PerformanceTracker:
    """Track and analyze Agent-as-a-Judge performance metrics."""
    
    def __init__(self, output_file: Optional[Path] = None):
        self.metrics: List[EvaluationMetrics] = []
        self.output_file = output_file
        self.start_time: Optional[float] = None
        
    def start_evaluation(self) -> None:
        """Start timing an evaluation."""
        self.start_time = time.time()
        
    def end_evaluation(
        self, 
        llm_cost: float,
        input_tokens: int,
        output_tokens: int,
        criteria_satisfied: bool,
        confidence_score: float = 0.0,
        agent_name: str = "unknown"
    ) -> EvaluationMetrics:
        """End timing and record evaluation metrics."""
        if self.start_time is None:
            raise ValueError("Must call start_evaluation() first")
            
        evaluation_time = time.time() - self.start_time
        
        metrics = EvaluationMetrics(
            timestamp=time.time(),
            evaluation_time=evaluation_time,
            llm_cost=llm_cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            criteria_satisfied=criteria_satisfied,
            confidence_score=confidence_score,
            agent_name=agent_name
        )
        
        self.metrics.append(metrics)
        self.start_time = None
        
        logging.info(f"Evaluation completed in {evaluation_time:.2f}s, cost: ${llm_cost:.4f}")
        return metrics
        
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all recorded evaluations."""
        if not self.metrics:
            return {}
            
        total_evaluations = len(self.metrics)
        total_time = sum(m.evaluation_time for m in self.metrics)
        total_cost = sum(m.llm_cost for m in self.metrics)
        total_tokens = sum(m.input_tokens + m.output_tokens for m in self.metrics)
        satisfaction_rate = sum(1 for m in self.metrics if m.criteria_satisfied) / total_evaluations
        
        return {
            "total_evaluations": total_evaluations,
            "total_time_seconds": total_time,
            "average_time_per_eval": total_time / total_evaluations,
            "total_cost_usd": total_cost,
            "average_cost_per_eval": total_cost / total_evaluations,
            "total_tokens": total_tokens,
            "satisfaction_rate": satisfaction_rate,
            "average_confidence": sum(m.confidence_score for m in self.metrics) / total_evaluations
        }
        
    def save_metrics(self, filepath: Optional[Path] = None) -> None:
        """Save metrics to JSON file."""
        output_path = filepath or self.output_file
        if not output_path:
            raise ValueError("No output file specified")
            
        data = {
            "summary": self.get_summary_stats(),
            "detailed_metrics": [asdict(m) for m in self.metrics]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logging.info(f"Saved performance metrics to {output_path}")
        
    def load_metrics(self, filepath: Path) -> None:
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.metrics = [
            EvaluationMetrics(**m) for m in data.get("detailed_metrics", [])
        ]
        
        logging.info(f"Loaded {len(self.metrics)} metrics from {filepath}")