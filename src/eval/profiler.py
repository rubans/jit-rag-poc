import time
from typing import List, Dict, Any
import statistics


class LatencyProfiler:
    """
    Instruments pipeline stages and computes statistical distributions
    (P50, P90, P99, Mean) for latency in milliseconds.
    """
    def __init__(self):
        self.records: Dict[str, List[float]] = {
            "ingestion": [],
            "embedding": [],
            "retrieval": [],
            "generation": [],
            "end_to_end": []
        }

    def record(self, stage: str, latency_ms: float):
        if stage not in self.records:
            self.records[stage] = []
        self.records[stage].append(latency_ms)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for stage, times in self.records.items():
            if not times:
                summary[stage] = {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
                continue
            sorted_times = sorted(times)
            count = len(sorted_times)
            
            def percentile(p: float) -> float:
                if count == 1:
                    return sorted_times[0]
                k = (count - 1) * p
                f = int(k)
                c = min(f + 1, count - 1)
                d = k - f
                return sorted_times[f] + d * (sorted_times[c] - sorted_times[f])

            summary[stage] = {
                "count": count,
                "mean": round(statistics.mean(sorted_times), 2),
                "p50": round(percentile(0.50), 2),
                "p90": round(percentile(0.90), 2),
                "p99": round(percentile(0.99), 2),
                "min": round(sorted_times[0], 2),
                "max": round(sorted_times[-1], 2)
            }
        return summary

