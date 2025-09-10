# ui_tree_adaptor.py

import json
import sys
import os
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the local gepa folder to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gepa', 'src'))

try:
    from gepa.core.adapter import GEPAAdapter, EvaluationBatch  # type: ignore
    GEPA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: GEPA not available: {e}")
    GEPA_AVAILABLE = False
    # Create dummy classes for when GEPA is not available
    class GEPAAdapter:
        pass
    
    class EvaluationBatch:
        def __init__(self, outputs, scores, trajectories=None):
            self.outputs = outputs
            self.scores = scores
            self.trajectories = trajectories

from utils.evaluation import UITreeEvaluator

class UITreeAdapter(GEPAAdapter):
    """
    GEPA-compatible adapter for UI tree extraction optimization.
    Leverages the diagnostic evaluator to provide rich, reflective feedback.
    """
    def __init__(self, vision_model, metric_weights: Optional[Dict[str, float]] = None):
        if GEPA_AVAILABLE:
            super().__init__()
        self.vision_model = vision_model
        self.evaluator = UITreeEvaluator(weights=metric_weights or {})
        print("✅ UITreeAdapter initialized with Diagnostic UITreeEvaluator.")

    def evaluate(
        self,
        batch: List[Dict],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """
        Run the UI tree extraction program defined by `candidate` on a batch of data.
        
        Returns:
        - EvaluationBatch with outputs, scores, and optionally trajectories
        """
        outputs, scores, trajectories = [], [], []
        print(f"\nEvaluating candidate on {len(batch)} samples...")

        def _process_one(index: int, example: Dict) -> Dict[str, Any]:
            try:
                response_str = self.vision_model.generate(prompt=candidate, image=example['image'])
                try:
                    predicted_data = json.loads(response_str)
                    predicted_ui_tree = predicted_data.get("root", {})
                    parse_success = True
                except json.JSONDecodeError:
                    print(f"  - Sample {index+1}: ❌ JSON Parse Error. Assigning score 0.")
                    predicted_ui_tree = {"error": "Invalid JSON"}
                    parse_success = False
                if parse_success and "error" not in predicted_ui_tree:
                    expected_ui_tree = example['expected_ui_tree']
                    evaluation_results = self.evaluator.evaluate(
                        predicted=predicted_ui_tree,
                        expected=expected_ui_tree
                    )
                    score = evaluation_results.get('composite_score', 0.0)
                    print(f"  - Sample {index+1}: ✅ Scored {score:.3f}")
                else:
                    score = 0.0
                return {
                    "idx": index,
                    "predicted": predicted_ui_tree,
                    "score": score,
                    "response": response_str,
                }
            except Exception as e:
                print(f"  - Sample {index+1}: ❌ Evaluation Error: {e}. Assigning score 0.")
                return {
                    "idx": index,
                    "predicted": {"error": f"Evaluation failed: {str(e)}"},
                    "score": 0.0,
                    "response": "API call failed",
                }

        # Parallel execution across the minibatch with order preservation
        max_workers = min(4, len(batch)) if len(batch) > 0 else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(_process_one, i, item): i for i, item in enumerate(batch)}
            interim_results: Dict[int, Dict[str, Any]] = {}
            for future in as_completed(future_to_idx):
                res = future.result()
                interim_results[res["idx"]] = res

        for i in range(len(batch)):
            res = interim_results[i]
            predicted_ui_tree = res["predicted"]
            score = res["score"]
            response_str = res["response"]
            outputs.append(predicted_ui_tree)
            scores.append(score)
            if capture_traces:
                trajectories.append({
                    "prompt": candidate,
                    "response": response_str,
                    "predicted_tree": predicted_ui_tree,
                    "expected_tree": batch[i]['expected_ui_tree'],
                    "score": score,
                    "sample_index": i
                })

        avg_score = sum(scores) / len(scores) if scores else 0.0
        print(f"📊 Minibatch Average Score: {avg_score:.4f}")
        
        # Return EvaluationBatch as expected by GEPA
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None
        )

    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build a reflective dataset for instruction refinement by the teacher LLM.
        
        Returns:
        - Dict mapping component names to lists of reflection examples
        """
        reflection_examples = []
        
        if eval_batch.trajectories is None:
            print("⚠️ No trajectories available for reflection. Using basic feedback.")
            return {
                "system_prompt": [{"input": "UI screenshot", "prompt": candidate.get("system_prompt", ""), "output": "JSON output", "feedback": "No detailed feedback available"}],
                "ui_extraction_prompt": [{"input": "UI screenshot", "prompt": candidate.get("ui_extraction_prompt", ""), "output": "JSON output", "feedback": "No detailed feedback available"}]
            }

        def _schema_issues(predicted: Dict[str, Any]) -> List[str]:
            issues: List[str] = []
            required_keys = {"type", "id", "text", "style", "children"}
            if not isinstance(predicted, dict):
                return ["Predicted root is not an object"]
            def visit(node: Any, path: str):
                if not isinstance(node, dict):
                    issues.append(f"{path}: node is not an object")
                    return
                missing = required_keys - set(node.keys())
                for k in sorted(missing):
                    issues.append(f"{path}: missing key '{k}'")
                if 'text' in node and node.get('text', None) is not None and not isinstance(node['text'], str):
                    issues.append(f"{path}: 'text' must be string or null")
                if 'children' in node:
                    ch = node['children']
                    if not isinstance(ch, list):
                        issues.append(f"{path}: 'children' must be a list")
                    else:
                        for idx, child in enumerate(ch):
                            visit(child, f"{path}.children[{idx}]")
            visit(predicted, "root")
            return issues

        # Process each trajectory to create reflection examples
        for trajectory in eval_batch.trajectories:
            if trajectory['score'] < 0.95:  # Focus on examples that need improvement
                try:
                    rich_feedback_list = self.evaluator.generate_rich_feedback(
                        predicted=trajectory['predicted_tree'],
                        expected=trajectory['expected_tree']
                    )
                    feedback_comment = " ".join(rich_feedback_list)
                except Exception as e:
                    feedback_comment = f"Error generating feedback: {str(e)}"

                # Add component-specific notes for extraction instructions
                extraction_schema_notes = _schema_issues(trajectory['predicted_tree'])
                extraction_schema_summary = "; ".join(extraction_schema_notes[:10]) if extraction_schema_notes else "No schema issues detected"

                example = {
                    "Inputs": {
                        "screenshot": "A mobile UI screenshot that was processed",
                        "expected_output": json.dumps(trajectory['expected_tree'], indent=2)
                    },
                    "Generated Outputs": trajectory['response'],
                    "Feedback": {
                        "score": trajectory['score'],
                        "comment": feedback_comment,
                        "schema_findings_for_extraction": extraction_schema_summary,
                        "sample_index": trajectory.get('sample_index', 'unknown')
                    }
                }
                reflection_examples.append(example)

        print(f"🧠 Created a reflective dataset with {len(reflection_examples)} detailed examples for the optimizer.")

        # Return dataset for each component that needs updating
        return {
            component: reflection_examples for component in components_to_update
        }