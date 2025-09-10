# utils/evaluation.py

import json
from collections import Counter
from typing import Dict, Any, List, Optional

class UITreeEvaluator:
    """
    Acts as a "diagnostician" that scores quality and provides
    actionable feedback on the LLM's output.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            'structural_similarity': 0.4,
            'element_type_accuracy': 0.3,
            'spatial_accuracy': 0.1,
            'text_content_accuracy': 0.1,
            'completeness_score': 0.1
        }

    def evaluate(self, predicted: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, float]:
        if not predicted or not isinstance(predicted, dict):
            return self._zero_scores()
        
        scores = {
            'structural_similarity': self.structural_similarity(predicted, expected),
            'element_type_accuracy': self.element_type_accuracy(predicted, expected),
            'spatial_accuracy': self.spatial_accuracy(predicted, expected),
            'text_content_accuracy': self.text_content_accuracy(predicted, expected),
            'completeness_score': self.completeness_score(predicted, expected)
        }
        
        scores['composite_score'] = self._compute_composite_score(scores)
        return scores
        
    def generate_rich_feedback(self, predicted: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
        feedback = []
        
        if not predicted or not isinstance(predicted, dict):
            return ["CRITICAL FAILURE: The model did not produce a valid JSON object with a 'root' element."]

        pred_elements = self._extract_all_elements(predicted)
        exp_elements = self._extract_all_elements(expected)

        exp_signatures = {f"{elem.get('type')}|{elem.get('id')}" for elem in exp_elements}
        pred_signatures = {f"{elem.get('type')}|{elem.get('id')}" for elem in pred_elements}

        missing_elements = exp_signatures - pred_signatures
        extra_elements = pred_signatures - exp_signatures

        if missing_elements:
            feedback.append(f"Feedback: The model failed to identify these required elements: {', '.join(missing_elements)}.")
        if extra_elements:
            feedback.append(f"Feedback: The model hallucinated these extra elements: {', '.join(extra_elements)}.")

        exp_map = {elem.get('id'): elem for elem in exp_elements}
        pred_map = {elem.get('id'): elem for elem in pred_elements}
        common_ids = set(exp_map.keys()) & set(pred_map.keys())

        for element_id in common_ids:
            exp_el, pred_el = exp_map[element_id], pred_map[element_id]
            
            if exp_el.get('type') != pred_el.get('type'):
                feedback.append(f"Feedback: For element '{element_id}', the type was wrong. Expected '{exp_el.get('type')}', but got '{pred_el.get('type')}'.")
            
            if exp_el.get('text') != pred_el.get('text'):
                exp_text, pred_text = exp_el.get('text') or "null", pred_el.get('text') or "null"
                feedback.append(f"Feedback: For element '{element_id}', the text was wrong. Expected '{exp_text}', but got '{pred_text}'.")
        
        if not feedback:
            feedback.append("Feedback: The generated UI tree was highly accurate with no major errors found.")
            
        return feedback

    def _zero_scores(self) -> Dict[str, float]:
        zero_metrics = {key: 0.0 for key in self.weights.keys()}
        zero_metrics['composite_score'] = 0.0
        return zero_metrics

    def _compute_composite_score(self, scores: Dict[str, float]) -> float:
        return sum(self.weights[key] * scores.get(key, 0.0) for key in self.weights)

    def _extract_all_elements(self, tree: Dict[str, Any]) -> list:
        elements = []
        if isinstance(tree, dict):
            elements.append(tree)
            for child in tree.get('children', []):
                elements.extend(self._extract_all_elements(child))
        return elements

    def structural_similarity(self, predicted: Dict, expected: Dict) -> float:
        try:
            pred_elements, exp_elements = self._extract_all_elements(predicted), self._extract_all_elements(expected)
            if not exp_elements: return 1.0 if not pred_elements else 0.0
            return min(len(pred_elements), len(exp_elements)) / max(len(pred_elements), len(exp_elements))
        except Exception: return 0.0

    def element_type_accuracy(self, predicted: Dict, expected: Dict) -> float:
        try:
            pred_types = Counter(e.get('type', 'unknown') for e in self._extract_all_elements(predicted))
            exp_types = Counter(e.get('type', 'unknown') for e in self._extract_all_elements(expected))
            intersection = sum((pred_types & exp_types).values())
            union = sum((pred_types | exp_types).values())
            return intersection / union if union > 0 else 1.0
        except Exception: return 0.0

    def spatial_accuracy(self, predicted: Dict, expected: Dict) -> float:
        return 1.0

    def text_content_accuracy(self, predicted: Dict, expected: Dict) -> float:
        try:
            pred_texts = {e.get('text') for e in self._extract_all_elements(predicted) if e.get('text')}
            exp_texts = {e.get('text') for e in self._extract_all_elements(expected) if e.get('text')}
            if not exp_texts: return 1.0 if not pred_texts else 0.0
            intersection = len(pred_texts & exp_texts)
            union = len(pred_texts | exp_texts)
            return intersection / union if union > 0 else 1.0
        except Exception: return 0.0

    def completeness_score(self, predicted: Dict, expected: Dict) -> float:
        try:
            pred_sigs = {f"{e.get('type','')}|{e.get('text','')}" for e in self._extract_all_elements(predicted)}
            exp_sigs = {f"{e.get('type','')}|{e.get('text','')}" for e in self._extract_all_elements(expected)}
            if not exp_sigs: return 1.0
            return len(pred_sigs & exp_sigs) / len(exp_sigs)
        except Exception: return 0.0