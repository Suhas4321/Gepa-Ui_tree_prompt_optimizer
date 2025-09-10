# optimizer.py

import sys
import os
# Add the local gepa folder to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gepa', 'src'))

try:
    import gepa
    GEPA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: GEPA not available: {e}")
    GEPA_AVAILABLE = False
    gepa = None

import json
import base64
import logging
import yaml
from pathlib import Path
from io import BytesIO
from typing import List, Any, Dict, Optional, Tuple, cast
from pydantic import BaseModel, Field
from ui_tree_adaptor import UITreeAdapter
from utils.dataset_loader import prepare_dataset, split_dataset
from dotenv import load_dotenv
from datetime import datetime, timezone
import litellm

# --- Configuration and Logging Setup ---

def load_config(config_path="config.yaml") -> Dict:
    """Loads the YAML configuration file, creating a default if it doesn't exist."""
    if not Path(config_path).exists():
        logging.warning(f"Config file not found at {config_path}. Creating a default one.")
        default_config = {
            'paths': {'screenshots_dir': 'screenshots', 'json_dir': 'json_tree', 'output_dir': 'results'},
            'dataset': {'train_ratio': 0.7, 'val_ratio': 0.3, 'random_state': 42},
            'llm': {'model_name': 'gpt-4o', 'max_tokens': 4096},
            'gepa_params': {
                'reflection_lm': 'openai/gpt-4-turbo',
                'max_metric_calls': 75,
                'reflection_minibatch_size': 3,
                'candidate_selection_strategy': 'pareto',
                'use_merge': True,
                'max_merge_invocations': 3,
                'track_best_outputs': False,
                'display_progress_bar': False
            },
            'metric_weights': {
                'structural_similarity': 0.4,
                'element_type_accuracy': 0.3,
                'spatial_accuracy': 0.1,
                'text_content_accuracy': 0.1,
                'completeness_score': 0.1
            },
            'logging': {'log_level': 'INFO', 'log_file': 'optimization.log'}
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, indent=2)
        return default_config
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_logging(config: Dict):
    """Sets up structured logging to file and console."""
    log_config = config['logging']
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / log_config['log_file']
    
    logging.basicConfig(
        level=log_config.get('log_level', 'INFO').upper(),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )
    logging.info("Logging configured successfully.")

# --- Pydantic Schemas (Single Source of Truth) ---

class UIElement(BaseModel):
    type: str = Field(..., description="Component type.")
    id: str = Field(..., description="A descriptive, unique snake_case identifier.")
    text: Optional[str] = Field(None, description="Visible text content.")
    style: Dict[str, Any] = Field(default_factory=dict, description="Key-value style properties.")
    children: List['UIElement'] = Field(default_factory=list, description="Nested child elements.")

UIElement.model_rebuild()

class UITree(BaseModel):
    root: UIElement

# --- Core Components ---

def create_fortified_seed_prompt() -> Dict[str, str]:
    """
    Creates a seed prompt with explicit "guardrails" to prevent the reflection
    model from changing the core schema requirements.
    """
    system_prompt = "You are an AI that analyzes app screenshots and converts them into a structured JSON format. Your goal is to accurately describe the UI hierarchy. Improve your analysis based on the feedback provided about past mistakes. IMPORTANT: When proposing new instructions, do not include example JSON in your response that violates the required schema."
    
    ui_extraction_prompt = """Analyze the provided screenshot and generate a JSON object representing its UI hierarchy.

---SCHEMA (DO NOT CHANGE THIS BLOCK)---
- The entire output must be a single, valid JSON object.
- The JSON must have a single top-level key: "root".
- Every UI element in the JSON tree MUST have these exact keys: "type", "id", "text", "style", "children".
- The "id" for each element must be a descriptive, unique identifier in snake_case.
- If an element has no visible text, its "text" value must be `null`.
- Elements with no children must have an empty list `[]` for the "children" value.
---END SCHEMA---

Respond with nothing but the JSON object.
"""
    
    return {"system_prompt": system_prompt, "ui_extraction_prompt": ui_extraction_prompt}

class StructuredVisionLLM:
    """A robust, configurable wrapper for the LiteLLM Vision API."""
    def __init__(self, config: Dict):
        load_dotenv()
        litellm.api_key = os.getenv("OPENAI_API_KEY")
        if not litellm.api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file or environment.")
        self.config = config['llm']
        
        # Configure litellm with robust settings
        # Note: These settings may vary by litellm version
        # Removed litellm.set_verbose as it is not a valid attribute
        
        try:
            litellm.drop_params = True
        except AttributeError:
            pass  # Ignore if not available in this version
        
        logging.info(f"Initialized VisionLLM with model: {self.config['model_name']}")

    def generate(self, prompt: Dict[str, str], image: Any) -> str:
        full_prompt = f"{prompt['system_prompt']}\n\n{prompt['ui_extraction_prompt']}"
        
        buffered = BytesIO()
        if hasattr(image, 'mode') and image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffered, format="PNG")
        img_base_64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        try:
            response = litellm.completion(
                model=self.config['model_name'],
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base_64}"}}
                    ]
                }],
                max_tokens=self.config['max_tokens'],
                response_format={"type": "json_object"},
                timeout=120.0,
                temperature=0.1  # Lower temperature for more consistent outputs
            )
            
            # Handle different response formats safely
            resp_any = cast(Any, response)
            if hasattr(resp_any, 'choices') and getattr(resp_any, 'choices'):
                first_choice = getattr(resp_any, 'choices')[0]
                if hasattr(first_choice, 'message'):
                    content = cast(Any, first_choice).message.content
                else:
                    content = str(first_choice)
            else:
                content = str(resp_any)
                
            if content is None:
                logging.warning("LLM returned a None response. Using fallback.")
                return json.dumps({"root": {"type": "Screen", "id": "error_screen_null", "text": None, "style": {}, "children": []}})
            return content
        except Exception as e:
            logging.error(f"LLM API call failed: {e}. Using fallback.", exc_info=True)
            return json.dumps({"root": {"type": "Screen", "id": "error_screen_exception", "text": None, "style": {}, "children": []}})

# --- Main Workflow Functions ---

def run_optimization(config: Dict, adapter: UITreeAdapter, seed_candidate: Dict, trainset: List, valset: List) -> Any:
    gepa_params = config['gepa_params']
    logging.info(f"Starting GEPA optimization with budget of {gepa_params['max_metric_calls']} calls...")
    try:
        # Ensure GEPA is available at runtime and for type checking
        if not GEPA_AVAILABLE:
            raise RuntimeError("GEPA library is not available. Install or ensure 'gepa/src' is on sys.path.")
        gepa_result = cast(Any, gepa).optimize(
            seed_candidate=seed_candidate,
            adapter=adapter,
            trainset=trainset,
            valset=valset,
            reflection_lm=gepa_params['reflection_lm'],
            max_metric_calls=gepa_params['max_metric_calls'],
            reflection_minibatch_size=gepa_params.get('reflection_minibatch_size', 3),
            candidate_selection_strategy=gepa_params.get('candidate_selection_strategy', 'pareto'),
            use_merge=gepa_params.get('use_merge', False),
            max_merge_invocations=gepa_params.get('max_merge_invocations', 3),
            track_best_outputs=gepa_params.get('track_best_outputs', False),
            display_progress_bar=gepa_params.get('display_progress_bar', False)
        )
        logging.info("🎉 GEPA optimization process completed.")
        return gepa_result
    except Exception as e:
        logging.critical(f"❌ GEPA optimization failed with a critical error: {e}", exc_info=True)
        return None

def process_and_save_results(config: Dict, gepa_result: Any, seed_candidate: Dict, train_samples: int, val_samples: int):
    output_dir = Path(config['paths']['output_dir'])
    
    if gepa_result is None:
        logging.warning("GEPA result is None. Saving fallback results.")
        best_score, best_candidate = 0.0, seed_candidate
        success = False 
    else:
        best_score, best_candidate = extract_gepa_results_safely(gepa_result)
        success = True
        if best_candidate is None:
            logging.warning("Could not extract an optimized candidate, using seed prompt as fallback.")
            best_candidate = seed_candidate
    
    results = {
        "optimization_summary": {
            "best_score": best_score,
            "optimization_successful": success,
            "improved_on_seed": best_candidate != seed_candidate and best_score > 0,
            "dataset_info": {"training": train_samples, "validation": val_samples},
        },
        "prompts": {"seed_prompt": seed_candidate, "optimized_prompt": best_candidate},
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), 
            "gepa_version": getattr(gepa, '__version__', 'unknown')
        }
    }
    
    results_path = output_dir / "gepa_optimization_results.json"
    prompts_path = output_dir / "optimized_prompts.json"
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    with open(prompts_path, 'w', encoding='utf-8') as f:
        json.dump(best_candidate, f, indent=2)
        
    logging.info(f"🏆 Best validation score: {best_score:.4f}")
    logging.info(f"💾 Optimized prompt saved to {prompts_path}")
    logging.info(f"📊 Full report saved to {results_path}")

def extract_gepa_results_safely(gepa_result: Any) -> Tuple[float, Optional[Dict]]:
    """
    Safely extract the best score and candidate from GEPA result.
    Handles the correct GEPA result structure with val_aggregate_scores.
    """
    logging.info("🔍 Analyzing GEPA result structure...")
    
    # Check if this is a proper GEPA result object
    if hasattr(gepa_result, 'val_aggregate_scores') and gepa_result.val_aggregate_scores:
        best_score = max(gepa_result.val_aggregate_scores)
        best_idx = gepa_result.best_idx
        best_candidate = gepa_result.candidates[best_idx]
        logging.info(f"✅ Found best score via GEPA result: {best_score:.4f}")
        return best_score, best_candidate
    
    # Fallback: try to find score in various attributes
    best_score = 0.0
    score_attrs = ['best_valset_aggregate_score', 'best_score', 'best_candidate_score', 'final_score']
    for attr in score_attrs:
        if hasattr(gepa_result, attr):
            value = getattr(gepa_result, attr)
            if isinstance(value, (int, float)) and value > 0:
                best_score = float(value)
                logging.info(f"✅ Found score via '{attr}': {best_score:.4f}")
                break
    
    best_candidate = None
    candidate_attrs = ['best_candidate', 'candidate']
    for attr in candidate_attrs:
        if hasattr(gepa_result, attr):
            candidate_value = getattr(gepa_result, attr)
            if isinstance(candidate_value, dict):
                best_candidate = candidate_value
                logging.info(f"✅ Found candidate via '{attr}'")
                break
    
    if best_score == 0.0:
        logging.warning("Could not extract a valid score from the final result object. Check logs for scores from previous iterations.")

    return best_score, best_candidate

def main():
    """Main execution entry point for the optimization workflow."""
    print("🚀 Starting UI Tree Optimization Pipeline...")
    try:
        config = load_config()
        setup_logging(config)
        logging.info("="*60)
        logging.info("🚀 GEPA UI TREE OPTIMIZATION STARTED")
        logging.info("="*60)
    except Exception as e:
        logging.basicConfig(level='ERROR')
        logging.critical(f"FATAL: Error during setup: {e}", exc_info=True)
        return
    
    logging.info("📁 Loading and splitting dataset...")
    paths_config, dataset_config = config['paths'], config['dataset']
    full_dataset = prepare_dataset(paths_config['screenshots_dir'], paths_config['json_dir'])
    if not full_dataset:
        logging.critical("❌ Dataset is empty. Exiting.")
        return
    
    trainset, valset = split_dataset(full_dataset, **dataset_config)
    logging.info(f"📊 Dataset split: {len(trainset)} training, {len(valset)} validation")
    
    logging.info("🔧 Initializing components...")
    vision_model = StructuredVisionLLM(config)
    
    # Initialize adapter with metric weights from config
    metric_weights = config.get('metric_weights', {})
    adapter = UITreeAdapter(vision_model, metric_weights=metric_weights)
    
    seed_candidate = create_fortified_seed_prompt()
    logging.info("🌱 Using a fortified seed prompt with guardrails to maximize learning.")

    gepa_result = run_optimization(config, adapter, seed_candidate, trainset, valset)
    
    process_and_save_results(config, gepa_result, seed_candidate, len(trainset), len(valset))
    
    logging.info("="*60)
    logging.info("🎉 OPTIMIZATION RUN FINISHED")
    logging.info("="*60)

if __name__ == "__main__":
    main()