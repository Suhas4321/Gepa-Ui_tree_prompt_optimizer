# Complete Workflow Guide: From Start to Finish
## A 5-Year-Old's Guide to Understanding the Technical Magic

---

## Table of Contents
1. [The Big Picture](#the-big-picture)
2. [Step-by-Step Journey](#step-by-step-journey)
3. [File-by-File Function Calls](#file-by-file-function-calls)
4. [GEPA Framework Deep Dive](#gepa-framework-deep-dive)
5. [Data Flow Visualization](#data-flow-visualization)
6. [Why Things Happen](#why-things-happen)

---

## The Big Picture

Imagine you have a magic box (your computer) that:
1. Takes pictures of phone screens
2. Has a smart assistant (AI) that looks at pictures and writes descriptions
3. Has a teacher (GEPA) that makes the assistant smarter by learning from mistakes
4. Keeps trying until the assistant gets really good at describing phone screens

---

## Step-by-Step Journey

### 🚀 **STEP 1: You Run `python optimizer.py`**

**What happens**: Your computer starts the magic show!

**File activated**: `optimizer.py`
**Function called**: `main()`

```python
def main():
    print("🚀 Starting UI Tree Optimization Pipeline...")
    # The show begins!
```

---

### 📋 **STEP 2: Loading Configuration**

**What happens**: The system reads the instruction manual

**File**: `optimizer.py`
**Function**: `load_config()`

```python
def load_config(config_path="config.yaml") -> Dict:
    # Reads config.yaml file
    # Creates default config if missing
    # Returns all settings (paths, models, etc.)
```

**Why**: The system needs to know where your pictures are, which AI model to use, how many tries to make, etc.

**Files touched**:
- `config.yaml` (read)
- Creates default config if missing

---

### 📝 **STEP 3: Setting Up Logging**

**What happens**: The system sets up a diary to remember everything

**File**: `optimizer.py`
**Function**: `setup_logging()`

```python
def setup_logging(config: Dict):
    # Creates log file: results/optimization_run.log
    # Sets up console output
    # Configures log levels (INFO, DEBUG, etc.)
```

**Why**: So you can see what's happening and debug if something goes wrong.

**Files created**:
- `results/optimization_run.log`

---

### 📁 **STEP 4: Loading Your Data**

**What happens**: The system finds and organizes your pictures and answers

**File**: `optimizer.py`
**Function**: `main()` calls `prepare_dataset()`

**File**: `utils/dataset_loader.py`
**Function**: `prepare_dataset()`

```python
def prepare_dataset(screenshots_dir: "screenshots", json_dir: "json_tree"):
    # Scans screenshots/ folder for images
    # Scans json_tree/ folder for JSON files
    # Pairs them up (1.jpg with 1.json, 2.jpg with 2.json, etc.)
    # Validates that files exist and are readable
    # Returns list of paired data
```

**Why**: The system needs to know what pictures to look at and what the correct answers should be.

**Files touched**:
- `screenshots/` folder (scanned)
- `json_tree/` folder (scanned)
- Creates paired dataset

---

### ✂️ **STEP 5: Splitting Data**

**What happens**: The system divides your data into practice and test sets

**File**: `utils/dataset_loader.py`
**Function**: `split_dataset()`

```python
def split_dataset(full_dataset, train_ratio=0.7, val_ratio=0.3, random_state=42):
    # Takes your 22 pairs
    # Puts 15 pairs in training set (for learning)
    # Puts 7 pairs in validation set (for testing)
    # Uses random_state=42 for reproducible splits
```

**Why**: 
- **Training set (15)**: For the AI to practice and learn from
- **Validation set (7)**: For testing how good the AI is getting

**Result**: 
- `trainset`: 15 pairs
- `valset`: 7 pairs

---

### 🤖 **STEP 6: Creating the AI Assistant**

**What happens**: The system creates a smart assistant that can look at pictures

**File**: `optimizer.py`
**Function**: `main()` creates `StructuredVisionLLM`

**File**: `optimizer.py`
**Class**: `StructuredVisionLLM`

```python
class StructuredVisionLLM:
    def __init__(self, config):
        # Sets up OpenAI API connection
        # Configures model (gpt-4o)
        # Sets up image processing
        # Configures response format (JSON)
```

**Why**: This is the AI that will look at your phone screenshots and try to describe them.

**Files touched**:
- `.env` (reads API key)
- Sets up LiteLLM connection

---

### 📊 **STEP 7: Creating the Judge**

**What happens**: The system creates a judge to score the AI's answers

**File**: `optimizer.py`
**Function**: `main()` creates `UITreeAdapter`

**File**: `ui_tree_adaptor.py`
**Class**: `UITreeAdapter`

```python
class UITreeAdapter(GEPAAdapter):
    def __init__(self, vision_model, metric_weights):
        # Takes the AI assistant
        # Creates a judge (UITreeEvaluator)
        # Sets up scoring weights
```

**File**: `utils/evaluation.py`
**Class**: `UITreeEvaluator`

```python
class UITreeEvaluator:
    def __init__(self, weights):
        # Sets up 5 different ways to score answers:
        # 1. Structural similarity (40%)
        # 2. Element type accuracy (30%)
        # 3. Spatial accuracy (10%)
        # 4. Text content accuracy (10%)
        # 5. Completeness score (10%)
```

**Why**: The judge needs to know how to score the AI's answers fairly.

---

### 🌱 **STEP 8: Creating the Starting Instructions**

**What happens**: The system writes the first set of instructions for the AI

**File**: `optimizer.py`
**Function**: `create_fortified_seed_prompt()`

```python
def create_fortified_seed_prompt() -> Dict[str, str]:
    system_prompt = "You are an AI that analyzes app screenshots..."
    ui_extraction_prompt = """Analyze the provided screenshot and generate a JSON object...
    ---SCHEMA (DO NOT CHANGE THIS BLOCK)---
    - The entire output must be a single, valid JSON object.
    - The JSON must have a single top-level key: "root".
    - Every UI element in the JSON tree MUST have these exact keys: "type", "id", "text", "style", "children".
    ---END SCHEMA---"""
    
    return {"system_prompt": system_prompt, "ui_extraction_prompt": ui_extraction_prompt}
```

**Why**: The AI needs starting instructions. These are like the first lesson plan.

**Result**: One prompt pair with two parts:
- `system_prompt`: General behavior instructions
- `ui_extraction_prompt`: Specific JSON generation rules

---

### 🎯 **STEP 9: First Test - Baseline Score**

**What happens**: The system tests the starting instructions on all 7 validation images


**File**: `optimizer.py`
**Function**: `run_optimization()` calls GEPA
**File**: `gepa/src/gepa/api.py`
**Function**: `optimize()`

**File**: `gepa/src/gepa/core/engine.py`
**Function**: `GEPAEngine.run()`

**File**: `gepa/src/gepa/core/state.py`
**Function**: `GEPAState.__init__()`

```python
def __init__(self, seed_candidate, base_valset_eval_output):
    # Takes the seed prompt
    # Evaluates it on all 7 validation images
    # Records the baseline score
    # Sets up optimization state
```

**File**: `ui_tree_adaptor.py`
**Function**: `UITreeAdapter.evaluate()`

```python
def evaluate(self, batch, candidate, capture_traces=False):
    # For each of the 7 images:
    #   1. Send image + prompt to AI
    #   2. Get JSON response
    #   3. Compare with ground truth
    #   4. Calculate score
    # Return all scores
```

**Why**: We need to know how good the starting instructions are before trying to improve them.

**Process**:
1. Same seed prompt applied to all 7 validation images
2. Each image generates different JSON (because images are different)
3. Each JSON compared to its ground truth
4. Each gets a different score
5. Average score becomes baseline (e.g., 0.46)

**Files touched**:
- All 7 images in validation set
- All 7 corresponding JSON files
- Vision model API calls

---

### 🔄 **STEP 10: The Learning Loop Begins**

**What happens**: The system starts trying to make the AI smarter

**File**: `gepa/src/gepa/core/engine.py`
**Function**: `GEPAEngine.run()`

```python
def run(self):
    while not self.budget_exhausted():
        # Select best candidate so far
        # Sample small batch for learning
        # Evaluate current prompts
        # Generate new prompts
        # Test new prompts
        # Accept if better, reject if worse
```

---

### 🎲 **STEP 11: Selecting What to Work On**

**What happens**: The system picks which version of instructions to improve

**File**: `gepa/src/gepa/strategies/candidate_selector.py`
**Function**: `ParetoCandidateSelector.select_candidate_idx()`

```python
def select_candidate_idx(self, state):
    # Looks at all candidates tried so far
    # Picks the best one using Pareto selection
    # Returns the index of chosen candidate
```

**Why**: We want to improve the best version we have so far.

---

### 📦 **STEP 12: Sampling Training Examples**

**What happens**: The system picks 3 random examples to learn from

**File**: `gepa/src/gepa/strategies/batch_sampler.py`
**Function**: `EpochShuffledBatchSampler.next_minibatch_indices()`

```python
def next_minibatch_indices(self, dataset_size, iteration):
    # Randomly picks 3 examples from training set
    # Returns indices [2, 7, 12] for example
```

**Why**: Learning from all 15 examples would be slow. 3 examples give enough feedback.

---

### 🔍 **STEP 13: Testing Current Instructions**

**What happens**: The system tests current instructions on the 3 selected examples

**File**: `gepa/src/gepa/proposer/reflective_mutation/reflective_mutation.py`
**Function**: `ReflectiveMutationProposer.propose()`

**File**: `ui_tree_adaptor.py`
**Function**: `UITreeAdapter.evaluate()` (parallel version)

```python
def evaluate(self, batch, candidate, capture_traces=True):
    # Creates parallel tasks for 3 examples
    with ThreadPoolExecutor(max_workers=4) as executor:
        # For each example:
        #   1. Send to vision model
        #   2. Get JSON response
        #   3. Parse and validate
        #   4. Compare with ground truth
        #   5. Calculate score
        #   6. Capture detailed traces
```

**Why**: We need to see how current instructions perform and what mistakes they make.

**Files touched**:
- 3 selected training images
- 3 corresponding JSON files
- Vision model API calls (parallel)

---

### 🧠 **STEP 14: Building Learning Material**

**What happens**: The system creates detailed feedback for the teacher

**File**: `ui_tree_adaptor.py`
**Function**: `UITreeAdapter.make_reflective_dataset()`

```python
def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
    # For each failed example:
    #   1. Analyze what went wrong
    #   2. Check for schema violations
    #   3. Generate specific feedback
    #   4. Create learning examples
    # Returns dataset for both system_prompt and ui_extraction_prompt
```

**File**: `ui_tree_adaptor.py`
**Function**: `_schema_issues()` (internal)

```python
def _schema_issues(predicted):
    # Checks JSON structure:
    # - Missing required keys (type, id, text, style, children)
    # - Wrong data types
    # - Invalid null handling
    # - Malformed nested structures
    # Returns list of specific problems
```

**Why**: The teacher needs specific examples of what went wrong to give better instructions.

---

### 👨‍🏫 **STEP 15: The Teacher Creates New Instructions**

**What happens**: A smarter AI (teacher) writes new instructions based on mistakes

**File**: `gepa/src/gepa/proposer/reflective_mutation/reflective_mutation.py`
**Function**: `ReflectiveMutationProposer.propose_new_texts()`

**File**: `gepa/src/gepa/strategies/instruction_proposal.py`
**Function**: `InstructionProposalSignature.run()`

```python
def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
    # For each component (system_prompt, ui_extraction_prompt):
    #   1. Take current instructions
    #   2. Take learning examples with feedback
    #   3. Send to teacher AI (gpt-4-turbo)
    #   4. Get improved instructions
    #   5. Return new prompt pair
```

**File**: `gepa/src/gepa/strategies/instruction_proposal.py`
**Function**: `InstructionProposalSignature.prompt_renderer()`

```python
def prompt_renderer(cls, input_dict):
    # Creates detailed prompt for teacher:
    # "Here are the current instructions: [current]
    #  Here are examples of mistakes: [examples]
    #  Please write better instructions that fix these problems."
```

**Why**: The teacher AI is better at writing instructions than the student AI.

**Files touched**:
- Teacher model API calls (gpt-4-turbo)
- Generates new system_prompt
- Generates new ui_extraction_prompt

---

### 🧪 **STEP 16: Testing New Instructions**

**What happens**: The system tests the new instructions on the same 3 examples

**File**: `ui_tree_adaptor.py`
**Function**: `UITreeAdapter.evaluate()` (again)

```python
def evaluate(self, batch, candidate, capture_traces=False):
    # Same process as Step 13, but with new instructions
    # Tests new system_prompt + ui_extraction_prompt
    # Calculates new average score
```

**Why**: We need to see if the new instructions are actually better.

---

### ⚖️ **STEP 17: The Decision - Accept or Reject**

**What happens**: The system decides whether to keep the new instructions

**File**: `gepa/src/gepa/proposer/reflective_mutation/reflective_mutation.py`
**Function**: `ReflectiveMutationProposer.propose()`

```python
# Compare scores:
old_score = 0.3817  # From Step 13
new_score = 0.6224  # From Step 16

if new_score > old_score:
    # ACCEPT: New instructions are better
    # Update best candidate
    # Continue with new instructions
else:
    # REJECT: New instructions are worse
    # Keep old instructions
    # Try again next iteration
```

**Why**: We only want to keep changes that actually make things better.

---

### 🏆 **STEP 18: Full Validation Test**

**What happens**: If accepted, test new instructions on all 7 validation images

**File**: `gepa/src/gepa/core/engine.py`
**Function**: `GEPAEngine.run()`

**File**: `ui_tree_adaptor.py`
**Function**: `UITreeAdapter.evaluate()` (full validation)

```python
# If new instructions accepted:
# Test on all 7 validation images
# Calculate full validation score
# Update best score tracking
# Update Pareto frontier
```

**Why**: We need to see how the new instructions perform on the full test set.

---

### 🔄 **STEP 19: Repeat Until Budget Exhausted**

**What happens**: The system keeps trying to improve until it runs out of tries

**File**: `gepa/src/gepa/core/engine.py`
**Function**: `GEPAEngine.run()`

```python
while self.total_num_evals < max_metric_calls:
    # Repeat Steps 11-18
    # Each iteration uses 6 metric calls (3 for current + 3 for new)
    # With budget of 75, expect ~12 iterations
```

**Why**: More tries = more chances to find better instructions.

---

### 📊 **STEP 20: Final Results**

**What happens**: The system saves the best instructions found

**File**: `optimizer.py`
**Function**: `process_and_save_results()`

```python
def process_and_save_results(config, gepa_result, seed_candidate, train_samples, val_samples):
    # Extract best candidate and score
    # Save optimized prompts to JSON
    # Save full results report
    # Log final performance
```

**Files created**:
- `results/optimized_prompts.json` (best instructions)
- `results/gepa_optimization_results.json` (full report)
- `results/optimization_run.log` (detailed logs)

---

## File-by-File Function Calls

### **optimizer.py** - The Main Director
```python
main()
├── load_config()                    # Read settings
├── setup_logging()                  # Set up diary
├── prepare_dataset()                # Load pictures and answers
├── split_dataset()                  # Divide into train/test
├── StructuredVisionLLM()            # Create AI assistant
├── UITreeAdapter()                  # Create judge
├── create_fortified_seed_prompt()   # Write starting instructions
├── run_optimization()               # Start learning process
└── process_and_save_results()       # Save final results
```

### **ui_tree_adaptor.py** - The Judge and Evaluator
```python
UITreeAdapter.__init__()
├── UITreeEvaluator()                # Create scoring system

UITreeAdapter.evaluate()
├── ThreadPoolExecutor()             # Parallel processing
├── vision_model.generate()          # Ask AI for JSON
├── json.loads()                     # Parse response
├── evaluator.evaluate()             # Score the answer
└── capture_traces()                 # Record details

UITreeAdapter.make_reflective_dataset()
├── _schema_issues()                 # Find JSON problems
├── evaluator.generate_rich_feedback() # Create detailed feedback
└── build_examples()                 # Create learning material
```

### **utils/dataset_loader.py** - The Data Manager
```python
prepare_dataset()
├── os.listdir()                     # Scan folders
├── PIL.Image.open()                 # Load images
├── json.load()                      # Load ground truth
└── validate_pairs()                 # Check data integrity

split_dataset()
├── random.Random()                  # Set random seed
├── random.shuffle()                 # Randomize order
└── split_indices()                  # Divide data
```

### **utils/evaluation.py** - The Scoring System
```python
UITreeEvaluator.evaluate()
├── calculate_structural_similarity()    # Check tree structure
├── calculate_element_type_accuracy()    # Check component types
├── calculate_spatial_accuracy()         # Check layout
├── calculate_text_content_accuracy()    # Check text
├── calculate_completeness_score()       # Check coverage
└── compute_composite_score()            # Combine all scores
```

---

## GEPA Framework Deep Dive

### **gepa/src/gepa/api.py** - The Main Entry Point
```python
optimize()
├── DefaultAdapter() or custom adapter    # Set up evaluation system
├── ParetoCandidateSelector()             # Choose what to improve
├── AllComponentsReflectionComponentSelector() # Choose which parts to update
├── EpochShuffledBatchSampler()           # Pick training examples
├── ReflectiveMutationProposer()          # Generate new instructions
├── GEPAEngine()                          # Run optimization loop
└── GEPAResult.from_state()               # Package results
```

### **gepa/src/gepa/core/engine.py** - The Optimization Engine
```python
GEPAEngine.run()
├── initialize_gepa_state()               # Set up initial state
├── reflective_proposer.propose()         # Generate new candidate
├── evaluator()                           # Test new candidate
├── update_pareto_frontier()              # Track best solutions
└── check_budget()                        # Stop when done
```

### **gepa/src/gepa/core/state.py** - The Memory System
```python
GEPAState.__init__()
├── program_candidates = [seed_candidate] # Store all versions tried
├── program_full_scores_val_set = [baseline_score] # Track validation scores
├── list_of_named_predictors = ["system_prompt", "ui_extraction_prompt"] # Components to optimize
└── pareto_front_valset = [baseline_scores] # Track best performance
```

### **gepa/src/gepa/strategies/component_selector.py** - The Component Chooser
```python
AllComponentsReflectionComponentSelector.select_modules()
└── return ["system_prompt", "ui_extraction_prompt"] # Update both together
```

### **gepa/src/gepa/proposer/reflective_mutation/reflective_mutation.py** - The Improvement Generator
```python
ReflectiveMutationProposer.propose()
├── candidate_selector.select_candidate_idx() # Pick best candidate
├── batch_sampler.next_minibatch_indices()    # Pick training examples
├── adapter.evaluate()                         # Test current instructions
├── module_selector.select_modules()          # Pick components to update
├── adapter.make_reflective_dataset()         # Create learning material
├── propose_new_texts()                       # Generate new instructions
├── adapter.evaluate()                        # Test new instructions
└── accept_or_reject()                        # Decide whether to keep
```

---

## Data Flow Visualization

```
📁 Your Data
├── screenshots/ (22 images)
└── json_tree/ (22 JSON files)
    ↓
📊 Dataset Loader
├── Pair images with JSON
├── Split: 15 train + 7 validation
└── Validate data integrity
    ↓
🌱 Seed Prompt Creation
├── system_prompt (behavior instructions)
└── ui_extraction_prompt (JSON rules)
    ↓
🎯 Baseline Evaluation
├── Apply seed to all 7 validation images
├── Generate 7 different JSON responses
├── Compare each with ground truth
└── Calculate average baseline score
    ↓
🔄 Optimization Loop (Repeat until budget exhausted)
├── 🎲 Select best candidate so far
├── 📦 Sample 3 training examples
├── 🔍 Test current instructions (parallel)
├── 🧠 Build learning material with feedback
├── 👨‍🏫 Teacher generates new instructions
├── 🧪 Test new instructions
├── ⚖️ Accept if better, reject if worse
└── 🏆 Full validation test if accepted
    ↓
📊 Final Results
├── Best instructions saved
├── Performance metrics recorded
└── Detailed logs generated
```

---

## Why Things Happen

### **Why Only One Seed Prompt?**
- **What**: You create one set of instructions
- **Why**: The AI needs starting instructions to learn from
- **How**: Same instructions applied to different images produce different results

### **Why Different Scores for Same Prompt?**
- **What**: Same prompt + different images = different scores
- **Why**: Each image has different complexity, clarity, and challenges
- **How**: Simple images get high scores, complex images get lower scores

### **Why Parallel Processing?**
- **What**: Test multiple images simultaneously
- **Why**: Much faster than testing one by one
- **How**: ThreadPoolExecutor runs multiple API calls at once

### **Why Component-Specific Feedback?**
- **What**: Different feedback for system_prompt vs ui_extraction_prompt
- **Why**: Each component has different responsibilities
- **How**: Schema violations help ui_extraction_prompt, behavioral issues help system_prompt

### **Why Coupled Updates?**
- **What**: Update both prompts together
- **Why**: They work together, so improving them together is more effective
- **How**: AllComponentsReflectionComponentSelector returns both component names

### **Why Acceptance Testing?**
- **What**: Only keep changes that improve performance
- **Why**: Prevents the system from getting worse
- **How**: Compare minibatch scores before and after

### **Why Pareto Frontier?**
- **What**: Track multiple good solutions, not just the best
- **Why**: Different solutions might be good for different types of images
- **How**: Maintain set of non-dominated candidates

### **Why Budget Management?**
- **What**: Limit total number of evaluations
- **Why**: API calls cost money and time
- **How**: Track metric_calls and stop when budget exhausted

---

This is the complete journey from when you type `python optimizer.py` until you get your optimized prompts. Every function call, every file touched, every decision made - all explained like you're 5 years old but with technical accuracy!
