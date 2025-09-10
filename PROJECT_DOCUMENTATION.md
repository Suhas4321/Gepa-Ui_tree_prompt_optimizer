# UI Tree Optimization Pipeline - Comprehensive Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [Data Flow](#data-flow)
4. [Process Flow](#process-flow)
5. [File Structure & Components](#file-structure--components)
6. [Core Functions & Implementation](#core-functions--implementation)
7. [Configuration Management](#configuration-management)
8. [Evaluation Metrics](#evaluation-metrics)
9. [GEPA Integration](#gepa-integration)
10. [Usage Guide](#usage-guide)
11. [Troubleshooting](#troubleshooting)

---

## Project Overview

### Purpose
This project optimizes AI prompts for converting mobile UI screenshots into structured JSON representations using the GEPA (Genetic Evolutionary Prompt Architecture) framework. The system learns to generate accurate UI hierarchy trees by iteratively improving both system-level instructions and extraction-specific prompts.

### Key Features
- **Dual-Prompt Optimization**: Simultaneously optimizes system behavior and extraction guidelines
- **Parallel Evaluation**: Concurrent processing for faster iteration cycles
- **Component-Specific Feedback**: Tailored reflection data for different prompt types
- **Schema-Aware Validation**: Ensures JSON output adheres to required structure
- **Pareto-Frontier Tracking**: Maintains diverse high-performing candidates

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    UI Tree Optimization Pipeline                │
├─────────────────────────────────────────────────────────────────┤
│  Input Layer                                                     │
│  ├── Screenshots (22 images)                                   │
│  └── Ground Truth JSON (22 files)                              │
├─────────────────────────────────────────────────────────────────┤
│  Data Processing Layer                                          │
│  ├── Dataset Loader (utils/dataset_loader.py)                  │
│  ├── Train/Val Split (70/30)                                   │
│  └── Data Validation                                           │
├─────────────────────────────────────────────────────────────────┤
│  Core Optimization Engine                                       │
│  ├── GEPA Framework (gepa/src/)                                │
│  ├── UITreeAdapter (ui_tree_adaptor.py)                        │
│  ├── Vision Model (StructuredVisionLLM)                        │
│  └── Evaluation System (utils/evaluation.py)                   │
├─────────────────────────────────────────────────────────────────┤
│  Prompt Management                                              │
│  ├── Seed Prompt Generation                                     │
│  ├── Coupled Updates (system + extraction)                     │
│  └── Schema Preservation                                       │
├─────────────────────────────────────────────────────────────────┤
│  Output Layer                                                   │
│  ├── Optimized Prompts (JSON)                                  │
│  ├── Performance Metrics                                       │
│  └── Training Logs                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### 1. Input Data Flow
```
Screenshots/ → Image Processing → Base64 Encoding → Vision Model Input
JSON Trees/ → Schema Validation → Ground Truth → Evaluation Reference
```

### 2. Training Data Flow
```
Raw Data → Dataset Loader → Train/Val Split → Minibatch Sampling → Evaluation
```

### 3. Optimization Data Flow
```
Current Prompts → Vision Model → Generated JSON → Evaluation → Score
Score → Reflection Dataset → Teacher LM → New Prompts → Acceptance Test
```

### 4. Output Data Flow
```
Best Candidate → JSON Serialization → Results Files → Performance Reports
```

---

## Process Flow

### Phase 1: Initialization
1. **Configuration Loading**: Load `config.yaml` with all parameters
2. **Dataset Preparation**: Load and split screenshots/JSON pairs
3. **Component Initialization**: Set up vision model, adapter, evaluator
4. **Seed Prompt Creation**: Generate fortified initial prompts

### Phase 2: Baseline Evaluation
1. **Full Validation Run**: Evaluate seed prompts on all 7 validation samples
2. **Score Recording**: Establish baseline performance metrics
3. **State Initialization**: Set up GEPA optimization state

### Phase 3: Iterative Optimization
```
For each iteration (until budget exhausted):
  1. Select Best Candidate
  2. Sample Training Minibatch (3 samples)
  3. Evaluate Current Prompts (parallel)
  4. Build Reflection Dataset
  5. Generate New Prompts (both components)
  6. Evaluate Proposed Prompts
  7. Acceptance Test (score improvement)
  8. Update Best/Pareto Tracking
```

### Phase 4: Results Generation
1. **Best Candidate Selection**: Choose highest-scoring prompts
2. **Results Serialization**: Save optimized prompts and metrics
3. **Logging**: Generate comprehensive training logs

---

## File Structure & Components

### Root Level Files

#### `optimizer.py` - Main Orchestrator
**Purpose**: Central coordination hub for the entire optimization pipeline
**Key Responsibilities**:
- Configuration management and validation
- Component initialization and wiring
- GEPA optimization execution
- Results processing and serialization

#### `config.yaml` - Configuration Management
**Purpose**: Centralized parameter store for all system components
**Sections**:
- `paths`: File and directory locations
- `dataset`: Train/validation split parameters
- `llm`: Vision model configuration
- `gepa_params`: Optimization engine settings
- `metric_weights`: Evaluation scoring weights
- `logging`: Logging configuration

#### `ui_tree_adaptor.py` - GEPA Integration Layer
**Purpose**: Bridge between GEPA framework and UI tree extraction system
**Key Features**:
- Parallel evaluation execution
- Component-specific reflection generation
- Schema-aware feedback construction
- Trajectory capture and management

### Utility Modules

#### `utils/dataset_loader.py` - Data Management
**Purpose**: Handles dataset loading, validation, and splitting
**Functions**:
- `prepare_dataset()`: Load and validate screenshot/JSON pairs
- `split_dataset()`: Create train/validation splits
- Data integrity checking and error handling

#### `utils/evaluation.py` - Performance Assessment
**Purpose**: Comprehensive evaluation system for generated UI trees
**Metrics**:
- Structural similarity
- Element type accuracy
- Spatial accuracy
- Text content accuracy
- Completeness score
- Composite scoring with configurable weights

### GEPA Framework (`gepa/src/`)

#### Core Components
- **`api.py`**: Main optimization interface
- **`core/`**: State management, adapters, results
- **`strategies/`**: Component selection, batch sampling
- **`proposer/`**: Prompt generation and mutation logic

#### Custom Extensions
- **`AllComponentsReflectionComponentSelector`**: Updates both prompts together
- **Enhanced reflection logic**: Component-specific feedback generation

---

## Core Functions & Implementation

### `optimizer.py` Functions

#### `load_config(config_path="config.yaml") -> Dict`
```python
def load_config(config_path="config.yaml") -> Dict:
    """Loads the YAML configuration file, creating a default if it doesn't exist."""
```
**Purpose**: Centralized configuration management with fallback defaults
**Implementation**:
- Checks for existing config file
- Creates default configuration if missing
- Validates and returns configuration dictionary
- Handles encoding and YAML parsing

#### `setup_logging(config: Dict)`
```python
def setup_logging(config: Dict):
    """Sets up structured logging to file and console."""
```
**Purpose**: Configure comprehensive logging system
**Features**:
- Dual output (file + console)
- Configurable log levels
- Structured formatting
- UTF-8 encoding support

#### `create_fortified_seed_prompt() -> Dict[str, str]`
```python
def create_fortified_seed_prompt() -> Dict[str, str]:
    """Creates a seed prompt with explicit guardrails to prevent schema violations."""
```
**Purpose**: Generate robust initial prompts with built-in protections
**Components**:
- `system_prompt`: High-level behavioral instructions
- `ui_extraction_prompt`: Schema-constrained extraction guidelines
- Guardrails to prevent reflection model from breaking schema

#### `StructuredVisionLLM` Class
```python
class StructuredVisionLLM:
    """A robust, configurable wrapper for the LiteLLM Vision API."""
```
**Purpose**: Reliable vision model interface with error handling
**Features**:
- Base64 image encoding
- JSON response format enforcement
- Comprehensive error handling with fallbacks
- Configurable model parameters
- Timeout and retry logic

#### `run_optimization()` Function
```python
def run_optimization(config: Dict, adapter: UITreeAdapter, 
                    seed_candidate: Dict, trainset: List, valset: List) -> Any:
```
**Purpose**: Execute GEPA optimization with full parameter exposure
**Parameters**:
- All GEPA configuration options
- Merge strategy settings
- Progress tracking options
- Budget management

### `ui_tree_adaptor.py` Functions

#### `UITreeAdapter` Class
```python
class UITreeAdapter(GEPAAdapter):
    """GEPA-compatible adapter for UI tree extraction optimization."""
```

##### `evaluate()` Method
```python
def evaluate(self, batch: List[Dict], candidate: Dict[str, str], 
            capture_traces: bool = False) -> EvaluationBatch:
```
**Purpose**: Parallel evaluation of prompt candidates on data batches
**Implementation**:
- **Parallel Processing**: Uses `ThreadPoolExecutor` for concurrent evaluation
- **Error Handling**: Graceful failure handling with fallback scores
- **Trajectory Capture**: Optional detailed execution traces
- **Score Aggregation**: Batch-level performance metrics

**Process Flow**:
1. Create parallel execution tasks for each sample
2. Execute vision model calls concurrently
3. Parse and validate JSON responses
4. Evaluate against ground truth
5. Aggregate results maintaining order

##### `make_reflective_dataset()` Method
```python
def make_reflective_dataset(self, candidate: Dict[str, str], 
                          eval_batch: EvaluationBatch, 
                          components_to_update: List[str]) -> Dict[str, List[Dict[str, Any]]]:
```
**Purpose**: Generate component-specific reflection data for prompt improvement
**Features**:
- **Schema Analysis**: Identifies JSON structure violations
- **Component-Specific Feedback**: Tailored guidance for different prompt types
- **Rich Context**: Includes input/output/feedback triplets
- **Quality Filtering**: Focuses on examples needing improvement

**Schema Validation Logic**:
```python
def _schema_issues(predicted: Dict[str, Any]) -> List[str]:
    """Identifies specific schema violations in generated JSON."""
```
- Validates required keys: `type`, `id`, `text`, `style`, `children`
- Checks data types and null handling
- Recursively validates nested structures
- Generates specific error messages for each violation

### `utils/dataset_loader.py` Functions

#### `prepare_dataset()` Function
```python
def prepare_dataset(screenshots_dir: str, json_dir: str) -> List[Dict]:
```
**Purpose**: Load and validate screenshot/JSON pairs
**Process**:
1. Scan directories for matching files
2. Validate image formats and JSON structure
3. Create paired data entries
4. Handle missing or corrupted files
5. Return validated dataset

#### `split_dataset()` Function
```python
def split_dataset(dataset: List[Dict], train_ratio: float, 
                 val_ratio: float, random_state: int) -> Tuple[List, List]:
```
**Purpose**: Create reproducible train/validation splits
**Features**:
- Configurable split ratios
- Random state for reproducibility
- Balanced distribution maintenance
- Validation of split parameters

### `utils/evaluation.py` Functions

#### `UITreeEvaluator` Class
```python
class UITreeEvaluator:
    """Comprehensive evaluator for UI tree extraction quality."""
```

##### Core Evaluation Metrics

**Structural Similarity**:
```python
def calculate_structural_similarity(self, predicted: Dict, expected: Dict) -> float:
```
- Compares tree structure and hierarchy
- Measures node relationships and nesting
- Accounts for missing or extra elements

**Element Type Accuracy**:
```python
def calculate_element_type_accuracy(self, predicted: Dict, expected: Dict) -> float:
```
- Validates correct component type assignments
- Handles type mapping and aliases
- Penalizes incorrect classifications

**Spatial Accuracy**:
```python
def calculate_spatial_accuracy(self, predicted: Dict, expected: Dict) -> float:
```
- Compares layout and positioning
- Validates style properties
- Measures spatial relationship accuracy

**Text Content Accuracy**:
```python
def calculate_text_content_accuracy(self, predicted: Dict, expected: Dict) -> float:
```
- Validates text extraction accuracy
- Handles null text cases
- Measures content fidelity

**Completeness Score**:
```python
def calculate_completeness_score(self, predicted: Dict, expected: Dict) -> float:
```
- Measures coverage of expected elements
- Penalizes missing components
- Rewards comprehensive extraction

##### Composite Scoring
```python
def evaluate(self, predicted: Dict, expected: Dict) -> Dict[str, float]:
```
**Purpose**: Generate weighted composite score from individual metrics
**Implementation**:
- Configurable metric weights
- Normalized scoring (0.0 to 1.0)
- Detailed breakdown for analysis
- Error handling for edge cases

---

## Configuration Management

### `config.yaml` Structure

#### Paths Configuration
```yaml
paths:
  screenshots_dir: "screenshots"    # Input image directory
  json_dir: "json_tree"            # Ground truth JSON directory
  output_dir: "results"            # Output files directory
```

#### Dataset Configuration
```yaml
dataset:
  train_ratio: 0.7                 # Training set proportion
  val_ratio: 0.3                   # Validation set proportion
  random_state: 42                 # Reproducibility seed
```

#### LLM Configuration
```yaml
llm:
  model_name: "gpt-4o"             # Vision model identifier
  max_tokens: 4096                 # Maximum response length
```

#### GEPA Parameters
```yaml
gepa_params:
  reflection_lm: "openai/gpt-4-turbo"  # Teacher model for prompt generation
  max_metric_calls: 75                 # Optimization budget
  reflection_minibatch_size: 3         # Samples per reflection
  candidate_selection_strategy: "pareto"  # Selection strategy
  use_merge: true                      # Enable merge operations
  max_merge_invocations: 3             # Merge budget
  track_best_outputs: false            # Output tracking
  display_progress_bar: false          # UI progress display
```

#### Metric Weights
```yaml
metric_weights:
  structural_similarity: 0.4       # Tree structure importance
  element_type_accuracy: 0.3       # Component type accuracy
  spatial_accuracy: 0.1            # Layout accuracy
  text_content_accuracy: 0.1       # Text extraction accuracy
  completeness_score: 0.1          # Coverage completeness
```

---

## Evaluation Metrics

### Metric Calculation Details

#### 1. Structural Similarity (Weight: 0.4)
**Purpose**: Measure how well the generated tree matches the expected hierarchy
**Calculation**:
- Compare node relationships and parent-child connections
- Measure tree depth and branching patterns
- Account for structural differences and missing branches
- Normalize by expected tree complexity

#### 2. Element Type Accuracy (Weight: 0.3)
**Purpose**: Validate correct component type identification
**Calculation**:
- Compare predicted vs expected component types
- Handle type aliases and synonyms
- Penalize incorrect classifications
- Reward exact type matches

#### 3. Spatial Accuracy (Weight: 0.1)
**Purpose**: Measure layout and positioning accuracy
**Calculation**:
- Compare style properties (position, size, alignment)
- Validate flex properties and layout directions
- Measure spatial relationship accuracy
- Account for visual positioning differences

#### 4. Text Content Accuracy (Weight: 0.1)
**Purpose**: Validate text extraction fidelity
**Calculation**:
- Compare extracted text with expected content
- Handle null text cases appropriately
- Measure character-level accuracy
- Account for text formatting differences

#### 5. Completeness Score (Weight: 0.1)
**Purpose**: Measure coverage of expected elements
**Calculation**:
- Count missing vs present elements
- Measure extraction comprehensiveness
- Penalize significant omissions
- Reward complete coverage

### Composite Score Calculation
```python
composite_score = (
    structural_similarity * 0.4 +
    element_type_accuracy * 0.3 +
    spatial_accuracy * 0.1 +
    text_content_accuracy * 0.1 +
    completeness_score * 0.1
)
```

---

## GEPA Integration

### GEPA Framework Overview
GEPA (Genetic Evolutionary Prompt Architecture) is a framework for optimizing text-based AI systems through evolutionary algorithms and reflective feedback.

### Custom Extensions

#### AllComponentsReflectionComponentSelector
```python
class AllComponentsReflectionComponentSelector(ReflectionComponentSelector):
    def select_modules(self, state, trajectories, subsample_scores, 
                      candidate_idx, candidate) -> list[str]:
        return list(state.list_of_named_predictors)
```
**Purpose**: Enable coupled updates of multiple prompt components
**Benefits**:
- Coordinated prompt improvements
- Joint optimization of related components
- Better convergence for interdependent prompts

#### Enhanced Reflection Logic
**Component-Specific Feedback**:
- System prompts receive high-level behavioral guidance
- Extraction prompts receive schema-specific feedback
- Tailored improvement suggestions for each component type

### GEPA Optimization Process

#### 1. State Management
- **Program Candidates**: Collection of prompt variations
- **Pareto Frontier**: Non-dominated high-performing solutions
- **Score Tracking**: Performance metrics for each candidate
- **Budget Management**: Metric call allocation and tracking

#### 2. Selection Strategies
- **Pareto Selection**: Maintains diverse high-performing candidates
- **Current Best**: Focuses on single best performer
- **Round Robin**: Cycles through different candidates

#### 3. Proposal Generation
- **Reflective Mutation**: Uses feedback to generate improvements
- **Merge Operations**: Combines strengths from multiple candidates
- **Component Updates**: Modifies specific prompt components

#### 4. Acceptance Criteria
- **Score Improvement**: New candidates must improve performance
- **Minibatch Validation**: Test on small sample before full evaluation
- **Pareto Updates**: Maintain frontier of non-dominated solutions

---

## Usage Guide

### Prerequisites
1. **Python Environment**: Python 3.8+ with virtual environment
2. **Dependencies**: Install requirements from `requirements.txt`
3. **API Keys**: OpenAI API key in `.env` file
4. **Data**: Screenshots and corresponding JSON files

### Setup Instructions

#### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

#### 2. Configuration
```bash
# Copy and edit configuration
cp config.yaml.example config.yaml
# Edit config.yaml with your parameters
```

#### 3. Data Preparation
```bash
# Organize your data
screenshots/
├── 1.jpg
├── 2.jpg
└── ...

json_tree/
├── 1.json
├── 2.json
└── ...
```

#### 4. API Key Setup
```bash
# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Running the Optimization

#### Basic Execution
```bash
python optimizer.py
```

#### Configuration Options
Edit `config.yaml` to customize:
- **Budget**: Increase `max_metric_calls` for longer runs
- **Model**: Change `model_name` for different vision models
- **Weights**: Adjust `metric_weights` for different priorities
- **Split**: Modify `train_ratio`/`val_ratio` for different data splits

### Output Files

#### Results Files
- `results/gepa_optimization_results.json`: Complete optimization results
- `results/optimized_prompts.json`: Best performing prompts
- `results/optimization_run.log`: Detailed training logs

#### Results Structure
```json
{
  "optimization_summary": {
    "best_score": 0.668,
    "optimization_successful": true,
    "improved_on_seed": true,
    "dataset_info": {"training": 15, "validation": 7}
  },
  "prompts": {
    "seed_prompt": {...},
    "optimized_prompt": {...}
  },
  "metadata": {
    "timestamp_utc": "2025-09-04T11:44:16.804353+00:00",
    "gepa_version": "unknown"
  }
}
```

---

## Troubleshooting

### Common Issues

#### 1. API Key Errors
**Problem**: `OPENAI_API_KEY not found`
**Solution**: 
- Verify `.env` file exists and contains valid API key
- Check API key permissions and billing status
- Ensure no extra spaces or quotes in key

#### 2. Dataset Loading Issues
**Problem**: Empty dataset or missing files
**Solution**:
- Verify screenshot and JSON directories exist
- Check file naming consistency (1.jpg → 1.json)
- Validate JSON file format and structure
- Ensure image files are readable

#### 3. Memory Issues
**Problem**: Out of memory during parallel evaluation
**Solution**:
- Reduce `reflection_minibatch_size` in config
- Lower `max_workers` in ThreadPoolExecutor
- Process smaller batches or use fewer parallel threads

#### 4. Poor Optimization Results
**Problem**: No improvement or score degradation
**Solution**:
- Increase `max_metric_calls` for longer optimization
- Adjust `metric_weights` to emphasize important aspects
- Check data quality and ground truth accuracy
- Verify evaluation metrics are appropriate

#### 5. JSON Parsing Errors
**Problem**: Generated JSON fails validation
**Solution**:
- Check schema constraints in `ui_extraction_prompt`
- Verify model response format settings
- Review error handling in `StructuredVisionLLM`
- Consider adjusting `max_tokens` for longer responses

### Performance Optimization

#### Speed Improvements
1. **Parallel Processing**: Already implemented in `UITreeAdapter`
2. **Batch Size**: Adjust `reflection_minibatch_size` for speed/quality tradeoff
3. **Model Selection**: Use faster models for evaluation, stronger for generation
4. **Caching**: Implement response caching for repeated evaluations

#### Quality Improvements
1. **Metric Weights**: Tune weights based on your priorities
2. **Budget**: Increase `max_metric_calls` for more iterations
3. **Data Quality**: Ensure high-quality ground truth annotations
4. **Schema Design**: Optimize JSON schema for your use case

### Debugging Tips

#### Enable Verbose Logging
```yaml
logging:
  log_level: "DEBUG"
```

#### Monitor Progress
- Watch `results/optimization_run.log` for real-time updates
- Check minibatch scores for iteration progress
- Monitor acceptance rates for optimization effectiveness

#### Analyze Results
- Compare seed vs optimized prompts
- Review metric breakdowns for improvement areas
- Examine failed examples for common patterns

---

## Advanced Configuration

### Custom Evaluation Metrics
To add custom metrics, extend `UITreeEvaluator`:

```python
def calculate_custom_metric(self, predicted: Dict, expected: Dict) -> float:
    # Your custom metric implementation
    return score

# Add to evaluate() method
custom_score = self.calculate_custom_metric(predicted, expected)
```

### Custom Prompt Components
To add new prompt components:

1. Update `create_fortified_seed_prompt()` to include new component
2. Modify `make_reflective_dataset()` to handle new component
3. Update component-specific feedback logic

### Integration with Other Models
To use different vision models:

1. Update `StructuredVisionLLM` configuration
2. Modify API call parameters in `generate()` method
3. Adjust response parsing for different formats

---

This documentation provides a comprehensive understanding of the UI Tree Optimization Pipeline, from high-level architecture to implementation details. Use it as a reference for development, debugging, and extending the system.
