#!/usr/bin/env python3
"""
Script to check which stages didn't save a best model checkpoint for each LLM.

This script analyzes the results directory structure created by curriculum_learning.py
and reports which stages are missing checkpoints for each tested LLM.

Expected LLMs tested:
- meta-llama/Llama-3.2-1B -> Llama_3_2_1B
- meta-llama/Llama-3.2-3B -> Llama_3_2_3B  
- google/gemma-2-2b -> gemma_2_2b
- google/gemma-2-9b -> gemma_2_9b
- google/gemma-2-27b -> gemma_2_27b
- google/gemma-3-270m -> gemma_3_270m
- google/gemma-3-1b-pt -> gemma_3_1b_pt

Expected stages:
- stage1_mcq
- stage2_captioning
- stage3_cot
- stage4_sleep_cot
- stage5_ecg_cot
"""

import os
import json
from typing import Dict, List, Set, Tuple
from pathlib import Path


def sanitize_llm_id(llm_id: str) -> str:
    """Sanitize llm_id for use in directory names (same logic as curriculum_learning.py)"""
    if not llm_id:
        return "unknown_llm"
    # Take last part after /, replace . and - with _
    name = llm_id.split("/")[-1]
    name = name.replace(".", "_").replace("-", "_")
    # Remove duplicate underscores
    while "__" in name:
        name = name.replace("__", "_")
    return name


def get_expected_llm_directories() -> Dict[str, str]:
    """Get expected LLM directory names and their original IDs"""
    expected_llms = [
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B", 
        "google/gemma-2-2b",
        "google/gemma-2-9b",
        "google/gemma-2-27b",
        "google/gemma-3-270m",
        "google/gemma-3-1b-pt"
    ]
    
    return {sanitize_llm_id(llm_id): llm_id for llm_id in expected_llms}


def get_expected_stages() -> List[str]:
    """Get expected curriculum stages"""
    return [
        "stage1_mcq",
        "stage2_captioning", 
        "stage3_cot",
        "stage4_sleep_cot",
        "stage5_ecg_cot"
    ]


def get_expected_model_types() -> List[str]:
    """Get expected model types"""
    return ["EmbedHealthSP", "EmbedHealthFlamingo"]


def check_checkpoint_exists(results_dir: str, llm_dir: str, model_type: str, stage: str) -> bool:
    """Check if a checkpoint exists for a specific stage"""
    checkpoint_path = os.path.join(results_dir, llm_dir, model_type, stage, "checkpoints", "best_model.pt")
    return os.path.exists(checkpoint_path)


def check_metrics_exist(results_dir: str, llm_dir: str, model_type: str, stage: str) -> Tuple[bool, Dict]:
    """Check if metrics file exists and return its contents"""
    metrics_path = os.path.join(results_dir, llm_dir, model_type, stage, "results", "metrics.json")
    
    if not os.path.exists(metrics_path):
        return False, {}
    
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return True, metrics
    except (json.JSONDecodeError, IOError):
        return False, {}


def check_test_predictions_exist(results_dir: str, llm_dir: str, model_type: str, stage: str) -> bool:
    """Check if test predictions file exists"""
    predictions_path = os.path.join(results_dir, llm_dir, model_type, stage, "results", "test_predictions.jsonl")
    return os.path.exists(predictions_path)


def analyze_results_directory(results_dir: str = "results") -> Dict:
    """Analyze the results directory and return missing checkpoint information"""
    
    if not os.path.exists(results_dir):
        print(f"❌ Results directory '{results_dir}' does not exist!")
        return {}
    
    expected_llms = get_expected_llm_directories()
    expected_stages = get_expected_stages()
    expected_models = get_expected_model_types()
    
    # Find actual LLM directories
    actual_llm_dirs = []
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path):
            actual_llm_dirs.append(item)
    
    print(f"📁 Found LLM directories: {actual_llm_dirs}")
    print(f"🔍 Expected LLM directories: {list(expected_llms.keys())}")
    print()
    
    results = {
        "missing_checkpoints": {},
        "missing_metrics": {},
        "missing_predictions": {},
        "completed_stages": {},
        "summary": {}
    }
    
    for llm_dir in actual_llm_dirs:
        llm_original = expected_llms.get(llm_dir, f"Unknown LLM ({llm_dir})")
        
        print(f"🔍 Analyzing {llm_dir} ({llm_original})")
        
        results["missing_checkpoints"][llm_dir] = {}
        results["missing_metrics"][llm_dir] = {}
        results["missing_predictions"][llm_dir] = {}
        results["completed_stages"][llm_dir] = {}
        
        for model_type in expected_models:
            model_path = os.path.join(results_dir, llm_dir, model_type)
            
            if not os.path.exists(model_path):
                print(f"   ⚠️  {model_type} directory not found")
                results["missing_checkpoints"][llm_dir][model_type] = expected_stages.copy()
                results["missing_metrics"][llm_dir][model_type] = expected_stages.copy()
                results["missing_predictions"][llm_dir][model_type] = expected_stages.copy()
                results["completed_stages"][llm_dir][model_type] = []
                continue
            
            print(f"   📊 {model_type}:")
            
            missing_checkpoints = []
            missing_metrics = []
            missing_predictions = []
            completed_stages = []
            
            for stage in expected_stages:
                stage_path = os.path.join(model_path, stage)
                
                if not os.path.exists(stage_path):
                    print(f"      ❌ {stage}: Stage directory not found")
                    missing_checkpoints.append(stage)
                    missing_metrics.append(stage)
                    missing_predictions.append(stage)
                    continue
                
                # Check checkpoint
                checkpoint_exists = check_checkpoint_exists(results_dir, llm_dir, model_type, stage)
                
                # Check metrics
                metrics_exists, metrics_data = check_metrics_exist(results_dir, llm_dir, model_type, stage)
                
                # Check test predictions
                predictions_exists = check_test_predictions_exist(results_dir, llm_dir, model_type, stage)
                
                # Determine if stage is completed
                is_completed = (
                    checkpoint_exists and 
                    metrics_exists and 
                    predictions_exists and
                    metrics_data.get("completed", False)
                )
                
                if is_completed:
                    completed_stages.append(stage)
                    print(f"      ✅ {stage}: Complete")
                else:
                    issues = []
                    if not checkpoint_exists:
                        issues.append("no checkpoint")
                        missing_checkpoints.append(stage)
                    if not metrics_exists:
                        issues.append("no metrics")
                        missing_metrics.append(stage)
                    if not predictions_exists:
                        issues.append("no predictions")
                        missing_predictions.append(stage)
                    elif metrics_exists and not metrics_data.get("completed", False):
                        issues.append("not marked complete")
                    
                    print(f"      ⚠️  {stage}: {', '.join(issues)}")
            
            results["missing_checkpoints"][llm_dir][model_type] = missing_checkpoints
            results["missing_metrics"][llm_dir][model_type] = missing_metrics
            results["missing_predictions"][llm_dir][model_type] = missing_predictions
            results["completed_stages"][llm_dir][model_type] = completed_stages
        
        print()
    
    # Generate summary
    for llm_dir in actual_llm_dirs:
        results["summary"][llm_dir] = {}
        for model_type in expected_models:
            if model_type in results["completed_stages"][llm_dir]:
                completed = len(results["completed_stages"][llm_dir][model_type])
                total = len(expected_stages)
                results["summary"][llm_dir][model_type] = {
                    "completed": completed,
                    "total": total,
                    "completion_rate": completed / total if total > 0 else 0
                }
    
    return results


def print_summary_report(results: Dict):
    """Print a summary report of missing checkpoints"""
    
    print("=" * 80)
    print("📊 SUMMARY REPORT")
    print("=" * 80)
    
    expected_stages = get_expected_stages()
    expected_models = get_expected_model_types()
    
    for llm_dir, llm_data in results["summary"].items():
        print(f"\n🤖 {llm_dir}")
        print("-" * 40)
        
        for model_type in expected_models:
            if model_type in llm_data:
                summary = llm_data[model_type]
                completion_rate = summary["completion_rate"]
                
                print(f"   {model_type}:")
                print(f"      Completed: {summary['completed']}/{summary['total']} stages ({completion_rate:.1%})")
                
                # Show missing checkpoints
                missing_checkpoints = results["missing_checkpoints"][llm_dir].get(model_type, [])
                if missing_checkpoints:
                    print(f"      ❌ Missing checkpoints: {', '.join(missing_checkpoints)}")
                else:
                    print(f"      ✅ All checkpoints present")
                
                # Show completed stages
                completed_stages = results["completed_stages"][llm_dir].get(model_type, [])
                if completed_stages:
                    print(f"      ✅ Completed stages: {', '.join(completed_stages)}")


def print_detailed_report(results: Dict):
    """Print a detailed report of all findings"""
    
    print("\n" + "=" * 80)
    print("📋 DETAILED REPORT")
    print("=" * 80)
    
    expected_stages = get_expected_stages()
    expected_models = get_expected_model_types()
    
    for llm_dir, llm_data in results["missing_checkpoints"].items():
        print(f"\n🤖 {llm_dir}")
        print("=" * 50)
        
        for model_type in expected_models:
            if model_type in llm_data:
                print(f"\n📊 {model_type}")
                print("-" * 30)
                
                # Missing checkpoints
                missing_checkpoints = results["missing_checkpoints"][llm_dir][model_type]
                if missing_checkpoints:
                    print(f"❌ Missing checkpoints ({len(missing_checkpoints)}):")
                    for stage in missing_checkpoints:
                        print(f"   - {stage}")
                else:
                    print("✅ All checkpoints present")
                
                # Missing metrics
                missing_metrics = results["missing_metrics"][llm_dir][model_type]
                if missing_metrics:
                    print(f"📊 Missing metrics ({len(missing_metrics)}):")
                    for stage in missing_metrics:
                        print(f"   - {stage}")
                
                # Missing predictions
                missing_predictions = results["missing_predictions"][llm_dir][model_type]
                if missing_predictions:
                    print(f"📝 Missing predictions ({len(missing_predictions)}):")
                    for stage in missing_predictions:
                        print(f"   - {stage}")
                
                # Completed stages
                completed_stages = results["completed_stages"][llm_dir][model_type]
                if completed_stages:
                    print(f"✅ Completed stages ({len(completed_stages)}):")
                    for stage in completed_stages:
                        print(f"   - {stage}")


def save_results_to_file(results: Dict, output_file: str = "checkpoint_analysis.json"):
    """Save results to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")


def main():
    """Main function"""
    print("🔍 EmbedHealth Checkpoint Analysis")
    print("=" * 50)
    
    # Analyze results directory
    results = analyze_results_directory()
    
    if not results:
        print("❌ No results found or analysis failed")
        return
    
    # Print reports
    print_summary_report(results)
    print_detailed_report(results)
    
    # Save results to file
    save_results_to_file(results)
    
    print("\n🎉 Analysis complete!")


if __name__ == "__main__":
    main()
