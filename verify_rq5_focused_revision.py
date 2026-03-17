#!/usr/bin/env python3
"""
Verification script for RQ5 focused revision with dataset consistency and time reduction emphasis
"""

def verify_dataset_consistency():
    """Verify that RQ5 correctly describes dataset and model consistency with RQ2"""
    
    print("=" * 70)
    print("Dataset and Model Consistency Verification")
    print("=" * 70)
    
    try:
        with open("paper/eval.tex", "r") as f:
            content = f.read()
        
        # Extract RQ5 section
        rq5_start = content.find("\\subsection{RQ5:")
        rq5_end = content.find("\\end{tcolorbox}", rq5_start) + len("\\end{tcolorbox}")
        
        if rq5_start == -1 or rq5_end == -1:
            print("❌ Could not extract RQ5 section")
            return False
        
        rq5_content = content[rq5_start:rq5_end]
        
        # Check for dataset consistency elements
        consistency_checks = [
            ("Same QF_NIA dataset as RQ2", "same QF\\_NIA dataset as RQ2"),
            ("50% training split", "50\\% (5,021 constraints) for training"),
            ("50% testing split", "50\\% (5,022 constraints) for testing"),
            ("Identical experimental conditions", "identical experimental conditions"),
            ("Same methodology", "same methodology"),
            ("Predictive models identical to RQ2", "identical to those validated in RQ2"),
            ("Experimental consistency", "experimental consistency"),
        ]
        
        print("\nDataset Consistency Elements:")
        all_present = True
        for check_name, pattern in consistency_checks:
            if pattern in rq5_content:
                print(f"✅ {check_name}: Present")
            else:
                print(f"❌ {check_name}: Missing")
                all_present = False
        
        return all_present
        
    except Exception as e:
        print(f"❌ Error in dataset consistency verification: {e}")
        return False

def verify_predictive_model_specification():
    """Verify that predictive models are properly specified"""
    
    print("\n" + "=" * 70)
    print("Predictive Model Specification Verification")
    print("=" * 70)
    
    try:
        with open("paper/eval.tex", "r") as f:
            content = f.read()
        
        # Extract RQ5 section
        rq5_start = content.find("\\subsection{RQ5:")
        rq5_end = content.find("\\end{tcolorbox}", rq5_start) + len("\\end{tcolorbox}")
        rq5_content = content[rq5_start:rq5_end]
        
        # Check for model specification elements
        model_checks = [
            ("Binary classification model", "Binary Classification Model"),
            ("SAT vs non-SAT prediction", "SAT vs. non-SAT"),
            ("Time estimation model", "Time Estimation Model"),
            ("Threshold-based approach", "threshold-based approach"),
            ("Threshold ≥ 4", "threshold (≥ 4)"),
            ("Same architecture and training", "identical architecture and training parameters"),
            ("Established in RQ2", "established in RQ2"),
        ]
        
        print("\nPredictive Model Specification Elements:")
        all_present = True
        for check_name, pattern in model_checks:
            if pattern in rq5_content:
                print(f"✅ {check_name}: Present")
            else:
                print(f"❌ {check_name}: Missing")
                all_present = False
        
        return all_present
        
    except Exception as e:
        print(f"❌ Error in model specification verification: {e}")
        return False

def verify_results_table_format():
    """Verify that the results table has the correct format"""
    
    print("\n" + "=" * 70)
    print("Results Table Format Verification")
    print("=" * 70)
    
    try:
        with open("paper/eval.tex", "r") as f:
            content = f.read()
        
        # Extract table
        table_start = content.find("\\begin{table}[htbp]")
        table_end = content.find("\\end{table}", table_start) + len("\\end{table}")
        
        if table_start == -1 or table_end == -1:
            print("❌ Could not extract results table")
            return False
        
        table_content = content[table_start:table_end]
        
        # Check for required table elements
        table_checks = [
            ("5,022 constraints in caption", "5,022 constraints"),
            ("SAT Count column", "SAT Count"),
            ("UNSAT Count column", "UNSAT Count"),
            ("UNKNOWN Count column", "UNKNOWN Count"),
            ("SAT Avg. Time column", "SAT Avg. Time"),
            ("Overall Avg. Time column", "Overall Avg. Time"),
            ("Direct Solving Only row", "Direct Solving Only"),
            ("Hybrid Routing row", "Hybrid Routing (Predictive)"),
            ("No RL+LLM Only row", "RL+LLM Only"),
        ]
        
        print("\nResults Table Elements:")
        all_present = True
        for check_name, pattern in table_checks:
            if check_name == "No RL+LLM Only row":
                if pattern not in table_content:
                    print(f"✅ {check_name}: Correctly absent")
                else:
                    print(f"❌ {check_name}: Should be removed")
                    all_present = False
            else:
                if pattern in table_content:
                    print(f"✅ {check_name}: Present")
                else:
                    print(f"❌ {check_name}: Missing")
                    all_present = False
        
        return all_present
        
    except Exception as e:
        print(f"❌ Error in table format verification: {e}")
        return False

def verify_time_reduction_emphasis():
    """Verify that the analysis emphasizes time reduction benefits"""
    
    print("\n" + "=" * 70)
    print("Time Reduction Emphasis Verification")
    print("=" * 70)
    
    try:
        with open("paper/eval.tex", "r") as f:
            content = f.read()
        
        # Extract RQ5 section
        rq5_start = content.find("\\subsection{RQ5:")
        rq5_end = content.find("\\end{tcolorbox}", rq5_start) + len("\\end{tcolorbox}")
        rq5_content = content[rq5_start:rq5_end]
        
        # Check for time reduction emphasis
        time_checks = [
            ("12.6% SAT time reduction", "12.6\\% reduction"),
            ("11.9% overall time reduction", "11.9\\% reduction"),
            ("157.6 seconds saved per SAT", "157.6 seconds saved"),
            ("114.2 seconds saved per constraint", "114.2 seconds saved"),
            ("Time reduction benefits", "time reduction benefits"),
            ("Significant time reduction", "Significant Time Reduction"),
            ("Time Reduction Analysis", "Time Reduction Analysis"),
            ("Efficiency gains", "efficiency gains"),
        ]
        
        print("\nTime Reduction Emphasis Elements:")
        all_present = True
        for check_name, pattern in time_checks:
            if pattern in rq5_content:
                print(f"✅ {check_name}: Present")
            else:
                print(f"❌ {check_name}: Missing")
                all_present = False
        
        return all_present
        
    except Exception as e:
        print(f"❌ Error in time reduction verification: {e}")
        return False

def verify_solution_quality_maintenance():
    """Verify that solution quality maintenance is addressed"""
    
    print("\n" + "=" * 70)
    print("Solution Quality Maintenance Verification")
    print("=" * 70)
    
    try:
        with open("paper/eval.tex", "r") as f:
            content = f.read()
        
        # Extract RQ5 section
        rq5_start = content.find("\\subsection{RQ5:")
        rq5_end = content.find("\\end{tcolorbox}", rq5_start) + len("\\end{tcolorbox}")
        rq5_content = content[rq5_start:rq5_end]
        
        # Check for solution quality elements
        quality_checks = [
            ("41 additional SAT instances", "41 additional SAT instances"),
            ("3,300 vs. 3,259", "3,300 vs. 3,259"),
            ("Solution quality", "solution quality"),
            ("Enhanced solution capability", "Enhanced Solution Capability"),
            ("Satisfiability categories", "satisfiability categories"),
            ("Maintaining solution quality", "maintaining solution quality"),
            ("All satisfiability categories", "all satisfiability categories"),
        ]
        
        print("\nSolution Quality Elements:")
        all_present = True
        for check_name, pattern in quality_checks:
            if pattern in rq5_content:
                print(f"✅ {check_name}: Present")
            else:
                print(f"❌ {check_name}: Missing")
                all_present = False
        
        return all_present
        
    except Exception as e:
        print(f"❌ Error in solution quality verification: {e}")
        return False

def verify_focused_comparison():
    """Verify that the comparison is focused on hybrid vs direct solving"""
    
    print("\n" + "=" * 70)
    print("Focused Comparison Verification")
    print("=" * 70)
    
    try:
        with open("paper/eval.tex", "r") as f:
            content = f.read()
        
        # Extract RQ5 section
        rq5_start = content.find("\\subsection{RQ5:")
        rq5_end = content.find("\\end{tcolorbox}", rq5_start) + len("\\end{tcolorbox}")
        rq5_content = content[rq5_start:rq5_end]
        
        # Check for focused comparison elements
        comparison_checks = [
            ("Hybrid vs direct solving", "direct solving and our hybrid routing"),
            ("Comprehensive comparison", "comprehensive comparison"),
            ("Compared to direct solving alone", "compared to direct solving alone"),
            ("Traditional solvers alone", "traditional solvers alone"),
            ("Selective RL+LLM application", "selective RL+LLM application"),
            ("0.8% of constraints", "0.8\\% of constraints"),
        ]
        
        print("\nFocused Comparison Elements:")
        all_present = True
        for check_name, pattern in comparison_checks:
            if pattern in rq5_content:
                print(f"✅ {check_name}: Present")
            else:
                print(f"❌ {check_name}: Missing")
                all_present = False
        
        return all_present
        
    except Exception as e:
        print(f"❌ Error in focused comparison verification: {e}")
        return False

def main():
    """Main verification function"""
    
    print("RQ5 Focused Revision Verification")
    print("Verifying dataset consistency, model specification, and time reduction emphasis")
    
    # Run all verification checks
    consistency_ok = verify_dataset_consistency()
    models_ok = verify_predictive_model_specification()
    table_ok = verify_results_table_format()
    time_ok = verify_time_reduction_emphasis()
    quality_ok = verify_solution_quality_maintenance()
    comparison_ok = verify_focused_comparison()
    
    # Final summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    checks = [
        ("Dataset and Model Consistency", consistency_ok),
        ("Predictive Model Specification", models_ok),
        ("Results Table Format", table_ok),
        ("Time Reduction Emphasis", time_ok),
        ("Solution Quality Maintenance", quality_ok),
        ("Focused Comparison", comparison_ok),
    ]
    
    for check_name, result in checks:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {check_name}")
    
    overall_success = all(result for _, result in checks)
    
    if overall_success:
        print("\n🎉 OVERALL VERIFICATION: PASSED")
        print("\nRQ5 successfully revised with focused approach:")
        print("- Dataset and model consistency with RQ2 established")
        print("- Predictive models properly specified")
        print("- Results table shows comprehensive SAT/UNSAT/UNKNOWN breakdown")
        print("- Time reduction benefits clearly emphasized")
        print("- Solution quality maintenance demonstrated")
        print("- Focused comparison between hybrid and direct solving")
    else:
        print("\n⚠️  OVERALL VERIFICATION: NEEDS ATTENTION")
        print("\nSome aspects of the RQ5 revision require further refinement.")
    
    return overall_success

if __name__ == "__main__":
    success = main()
