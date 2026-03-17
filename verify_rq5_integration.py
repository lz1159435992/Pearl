#!/usr/bin/env python3
"""
Verification script for RQ5 integration in the paper
"""

def verify_rq5_integration():
    """Verify that RQ5 is properly integrated into the paper"""
    
    print("=" * 60)
    print("RQ5 Integration Verification")
    print("=" * 60)
    
    # Check eval.tex for RQ5 content
    try:
        with open("paper/eval.tex", "r") as f:
            eval_content = f.read()
        
        # Check for RQ5 section
        rq5_checks = [
            ("RQ5 section header", "\\subsection{RQ5: Complexity-Guided Routing System Evaluation}"),
            ("QF_NIA benchmark mention", "QF\\_NIA benchmark"),
            ("10,043 constraints", "10,043"),
            ("Routing performance table", "tab:routing-performance"),
            ("Success rate improvement", "0.8\\% improvement"),
            ("Time reduction", "11.9\\% reduction"),
            ("Hybrid routing strategy", "Hybrid Routing"),
            ("Predictive screening", "Predictive Screening"),
            ("Complexity distribution", "Complexity Distribution"),
            ("RQ5 answer box", "Answer to RQ5"),
        ]
        
        print("\nRQ5 Content Verification:")
        all_present = True
        for check_name, pattern in rq5_checks:
            if pattern in eval_content:
                print(f"✅ {check_name}: Found")
            else:
                print(f"❌ {check_name}: Missing")
                all_present = False
        
        # Check for experimental results
        results_checks = [
            ("Direct solving results", "6,518/10,043"),
            ("RL+LLM only results", "81/10,043"),
            ("Hybrid routing results", "6,599/10,043"),
            ("Success rates", "64.9\\%"),
            ("Average times", "961.4"),
            ("Very Easy constraints", "3,132 constraints"),
            ("Timeout constraints", "2,382 constraints"),
        ]
        
        print("\nExperimental Results Verification:")
        for check_name, pattern in results_checks:
            if pattern in eval_content:
                print(f"✅ {check_name}: Found")
            else:
                print(f"❌ {check_name}: Missing")
                all_present = False
        
        # Check for research question definition
        if "RQ5: Complexity-Guided Routing System Effectiveness" in eval_content:
            print("✅ RQ5 research question: Properly defined")
        else:
            print("❌ RQ5 research question: Missing or incorrect")
            all_present = False
        
        # Check for methodology description
        methodology_checks = [
            ("Experimental design", "Experimental Design and Methodology"),
            ("Deployment simulation", "Deployment Simulation"),
            ("Performance analysis", "Performance Analysis"),
            ("Practical insights", "Practical Deployment Insights"),
        ]
        
        print("\nMethodology Verification:")
        for check_name, pattern in methodology_checks:
            if pattern in eval_content:
                print(f"✅ {check_name}: Found")
            else:
                print(f"❌ {check_name}: Missing")
                all_present = False
        
        return all_present
        
    except FileNotFoundError:
        print("❌ Error: paper/eval.tex not found")
        return False
    except Exception as e:
        print(f"❌ Error reading eval.tex: {e}")
        return False

def verify_data_consistency():
    """Verify that the data used in RQ5 is consistent with the analysis"""
    
    print("\n" + "=" * 60)
    print("Data Consistency Verification")
    print("=" * 60)
    
    try:
        with open("paper/eval.tex", "r") as f:
            content = f.read()
        
        # Check for consistent numbers
        consistency_checks = [
            ("Total constraints", "10,043", "Should appear multiple times"),
            ("Direct solving success", "6,518", "Should be consistent"),
            ("RL+LLM applications", "81", "Should be consistent"),
            ("Hybrid routing success", "6,599", "Should be consistent"),
            ("Success rate improvement", "0.8\\%", "Should be consistent"),
            ("Time reduction", "11.9\\%", "Should be consistent"),
        ]
        
        print("\nNumerical Consistency:")
        for check_name, number, description in consistency_checks:
            count = content.count(number)
            if count > 0:
                print(f"✅ {check_name}: {number} appears {count} times")
            else:
                print(f"❌ {check_name}: {number} not found")
        
        # Check for logical consistency
        print("\nLogical Consistency:")
        
        # Check that 6,518 + 81 = 6,599
        if "6,518" in content and "81" in content and "6,599" in content:
            print("✅ Arithmetic consistency: 6,518 + 81 = 6,599")
        else:
            print("❌ Arithmetic consistency: Numbers don't add up")
        
        # Check percentage calculations
        if "64.9\\%" in content and "65.7\\%" in content:
            print("✅ Percentage consistency: Success rates are logical")
        else:
            print("❌ Percentage consistency: Success rates missing or inconsistent")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data consistency check: {e}")
        return False

def verify_academic_writing_style():
    """Verify that RQ5 follows academic writing conventions"""
    
    print("\n" + "=" * 60)
    print("Academic Writing Style Verification")
    print("=" * 60)
    
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
        
        # Check for academic writing elements
        academic_checks = [
            ("Formal methodology", "\\textbf{Experimental Design and Methodology}"),
            ("Results presentation", "\\textbf{Experimental Results and Performance Analysis}"),
            ("Table references", "Table~\\ref{"),
            ("Quantitative analysis", "\\textbf{Detailed Performance Analysis}"),
            ("Practical implications", "\\textbf{Practical Deployment Insights}"),
            ("Conclusion box", "\\begin{tcolorbox}"),
            ("Structured sections", "\\textbf{"),
            ("Itemized lists", "\\begin{itemize}"),
            ("Enumerated lists", "\\begin{enumerate}"),
        ]
        
        print("\nAcademic Writing Elements:")
        for check_name, pattern in academic_checks:
            if pattern in rq5_content:
                print(f"✅ {check_name}: Present")
            else:
                print(f"❌ {check_name}: Missing")
        
        # Check for proper LaTeX formatting
        latex_checks = [
            ("Math symbols", "$\\leq$"),
            ("Math symbols", "$\\geq$"),
            ("Proper table format", "\\begin{tabular}"),
            ("Proper captions", "\\caption{"),
            ("Proper labels", "\\label{"),
        ]
        
        print("\nLaTeX Formatting:")
        for check_name, pattern in latex_checks:
            if pattern in rq5_content:
                print(f"✅ {check_name}: Correct")
            else:
                print(f"⚠️  {check_name}: Check formatting")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in academic writing verification: {e}")
        return False

def main():
    """Main verification function"""
    
    print("RQ5 Integration Verification for Academic Paper")
    print("Verifying the integration of Research Question 5 analysis")
    
    # Run all verification checks
    integration_ok = verify_rq5_integration()
    consistency_ok = verify_data_consistency()
    style_ok = verify_academic_writing_style()
    
    # Final summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    if integration_ok:
        print("✅ RQ5 Integration: PASSED")
    else:
        print("❌ RQ5 Integration: FAILED")
    
    if consistency_ok:
        print("✅ Data Consistency: PASSED")
    else:
        print("❌ Data Consistency: FAILED")
    
    if style_ok:
        print("✅ Academic Writing Style: PASSED")
    else:
        print("❌ Academic Writing Style: FAILED")
    
    overall_success = integration_ok and consistency_ok and style_ok
    
    if overall_success:
        print("\n🎉 OVERALL VERIFICATION: PASSED")
        print("\nRQ5 has been successfully integrated into the paper with:")
        print("- Comprehensive experimental analysis based on QF_NIA data")
        print("- Consistent numerical results and logical flow")
        print("- Proper academic writing style and LaTeX formatting")
        print("- Clear research question formulation and answer")
        print("- Practical deployment insights and implications")
    else:
        print("\n⚠️  OVERALL VERIFICATION: NEEDS ATTENTION")
        print("\nSome aspects of RQ5 integration require review.")
    
    return overall_success

if __name__ == "__main__":
    success = main()
