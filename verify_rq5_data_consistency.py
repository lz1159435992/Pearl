#!/usr/bin/env python3
"""
Verification script for RQ5 data consistency
"""

def verify_table_data_consistency():
    """Verify that the table data adds up correctly"""
    
    print("=" * 60)
    print("RQ5 Table Data Consistency Verification")
    print("=" * 60)
    
    try:
        with open("paper/eval.tex", "r") as f:
            content = f.read()
        
        # Extract RQ5 table content specifically
        table_start = content.find("Performance comparison between direct solving and hybrid routing")
        table_start = content.rfind("\\begin{table}[htbp]", 0, table_start)
        table_end = content.find("\\end{table}", table_start) + len("\\end{table}")
        
        if table_start == -1 or table_end == -1:
            print("❌ Could not extract table")
            return False
        
        table_content = content[table_start:table_end]
        print("Table content found:")
        print(table_content)
        
        # Check for 10,043 constraints
        if "10,043 constraints" in table_content:
            print("✅ Total constraints: 10,043 correctly specified")
        else:
            print("❌ Total constraints: Should be 10,043")
            return False
        
        # Extract data rows
        lines = table_content.split('\n')
        direct_solving_line = None
        hybrid_routing_line = None
        
        for line in lines:
            if "Direct Solving Only" in line:
                direct_solving_line = line
            elif "Hybrid Routing" in line:
                hybrid_routing_line = line
        
        if not direct_solving_line or not hybrid_routing_line:
            print("❌ Could not find data rows")
            return False
        
        print(f"\nDirect Solving line: {direct_solving_line}")
        print(f"Hybrid Routing line: {hybrid_routing_line}")
        
        # Parse Direct Solving data
        direct_parts = direct_solving_line.split('&')
        if len(direct_parts) >= 4:
            try:
                direct_sat = int(direct_parts[1].strip().replace(',', ''))
                direct_unsat = int(direct_parts[2].strip().replace(',', ''))
                direct_unknown = int(direct_parts[3].strip())
                direct_total = direct_sat + direct_unsat + direct_unknown
                
                print(f"\nDirect Solving: SAT={direct_sat}, UNSAT={direct_unsat}, UNKNOWN={direct_unknown}")
                print(f"Direct Solving Total: {direct_total}")
                
                if direct_total == 10043:
                    print("✅ Direct Solving data adds up correctly")
                else:
                    print(f"❌ Direct Solving data error: {direct_total} ≠ 10043")
                    return False
                    
            except ValueError as e:
                print(f"❌ Error parsing Direct Solving data: {e}")
                return False
        
        # Parse Hybrid Routing data
        hybrid_parts = hybrid_routing_line.split('&')
        if len(hybrid_parts) >= 4:
            try:
                hybrid_sat = int(hybrid_parts[1].strip().replace(',', ''))
                hybrid_unsat = int(hybrid_parts[2].strip().replace(',', ''))
                hybrid_unknown = int(hybrid_parts[3].strip())
                hybrid_total = hybrid_sat + hybrid_unsat + hybrid_unknown
                
                print(f"\nHybrid Routing: SAT={hybrid_sat}, UNSAT={hybrid_unsat}, UNKNOWN={hybrid_unknown}")
                print(f"Hybrid Routing Total: {hybrid_total}")
                
                if hybrid_total == 10043:
                    print("✅ Hybrid Routing data adds up correctly")
                else:
                    print(f"❌ Hybrid Routing data error: {hybrid_total} ≠ 10043")
                    return False
                    
            except ValueError as e:
                print(f"❌ Error parsing Hybrid Routing data: {e}")
                return False
        
        # Check improvement
        sat_improvement = hybrid_sat - direct_sat
        print(f"\nSAT improvement: {sat_improvement} instances")
        
        if sat_improvement == 81:
            print("✅ SAT improvement matches text (81 instances)")
        else:
            print(f"❌ SAT improvement mismatch: {sat_improvement} ≠ 81")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data verification: {e}")
        return False

def verify_text_consistency():
    """Verify that the text mentions correct numbers"""
    
    print("\n" + "=" * 60)
    print("Text Consistency Verification")
    print("=" * 60)
    
    try:
        with open("paper/eval.tex", "r") as f:
            content = f.read()
        
        # Extract RQ5 section
        rq5_start = content.find("\\subsection{RQ5:")
        rq5_end = content.find("\\end{tcolorbox}", rq5_start) + len("\\end{tcolorbox}")
        rq5_content = content[rq5_start:rq5_end]
        
        # Check for consistent numbers
        checks = [
            ("10,043 constraints", "10,043"),
            ("81 additional SAT instances", "81"),
            ("6,599 vs. 6,518", "6,599"),
            ("6,599 vs. 6,518", "6,518"),
            ("3,444 vs. 3,525", "3,444"),
            ("3,444 vs. 3,525", "3,525"),
        ]
        
        print("\nNumber Consistency Checks:")
        all_consistent = True
        for check_name, number in checks:
            if number in rq5_content:
                print(f"✅ {check_name}: Found")
            else:
                print(f"❌ {check_name}: Missing")
                all_consistent = False
        
        return all_consistent
        
    except Exception as e:
        print(f"❌ Error in text verification: {e}")
        return False

def main():
    """Main verification function"""
    
    print("RQ5 Data Consistency Verification")
    print("Checking that all numbers add up correctly")
    
    table_ok = verify_table_data_consistency()
    text_ok = verify_text_consistency()
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    if table_ok:
        print("✅ Table Data Consistency: PASSED")
    else:
        print("❌ Table Data Consistency: FAILED")
    
    if text_ok:
        print("✅ Text Consistency: PASSED")
    else:
        print("❌ Text Consistency: FAILED")
    
    overall_success = table_ok and text_ok
    
    if overall_success:
        print("\n🎉 OVERALL VERIFICATION: PASSED")
        print("\nAll data is consistent:")
        print("- Total constraints: 10,043")
        print("- Direct Solving: 6,518 SAT + 3,525 UNSAT = 10,043")
        print("- Hybrid Routing: 6,599 SAT + 3,444 UNSAT = 10,043")
        print("- SAT improvement: 81 instances")
    else:
        print("\n⚠️  OVERALL VERIFICATION: FAILED")
        print("Data consistency issues found.")
    
    return overall_success

if __name__ == "__main__":
    success = main()
