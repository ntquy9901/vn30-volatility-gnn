"""
Master test runner for GNN-HAR implementation.

Runs all test suites:
- QLIKE loss tests
- Evaluation metrics tests
- GLASSO adjacency tests
- Model architecture tests

Usage:
    python run_all_tests.py

Date: 2026-05-30
"""

import sys
import os
from pathlib import Path


def run_test_file(test_file, description):
    """Run a single test file by importing and executing it."""
    print(f"\n{'='*70}")
    print(f"  RUNNING: {description}")
    print(f"  File: {test_file}")
    print(f"{'='*70}")

    try:
        # Read and execute the test file
        with open(test_file, 'r') as f:
            test_code = f.read()

        # Create a namespace for the test
        test_namespace = {
            '__name__': '__main__',
            '__file__': os.path.abspath(test_file),
            '__builtins__': __builtins__,
        }

        # Execute the test code
        exec(compile(test_code, test_file, 'exec'), test_namespace)

        print(f"\n[OK] {description} PASSED")
        return True

    except SystemExit as e:
        # Test file called sys.exit()
        if e.code == 0:
            print(f"\n[OK] {description} PASSED")
            return True
        else:
            print(f"\n[FAIL] {description} FAILED (exit code: {e.code})")
            return False
    except AssertionError as e:
        print(f"\n[FAIL] {description} FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] {description} raised exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all test suites."""
    print("\n" + "="*70)
    print("  GNN-HAR IMPLEMENTATION - MASTER TEST SUITE")
    print("  Date: 2026-05-30")
    print("="*70)

    # Change to gnnhar_paper directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Add parent (gnn) directory to path
    gnn_dir = script_dir.parent
    sys.path.insert(0, str(gnn_dir))

    # Test files to run
    test_files = [
        ("test_qlike_loss.py", "QLIKE Loss Tests"),
        ("tests/test_evaluation.py", "Evaluation Metrics Tests"),
        ("tests/test_glasso_adjacency.py", "GLASSO Adjacency Tests"),
        ("tests/test_model_arch.py", "Model Architecture Tests"),
    ]

    # Run all tests
    results = {}
    for test_file, description in test_files:
        if not Path(test_file).exists():
            print(f"\n[SKIP] {test_file} not found")
            results[description] = "SKIPPED"
            continue

        success = run_test_file(test_file, description)
        results[description] = "PASSED" if success else "FAILED"

    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v == "PASSED")
    failed = sum(1 for v in results.values() if v == "FAILED")
    skipped = sum(1 for v in results.values() if v == "SKIPPED")
    total = len(results)

    for description, result in results.items():
        status_symbol = {
            "PASSED": "[OK]",
            "FAILED": "[FAIL]",
            "SKIPPED": "[SKIP]"
        }[result]
        print(f"  {status_symbol} {description}")

    print("\n" + "="*70)
    print(f"  TOTAL: {passed}/{total} passed, {failed} failed, {skipped} skipped")

    if failed == 0 and skipped == 0:
        print("\n  [SUCCESS] ALL TESTS PASSED!")
        print("="*70)
        return 0
    elif failed == 0:
        print(f"\n  [WARNING] Some tests skipped ({skipped} skipped)")
        print("="*70)
        return 0
    else:
        print(f"\n  [FAILURE] {failed} test(s) failed")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
