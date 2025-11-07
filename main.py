"""
Main entry point for the Convex-PSRL experimental pipeline.

This script orchestrates:
1. Generation of conceptual figures (Figure 1 & 2)
2. Running experiments with all methods
3. Generation of learning curve comparison (Figure 3)

Usage:
    python main.py                    # Run full pipeline
    python main.py --figures-only     # Generate only conceptual figures
    python main.py --experiments-only # Run only experiments
"""

import sys
import argparse
from pathlib import Path
import time

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))


def generate_conceptual_figures():
    """Generate Figure 1 and Figure 2 (conceptual diagrams)."""
    print("\n" + "="*70)
    print("GENERATING CONCEPTUAL FIGURES")
    print("="*70)
    
    # Figure 1: Conceptual Contrast
    print("\n[1/2] Generating Figure 1: Conceptual Contrast Diagram...")
    try:
        from src.generate_figure1 import create_figure1
        figures_dir = Path(__file__).parent / 'figures'
        figures_dir.mkdir(exist_ok=True)
        create_figure1(str(figures_dir / 'figure1.pdf'))
        print("✓ Figure 1 generated successfully!")
    except Exception as e:
        print(f"✗ Error generating Figure 1: {e}")
        import traceback
        traceback.print_exc()
    
    # Figure 2: Pipeline Flowchart
    print("\n[2/2] Generating Figure 2: Convex Optimization Pipeline...")
    try:
        from src.generate_figure2 import create_figure2
        create_figure2(str(figures_dir / 'figure2.pdf'))
        print("✓ Figure 2 generated successfully!")
    except Exception as e:
        print(f"✗ Error generating Figure 2: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "-"*70)
    print("Conceptual figures complete!")
    print("-"*70 + "\n")


def run_experiments():
    """Run experiments and generate Figure 3."""
    print("\n" + "="*70)
    print("RUNNING EXPERIMENTS")
    print("="*70)
    
    print("\n⚠ Note: Experiments may take 10-30 minutes depending on your system.")
    print("Running with 5 seeds, 100 episodes per method...\n")
    
    start_time = time.time()
    
    try:
        from src.run_experiments import generate_all_environment_figures
        generate_all_environment_figures()
        print("✓ Experiments and Figure 3 generated successfully!")
    except Exception as e:
        print(f"✗ Error running experiments: {e}")
        import traceback
        traceback.print_exc()
    
    elapsed = time.time() - start_time
    print(f"\n⏱ Total experiment time: {elapsed/60:.1f} minutes")
    
    print("\n" + "-"*70)
    print("Experiments complete!")
    print("-"*70 + "\n")


def verify_outputs():
    """Verify that all expected outputs have been generated."""
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70 + "\n")
    
    figures_dir = Path(__file__).parent / 'figures'
    results_dir = Path(__file__).parent / 'results'
    
    expected_figures = [
        figures_dir / 'figure1.pdf',
        figures_dir / 'figure2.pdf',
        figures_dir / 'figure3.pdf'
    ]
    
    expected_results = [
        results_dir / 'cartpole_results.pkl'
    ]
    
    print("Checking generated files:")
    print("-" * 70)
    
    all_present = True
    
    # Check figures
    for fig_path in expected_figures:
        if fig_path.exists():
            size = fig_path.stat().st_size / 1024  # KB
            print(f"✓ {fig_path.name:<30} ({size:.1f} KB)")
        else:
            print(f"✗ {fig_path.name:<30} MISSING")
            all_present = False
    
    # Check results
    for res_path in expected_results:
        if res_path.exists():
            size = res_path.stat().st_size / 1024  # KB
            print(f"✓ {res_path.name:<30} ({size:.1f} KB)")
        else:
            print(f"✗ {res_path.name:<30} MISSING")
            all_present = False
    
    print("-" * 70)
    
    if all_present:
        print("\n✓ All expected outputs generated successfully!")
        print("\nFigures are located in: figures/")
        print("  - figure1.pdf: Conceptual contrast diagram")
        print("  - figure2.pdf: Convex optimization pipeline")
        print("  - figure3.pdf: Sample efficiency comparison")
        print("\nResults data is located in: results/")
    else:
        print("\n✗ Some outputs are missing. Check error messages above.")
    
    print("\n" + "="*70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Convex-PSRL Experimental Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     # Run full pipeline
  python main.py --figures-only      # Generate only conceptual figures
  python main.py --experiments-only  # Run only experiments
  python main.py --quick             # Quick test (fewer episodes/seeds)
        """
    )
    
    parser.add_argument('--figures-only', action='store_true',
                       help='Generate only conceptual figures (skip experiments)')
    parser.add_argument('--experiments-only', action='store_true',
                       help='Run only experiments (skip conceptual figures)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (fewer episodes and seeds)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" "*15 + "CONVEX-PSRL EXPERIMENTAL PIPELINE")
    print("="*70)
    print("\nEfficient Posterior Sampling in Model-Based RL")
    print("with 2-Layer ReLU Networks")
    print("="*70)
    
    # Quick mode: modify experiment parameters
    if args.quick:
        print("\n⚡ QUICK MODE ENABLED (reduced episodes/seeds for testing)")
        # This would require modifying run_experiments.py or passing parameters
        # For now, just warn the user
        print("   To enable quick mode, edit src/run_experiments.py")
        print("   and reduce n_seeds and n_episodes parameters.\n")
    
    # Execute pipeline based on arguments
    if args.experiments_only:
        run_experiments()
    elif args.figures_only:
        generate_conceptual_figures()
    else:
        # Full pipeline
        generate_conceptual_figures()
        run_experiments()
    
    # Verify outputs
    verify_outputs()
    
    print("="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Check the figures/ directory for generated plots")
    print("2. Review results/ directory for raw experimental data")
    print("3. Include figures in your paper with proper captions")
    print("4. See README.md for detailed reproduction instructions")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
