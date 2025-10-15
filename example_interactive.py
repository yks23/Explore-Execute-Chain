#!/usr/bin/env python3
"""
E2C Interactive Demo

Select from example problems or enter your own to see E2C reasoning in action.
"""

import json
import sys
from pathlib import Path
from example_inference import run_inference, parse_e2c_response


def load_example_problems():
    """Load example problems from JSON file."""
    problem_file = Path("example_problems.json")
    if not problem_file.exists():
        return []
    
    with open(problem_file, 'r') as f:
        return json.load(f)


def display_menu(problems):
    """Display problem selection menu."""
    print("\n" + "=" * 70)
    print("E2C Interactive Demo - Example Problems")
    print("=" * 70)
    print()
    
    for i, prob in enumerate(problems, 1):
        print(f"{i}. [{prob['category']}] - {prob['difficulty']}")
        # Show first 80 chars of problem
        preview = prob['problem'][:80] + "..." if len(prob['problem']) > 80 else prob['problem']
        print(f"   {preview}")
        print()
    
    print(f"{len(problems) + 1}. Enter your own problem")
    print(f"{len(problems) + 2}. Exit")
    print()


def main():
    # Check model exists
    model_path = Path("models/released/e2c-qwen3-4b")
    if not model_path.exists():
        print("❌ Error: E2C model not found!")
        print("\nPlease download the model first:")
        print("  cd models")
        print("  bash download_models.sh --model e2c-qwen3-4b")
        print("  cd ..")
        return
    
    # Load example problems
    problems = load_example_problems()
    if not problems:
        print("⚠️  Warning: example_problems.json not found, custom input only")
    
    print("\n" + "🌟" * 35)
    print("Welcome to E2C Interactive Demo!")
    print("🌟" * 35)
    
    while True:
        if problems:
            display_menu(problems)
            
            try:
                choice = input("Select an option (1-{}): ".format(len(problems) + 2))
                choice = int(choice)
            except (ValueError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break
            
            if choice == len(problems) + 2:  # Exit
                print("\nGoodbye!")
                break
            elif choice == len(problems) + 1:  # Custom input
                print("\nEnter your problem (press Enter twice to finish):")
                lines = []
                while True:
                    line = input()
                    if line == "":
                        if lines and lines[-1] == "":
                            break
                    lines.append(line)
                problem = "\n".join(lines[:-1])  # Remove last empty line
            elif 1 <= choice <= len(problems):
                selected = problems[choice - 1]
                print(f"\n📚 Selected: {selected['category']} - {selected['difficulty']}")
                print(f"📌 Expected Answer: {selected['answer']}")
                problem = selected['problem']
            else:
                print("❌ Invalid choice, please try again.")
                continue
        else:
            # No example problems, just ask for custom input
            print("\nEnter your problem (press Enter twice to finish):")
            lines = []
            while True:
                try:
                    line = input()
                    if line == "":
                        if lines and lines[-1] == "":
                            break
                    lines.append(line)
                except KeyboardInterrupt:
                    print("\n\nGoodbye!")
                    return
            problem = "\n".join(lines[:-1])
            
            if not problem.strip():
                print("❌ Empty problem, exiting.")
                break
        
        # Run inference
        try:
            print("\n⏳ Running E2C inference...")
            response = run_inference(
                model_path=str(model_path),
                problem=problem,
                max_tokens=2048,
                temperature=0.7
            )
            
            # Parse and display
            exploration, execution = parse_e2c_response(response)
            
            print("\n" + "=" * 70)
            print("🎯 E2C REASONING RESULT")
            print("=" * 70)
            
            if exploration:
                print("\n🔍 EXPLORATION (Planning Phase):")
                print("-" * 70)
                print(exploration)
                print()
            
            if execution:
                print("⚡ EXECUTION (Detailed Reasoning):")
                print("-" * 70)
                print(execution)
                print()
            
            if not exploration and not execution:
                print("\n📝 FULL RESPONSE:")
                print("-" * 70)
                print(response)
                print()
            
            print("=" * 70)
            
            # Ask to continue
            if problems:
                cont = input("\nTry another problem? (y/n): ").lower()
                if cont != 'y':
                    print("\nGoodbye!")
                    break
            else:
                break
                
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            if problems:
                cont = input("\nTry again? (y/n): ").lower()
                if cont != 'y':
                    break
            else:
                break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        sys.exit(0)

