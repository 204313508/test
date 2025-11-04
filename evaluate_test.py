import json
import argparse
import sys
import os

def calculate_accuracy(input_path):
    correct = 0
    total = 0
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON on line {line_num} in {input_path}. Skipping.", file=sys.stderr)
                    continue

                label = data.get('label')
                llm_answer = data.get('llm_answer')

                if label is None or llm_answer is None:
                    print(f"Warning: Missing 'label' or 'llm_answer' on line {line_num}. Skipping.", file=sys.stderr)
                    continue

                total += 1
                if label == llm_answer:
                    correct += 1

    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    if total == 0:
        accuracy = 0.0
    else:
        accuracy = correct / total

    return correct, total, accuracy

def main():
    parser = argparse.ArgumentParser(description="Calculate accuracy of LLM answers from a JSONL file.")
    parser.add_argument('--input', required=True, help='Path to the input JSONL file')
    parser.add_argument('--output', required=False, help='Path to the output result file (optional)')

    args = parser.parse_args()

    correct, total, accuracy = calculate_accuracy(args.input)

    input_filename = os.path.basename(args.input)

    result_str = f"File: {input_filename}\nAccuracy: {accuracy:.4f}\n"

    # Print to console
    print(result_str)

    # Write to file if --output is specified
    if args.output:
        try:
            with open(args.output, 'a', encoding='utf-8') as out_f:
                out_f.write(result_str)
            print("--------------------------------")
            print(f"Result saved to: {args.output}")
        except Exception as e:
            print(f"Error writing output file: {e}", file=sys.stderr)

if __name__ == '__main__':
    main()