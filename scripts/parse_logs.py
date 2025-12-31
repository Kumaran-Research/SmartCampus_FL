import os
import csv
import re

def parse_logs(log_dir, output_csv):
    results = []
    for log_file in os.listdir(log_dir):
        if log_file.endswith('.txt'):
            client_name = log_file.replace('_log.txt', '')
            with open(os.path.join(log_dir, log_file), 'r') as f:
                content = f.read()
                # Extract accuracy (simple regex for "accuracy: X.XX")
                match = re.search(r'accuracy: (\d+\.\d+)', content)
                if match:
                    accuracy = float(match.group(1))
                    results.append({'client': client_name, 'accuracy': accuracy})
    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['client', 'accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Parsed results saved to {output_csv}")

if __name__ == "__main__":
    parse_logs('experiments/exp_01_baseline/logs', 'results/tables/experiment_results.csv')
