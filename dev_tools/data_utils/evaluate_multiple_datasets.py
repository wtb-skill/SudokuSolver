import os
from dev_tools.data_utils.dataset_evaluation import DigitDatasetEvaluator

base_dir = 'all_versions'
evaluation_reports_dir = 'evaluation_reports'
os.makedirs(evaluation_reports_dir, exist_ok=True)

if __name__ == "__main__":
    for folder in os.listdir(base_dir):
        if folder.startswith('digit_dataset_v1'):
            version = folder.split('_')[-1]  # Get the version part, e.g., 'v1e'
            output_subdir = os.path.join(evaluation_reports_dir, version)
            os.makedirs(output_subdir, exist_ok=True)

            dataset_path = os.path.join(base_dir, folder)
            print(f'Running evaluation for: {folder}')

            try:
                evaluator = DigitDatasetEvaluator(dataset_path=dataset_path, output_dir=output_subdir)
                evaluator.run_full_evaluation('4,5,7,9,8,10,11')
                print(f'Evaluation completed for: {folder}')
            except Exception as e:
                print(f'Error evaluating {folder}: {e}')

    print('Batch evaluation completed.')
