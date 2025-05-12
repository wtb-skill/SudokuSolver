from dev_tools.data_utils.generate_digit_dataset import DigitDatasetGenerator

if __name__ == "__main__":

    dataset_names = [
        'digit_dataset_v1_2a',
        'digit_dataset_v1_2b',
        'digit_dataset_v1_2c',
        'digit_dataset_v1_2d',
        'digit_dataset_v1_2e',
        'digit_dataset_v1_2f',
        'digit_dataset_v1_2g',
        'digit_dataset_v1_2h',
        'digit_dataset_v1_2i',
        'digit_dataset_v1_2j',
    ]

    for name in dataset_names:
        print(f"[INFO] Generating dataset: {name}")

        generator = DigitDatasetGenerator(
        image_size = 32,
        output_dir = name,
        num_samples = 100,
        blur_level = 9,
        shift_range = 1,
        rotation_range = 10,
        noise_level = 10,
        fonts_dir = "fonts",
        clean_proportion = 0.3
        )
        generator.generate_images()
        print(f"[INFO] Completed dataset: {name}")

    print("[INFO] All datasets generated successfully!")