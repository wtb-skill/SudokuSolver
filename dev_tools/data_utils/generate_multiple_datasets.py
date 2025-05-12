from dev_tools.data_utils.generate_digit_dataset import DigitDatasetGenerator

if __name__ == "__main__":

    dataset_names = [
        'digit_dataset_v2a',
        'digit_dataset_v2b',
        'digit_dataset_v2c',
        'digit_dataset_v2d',
        'digit_dataset_v2e',
        # 'digit_dataset_v2f',
        # 'digit_dataset_v2g',
        # 'digit_dataset_v2h',
        # 'digit_dataset_v2i',
        # 'digit_dataset_v2j',
    ]

    for name in dataset_names:
        print(f"[INFO] Generating dataset: {name}")

        generator = DigitDatasetGenerator(
        image_size = 32,
        output_dir = name,
        num_samples = 10000,
        blur_level = 7,
        shift_range = 6,
        rotation_range = 20,
        noise_level = 10,
        fonts_dir = "fonts",
        clean_proportion = 0.05
        )
        generator.generate_images()
        print(f"[INFO] Completed dataset: {name}")

    print("[INFO] All datasets generated successfully!")