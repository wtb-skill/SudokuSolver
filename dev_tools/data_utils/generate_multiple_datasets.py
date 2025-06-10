from dev_tools.data_utils.generate_digit_dataset import DigitDatasetGenerator

if __name__ == "__main__":

    dataset_names = [
        "digit_dataset_v2_2a",
        "digit_dataset_v2_2b",
        "digit_dataset_v2_2c",
        "digit_dataset_v2_2d",
        "digit_dataset_v2_2e",
        "digit_dataset_v2_2f",
        "digit_dataset_v2_2g",
        "digit_dataset_v2_2h",
        "digit_dataset_v2_2i",
        "digit_dataset_v2_2j",
    ]

    for name in dataset_names:
        print(f"[INFO] Generating dataset: {name}")

        generator = DigitDatasetGenerator(
            image_size=32,
            output_dir=name,
            num_samples=10000,
            blur_level=7,
            shift_range=4,
            rotation_range=10,
            noise_level=8,
            fonts_dir="fonts",
            clean_proportion=0,
            font_sizes=[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
        )
        generator.generate_images()
        print(f"[INFO] Completed dataset: {name}")

    print("[INFO] All datasets generated successfully!")
