import os
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from collections import Counter
import cv2
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class ECGWaveformImageGenerator:
    def __init__(self, db_path, output_dir, image_size=(1920, 1080)):
        self.db_path = db_path
        self.output_dir = output_dir
        self.image_size = image_size

        self.classes = sorted([
            'Normal',
            'Preexcitation',
            'RBBB',
            'LBBB',
            'AV_Block_1',
            'Junctional_Escape',
            'VT',
            'IVR',
            'AVRT',
            'AVNRT',
            'Aberrated_AVRT',
            'Aberrated_AVNRT',
            'AVNRT_AV_Block_2',
            'AFIB',
            'EAT',
            'AFL',
            'PAC',  
            'Aberrated_PAC',
            'Preexcitated_PAC',
            'PVC', 
            'Fusion_Beat',
            'PJC',  
            'Atrial_Paced',
            'Ventricular_Paced',
            'Fusion_Paced_Normal'
        ])

        self.class_mapping = {
            ('N', ''): 'Normal',
            ('•', ''): 'Normal', 
            ('N', 'N-Prex'): 'Preexcitation',
            ('R', ''): 'RBBB',
            ('L', ''): 'LBBB',
            ('b', 'BI'): 'AV_Block_1',
            ('j', ''): 'Junctional_Escape',
            ('X', 'VT'): 'VT',
            ('X', 'IVR'): 'IVR',
            ('X', 'AVRT'): 'AVRT',
            ('X', 'AVNRT'): 'AVNRT',
            ('X', 'avrt'): 'Aberrated_AVRT',
            ('X', 'avnrt'): 'Aberrated_AVNRT',
            ('X', 'AVNRT+BII'): 'AVNRT_AV_Block_2',
            ('X', 'AFIB'): 'AFIB',
            ('X', 'EAT'): 'EAT',
            ('X', 'AFL'): 'AFL',
            ('A', ''): 'PAC',
            ('a', ''): 'Aberrated_PAC',
            ('A', 'A-Prex'): 'Preexcitated_PAC',
            ('V', ''): 'PVC',
            ('F', ''): 'Fusion_Beat',
            ('J', ''): 'PJC',
            ('/', '/A'): 'Atrial_Paced',
            ('/', '/V'): 'Ventricular_Paced',
            ('f', ''): 'Fusion_Paced_Normal',
        }

        self.records = self.get_records()
        self.create_output_dirs()

    def get_records(self):
        """
        MODIFIED: Get record names for the children's dataset (x001 - x029)
        - The filter is changed to find all records starting with 'x0'
        """
        records = []
        print(f"Searching for records in: {self.db_path}")
        for f in os.listdir(self.db_path):
            if f.endswith('.dat') and not f.startswith('.'):
                record_name = f.split('.')[0]
                if record_name.startswith('x0'):
                    atr_file = os.path.join(self.db_path, f"{record_name}.atr")
                    if os.path.exists(atr_file):
                        records.append(record_name)

        print(f"Found {len(records)} valid children's records (starting with 'x0')")
        return sorted(records)

    def create_output_dirs(self):
        """Create output directories for each class"""
        print(f"Creating output directories for {len(self.classes)} classes...")
        for class_name in self.classes:
            class_dir = os.path.join(self.output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

    def map_annotation_to_class(self, symbol, aux_note):
        """
        Map (symbol, aux_note) to our selected classes
        """
        key = (symbol, aux_note if aux_note else '')
        return self.class_mapping.get(key, None)

    def extract_beats(self, signal, annotation, fs=977, window_before=268, window_after=433):
        """
        Extract individual beats from ECG signal.
        - Default fs is 977 Hz for LHC dataset.
        - Window size is adjusted to keep the same *time* as the 360Hz script
        """
        beats = []
        labels = []
        positions = []

        if signal.shape[1] > 1:
            ecg_signal = signal[:, 0]
        else:
            ecg_signal = signal.flatten()

        total_samples = len(ecg_signal)

        if not hasattr(annotation, 'aux_note'):
            print(f"Warning: Record seems to be missing 'aux_note' field. Cannot map complex classes.")
            return [], [], []

        for i, sample in enumerate(annotation.sample):
            start = sample - window_before
            end = sample + window_after

            if start >= 0 and end < total_samples:
                beat = ecg_signal[start:end]

                symbol = annotation.symbol[i]
                aux = annotation.aux_note[i] if i < len(annotation.aux_note) else ''

                label = self.map_annotation_to_class(symbol, aux)

                if len(beat) == (window_before + window_after) and label is not None:
                    beats.append(beat)
                    labels.append(label)
                    positions.append(sample)

        return beats, labels, positions

    def create_waveform_image(self, beat, filename, class_name, fs=977):
        """
        Generates the 1920x1080 black, full-width, no-title image.
        """

        fig_width = self.image_size[0] / 100.0
        fig_height = self.image_size[1] / 100.0

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)
        ax = fig.add_subplot(111)

        time_axis = np.arange(len(beat))

        ax.plot(time_axis, beat, color='black', linewidth=3)

        y_min = beat.min()
        y_max = beat.max()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        ax.set_xlim(time_axis.min(), time_axis.max())

        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(False)

        plt.tight_layout(pad=0)
        plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def generate_images_for_record(self, record, current_totals, max_per_class):
        """
        Generate waveform images for a single record.
        """
        generated_count_this_record = {cls: 0 for cls in self.classes}

        try:
            record_path = os.path.join(self.db_path, record)
            signal, fields = wfdb.rdsamp(record_path)
            annotation = wfdb.rdann(record_path, 'atr')

            fs = fields.get('fs', 977)
            if fs != 977:
                print(f"Warning: Record {record} has fs={fs}, not 977. Adjusting window...")
                window_before = int(0.275 * fs)
                window_after = int(0.444 * fs)
            else:
                window_before = 268
                window_after = 433

            beats, labels, positions = self.extract_beats(signal, annotation, fs, window_before, window_after)

            for i, (beat, label, pos) in enumerate(zip(beats, labels, positions)):

                if current_totals.get(label, 0) + generated_count_this_record.get(label, 0) >= max_per_class:
                    continue

                filename = f"{record}_{pos:010d}_{label}.png"
                filepath = os.path.join(self.output_dir, label, filename)

                self.create_waveform_image(beat, filepath, label, fs)

                generated_count_this_record[label] += 1

            return generated_count_this_record

        except Exception as e:
            print(f"Error processing record {record}: {e}")
            return generated_count_this_record

    def generate_dataset(self, max_per_class=1000):
        """
        Generate the complete dataset by iterating through all records.
        """
        print("🚀 STARTING ECG WAVEFORM IMAGE GENERATION (LHC Dataset)")
        print("=" * 50)
        print(f"Target image size: {self.image_size}")
        print(f"Max images per class: {max_per_class}")
        print(f"Output directory: {self.output_dir}")
        print(f"Processing {len(self.records)} records for {len(self.classes)} classes...")

        total_generated = {cls: 0 for cls in self.classes}

        for record in tqdm(self.records, desc="Processing Records"):

            all_full = all(total_generated.get(cls, 0) >= max_per_class for cls in self.classes)
            if all_full:
                print("All class quotas met. Stopping early.")
                break

            generated_in_record = self.generate_images_for_record(record, total_generated, max_per_class)

            summary = []
            for cls, count in generated_in_record.items():
                if count > 0:
                    total_generated[cls] += count
                    summary.append(f"+{count} {cls}")

            if summary:
                tqdm.write(f"  {record}: {', '.join(summary)}")

        print("\n🎉 DATASET GENERATION COMPLETE!")
        print("=" * 30)
        total_images = sum(total_generated.values())
        print(f"Total images generated: {total_images}")
        for class_name in self.classes:
            if total_generated[class_name] > 0:
                print(f"  {class_name}: {total_generated[class_name]} images")

        return total_generated


def verify_dataset(output_dir):
    """
    Verify the generated dataset dynamically.
    """
    import glob

    print("\n🔍 VERIFYING GENERATED DATASET")
    print("=" * 30)

    try:
        classes_in_dir = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        classes_in_dir = sorted(classes_in_dir)
        if not classes_in_dir:
            print("No class folders found in output directory.")
            return
    except Exception as e:
        print(f"Could not read output directory {output_dir}: {e}")
        return

    print(f"Found {len(classes_in_dir)} class folders.")

    for class_name in classes_in_dir:
        class_dir = os.path.join(output_dir, class_name)
        images = glob.glob(os.path.join(class_dir, "*.png"))

        if images:
            img = cv2.imread(images[0])
            print(f"  {class_name}: {len(images)} images (e.g., {img.shape})")
        else:
            print(f"  {class_name}: 0 images")

    print("\n📸 DISPLAYING SAMPLE IMAGES (up to 6):")
    display_classes = classes_in_dir[:6]
    num_classes = len(display_classes)

    if num_classes == 0:
        print("No images to display.")
        return

    num_cols = 3
    num_rows = (num_classes + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    if num_rows == 1:
        if num_cols == 1:
            axes = np.array([axes])
        else:
            axes = np.array([axes])

    axes = axes.ravel()

    for i, class_name in enumerate(display_classes):
        class_dir = os.path.join(output_dir, class_name)
        images = glob.glob(os.path.join(class_dir, "*.png"))

        if images:
            img = cv2.imread(images[0])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            axes[i].imshow(img_rgb)
            axes[i].set_title(f'{class_name}\n{img.shape[1]}x{img.shape[0]}', fontweight='bold')
            axes[i].axis('off')

    for i in range(num_classes, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('dataset_samples_LHC.png', dpi=150, bbox_inches='tight')
    print("\nSaved sample image grid to 'dataset_samples_LHC.png'")


def main():
    DB_PATH = r'D:\Maestría\Master of Science in Electrical Engineering (Communications)\Fourth Semester\Heterogenous Computing\Project\leipzig-heart-center-ecg-database-arrhythmias-in-children-and-patients-with-congenital-heart-disease-1.0.0'
    OUTPUT_DIR = r'D:\Maestría\Master of Science in Electrical Engineering (Communications)\Fourth Semester\Heterogenous Computing\Project\ECG_Waveform_Images_HD_children_dataset'

    IMAGE_SIZE = (1920, 1080)
    IMAGES_PER_CLASS = 1000 

    generator = ECGWaveformImageGenerator(DB_PATH, OUTPUT_DIR, IMAGE_SIZE)

    results = generator.generate_dataset(max_per_class=IMAGES_PER_CLASS)

    verify_dataset(OUTPUT_DIR)

    print(f"\n✅ All ECG waveform images have been generated!")
    print(f"📍 Location: {os.path.abspath(OUTPUT_DIR)}")
    print(f"📁 Organization: Separate folders for each class")
    print(f"🖼️ Image size: {IMAGE_SIZE} pixels")
    print(f"📊 Total: {sum(results.values())} images")


if __name__ == "__main__":
    main()
