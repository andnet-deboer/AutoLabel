import os
import yaml
import argparse


class DatasetCleaner:
    def __init__(self, dataset_path):
        """
        Initialize the DatasetCleaner with a dataset path.
        
        Args:
            dataset_path: Path to the YOLO dataset directory
        """
        self.dataset_path = dataset_path
        
    def clean_blank_labels(self):
        """Remove image/label pairs where the label file is empty or blank."""
        removed_count = 0
        
        for split in ['train', 'val']:
            labels_dir = os.path.join(self.dataset_path, 'labels', split)
            images_dir = os.path.join(self.dataset_path, 'images', split)
            
            if not os.path.exists(labels_dir):
                continue
                
            label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            
            for label_file in label_files:
                label_path = os.path.join(labels_dir, label_file)
                
                # Check if label file is empty or contains only whitespace
                with open(label_path, 'r') as f:
                    content = f.read().strip()
                
                if not content:
                    # Find corresponding image file
                    base_name = os.path.splitext(label_file)[0]
                    
                    # Check for common image extensions
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        image_path = os.path.join(images_dir, base_name + ext)
                        if os.path.exists(image_path):
                            os.remove(image_path)
                            print(f"  Removed image: {base_name}{ext}")
                            break
                    
                    # Remove the blank label file
                    os.remove(label_path)
                    print(f"  Removed label: {label_file}")
                    removed_count += 1
        
        if removed_count > 0:
            print(f"\nCleaned {removed_count} blank label pairs from dataset.")
        else:
            print("\nNo blank labels found.")
        
        return removed_count
    
    def create_dataset_yaml(self, class_names):
        """
        Create a dataset.yaml file for YOLO training.
        
        Args:
            class_names: List of class names in order
        """
        yaml_content = {
            'path': os.path.abspath(self.dataset_path),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(class_names),
            'names': {i: name for i, name in enumerate(class_names)}
        }
        yaml_path = os.path.join(self.dataset_path, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        print(f"Generated config: {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean blank labels from YOLO dataset')
    parser.add_argument('--dataset', required=True, help='Path to the YOLO dataset directory')
    args = parser.parse_args()
    
    cleaner = DatasetCleaner(args.dataset)
    cleaner.clean_blank_labels()