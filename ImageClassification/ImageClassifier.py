import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from Train_Test import clean_preprocess, prepare_model
import argparse
import os
import json


class ImageClassifier:
    def __init__(self, model_name, checkpoint_path, num_classes=27, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model, self.preprocess = prepare_model(model_name, checkpoint_path=checkpoint_path,
                                                                      num_classes=num_classes)

        self.device = device
        self.model.to(device)
        self.model.eval()

    def predict_image(self, image_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Runs inference on a single image file.
        """
        img = (transforms.ToTensor()(Image.open(image_path).convert('RGB'))).to(torch.float)
        input_tensor = self.preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=-1)

            return probabilities.detach().cpu().numpy()

    def predict_batch(self, image_paths, batch_size=32):
        """
        Runs inference on a batch of image files.
        Returns zero probabilities for missing or corrupted images.

        Parameters
        ----------
        image_paths : list of str
            List of paths to image files
        batch_size : int
            Number of images to process at once (GPU memory dependent)

        Returns
        -------
        probs : np.ndarray, shape (N, num_classes)
            Softmax probabilities for N images (zeros for failed images)
        """
        all_probs = []

        from tqdm import tqdm
        # ... (at the top of the method)
        num_batches = (len(image_paths) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(image_paths), batch_size), total=num_batches, desc="Image batches"):
        # for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []

            # Load and preprocess images
            for img_path in batch_paths:
                # Check if file exists first (more efficient than try/except)
                if not os.path.exists(img_path):
                    batch_tensors.append(None)
                    continue

                try:
                    img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(torch.float)
                    preprocessed = self.preprocess(img)
                    batch_tensors.append(preprocessed)
                except Exception as e:
                    # Corrupted or unreadable image - use zeros
                    batch_tensors.append(None)

            # Filter out None values and track their positions
            valid_indices = [j for j, t in enumerate(batch_tensors) if t is not None]
            valid_tensors = [batch_tensors[j] for j in valid_indices]

            if len(valid_tensors) > 0:
                # Stack valid tensors and move to device
                input_batch = torch.stack(valid_tensors).to(self.device)

                # Predict
                with torch.no_grad():
                    output = self.model(input_batch)
                    probabilities = torch.nn.functional.softmax(output, dim=-1)
                    batch_probs = probabilities.detach().cpu().numpy()
            else:
                batch_probs = np.array([])

            # Reconstruct full batch with zeros for failed images
            full_batch_probs = []
            valid_idx = 0
            for j in range(len(batch_tensors)):
                if batch_tensors[j] is None:
                    # Failed image - use zeros
                    full_batch_probs.append(np.zeros(27))  # Assuming 27 classes
                else:
                    # Valid image - use prediction
                    full_batch_probs.append(batch_probs[valid_idx])
                    valid_idx += 1

            all_probs.extend(full_batch_probs)

        return np.array(all_probs)

if __name__ == "__main__":
    MODEL_TYPE = 'resnet50'

    parser = argparse.ArgumentParser(description='Train CNN models for Rakuten dataset (V2)')
    parser.add_argument('--model_folder', type=str, required=True, help='Path to the folder where the model and the mapping file are found')
    parser.add_argument('--model_name', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--file_name', type=str, required=True, help='Path to the trained model')

    args = parser.parse_args()

    model_folder = args.model_folder
    model_name = args.model_name
    model_path = os.path.join(model_folder, model_name)
    json_path = os.path.join(model_folder, 'classes.json')
    class_names = []
    with open(json_path, 'r') as f:
        class_names = json.load(f)


    ic = ImageClassifier(MODEL_TYPE, model_path)
    probs = ic.predict_image(args.file_name)
    class_id = probs.argmax().item()
    # print()
    class_name = class_names[probs.argmax().item()]
    print('Class ', class_name)
