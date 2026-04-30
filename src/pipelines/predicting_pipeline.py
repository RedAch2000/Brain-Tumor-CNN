import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from src.components import BrainTumorModel

class Predictor:
    def __init__(self, checkpoint_path):
        # 1. Load the model from checkpoint
        self.model = BrainTumorModel.load_from_checkpoint(checkpoint_path)
        self.model.eval()
        self.model.freeze()  # Stop gradient calculations for speed

        # 2. Define the exact same transforms used in validation
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            # If you added Normalization to your DataModule, add it here too:
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Class names (ImageFolder usually sorts them alphabetically: 'no', 'yes')
        self.classes = ["no", "yes"]

    def predict(self, image_path):
        # 3. Load and transform the image
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0) # Add batch dimension (1, 3, 128, 128)

        img_tensor = img_tensor.to(self.model.device) # Move to same device as model (CPU or GPU)
        # 4. Forward pass
        logits = self.model(img_tensor)
        
        # 5. Get probabilities and predicted class
        probs = F.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

        result = {
            "prediction": self.classes[pred_idx.item()],
            "confidence": conf.item(),
            "probabilities": {self.classes[i]: probs[0][i].item() for i in range(len(self.classes))}
        }
        
        return result

# --- Example Usage ---
if __name__ == "__main__":
    # Update this path to your best checkpoint
    CKPT_PATH = "models/checkpoints/last.ckpt" 
    IMAGE_TO_TEST = "dataset/raw/yes/Y1.jpg" # Path to an MRI image

    predictor = Predictor(CKPT_PATH)
    prediction = predictor.predict(IMAGE_TO_TEST)

    print(f"\nResult for {IMAGE_TO_TEST}:")
    print(f"Prediction: {prediction['prediction'].upper()}")
    print(f"Confidence: {prediction['confidence']*100:.2f}%")
    print(f"Full Probabilities: {prediction['probabilities']}")