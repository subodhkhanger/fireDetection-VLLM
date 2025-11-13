"""
Inference script for Fire-ViT

Run fire detection on images or videos
"""

import argparse
import cv2
import torch
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

from models.fire_vit import build_fire_vit
from utils.checkpoint import load_checkpoint
from utils.visualization import visualize_predictions
from utils.postprocess import decode_predictions


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fire-ViT Inference')

    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image, video, or directory')
    parser.add_argument('--output', type=str, default='./output',
                       help='Output directory for results')

    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--nms-threshold', type=float, default=0.5,
                       help='IoU threshold for NMS')
    parser.add_argument('--max-detections', type=int, default=300,
                       help='Maximum detections per image after NMS')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'mps'],
                       help='Device to use for inference')

    parser.add_argument('--save-vis', action='store_true',
                       help='Save visualization')
    parser.add_argument('--show', action='store_true',
                       help='Display results')

    return parser.parse_args()


class FireDetector:
    """Fire detection inference wrapper"""

    def __init__(
        self,
        config_path,
        checkpoint_path,
        device='cuda',
        conf_threshold=0.5,
        nms_threshold=0.5,
        max_detections=300
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Build model
        print("Loading model...")
        self.model = build_fire_vit(self.config)

        # Load checkpoint
        load_checkpoint(checkpoint_path, self.model, device=self.device)

        self.model.eval()
        self.model.to(self.device)

        # Get input size
        self.img_size = self.config['model']['input_size'][0]

        # Normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        print(f"✓ Model loaded on {self.device}")

    def preprocess(self, image):
        """
        Preprocess image for inference

        Args:
            image: numpy array (H, W, 3) in BGR

        Returns:
            tensor: Preprocessed tensor (1, 3, H, W)
            orig_size: Original image size (H, W)
        """
        orig_h, orig_w = image.shape[:2]

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))

        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std

        # To tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image = image.unsqueeze(0)  # Add batch dimension

        return image, (orig_h, orig_w)

    def postprocess(self, predictions, orig_size):
        """
        Convert raw model outputs to scaled detections.
        """
        decoded = decode_predictions(
            predictions,
            img_size=self.img_size,
            conf_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            max_detections=self.max_detections
        )[0]

        orig_h, orig_w = orig_size
        scale_x = orig_w / self.img_size
        scale_y = orig_h / self.img_size

        boxes = decoded['boxes'].clone()
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        return {
            'boxes': boxes,
            'labels': decoded['labels'],
            'scores': decoded['scores']
        }

    @torch.no_grad()
    def detect(self, image):
        """
        Run detection on single image

        Args:
            image: numpy array (H, W, 3) in BGR

        Returns:
            predictions: Detection results
        """
        # Preprocess
        input_tensor, orig_size = self.preprocess(image)
        input_tensor = input_tensor.to(self.device)

        # Inference
        predictions = self.model(input_tensor)

        # Postprocess
        result = self.postprocess(predictions, orig_size)

        return result


def process_image(detector, image_path, output_dir, save_vis=True, show=False):
    """Process single image"""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to read image: {image_path}")
        return

    # Detect
    predictions = detector.detect(image)

    print(f"Detected {len(predictions['boxes'])} objects")

    # Visualize
    if save_vis or show:
        vis_image = visualize_predictions(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            predictions,
            conf_threshold=detector.conf_threshold
        )
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

        if save_vis:
            output_path = output_dir / f"{Path(image_path).stem}_result.jpg"
            cv2.imwrite(str(output_path), vis_image)
            print(f"✓ Saved: {output_path}")

        if show:
            cv2.imshow('Detection Result', vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def process_video(detector, video_path, output_dir, save_vis=True):
    """Process video file"""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")

    # Setup output video
    if save_vis:
        output_path = output_dir / f"{Path(video_path).stem}_result.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Process frames
    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        # Detect
        predictions = detector.detect(frame)

        # Visualize
        if save_vis:
            vis_frame = visualize_predictions(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                predictions,
                conf_threshold=detector.conf_threshold
            )
            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
            out.write(vis_frame)

    cap.release()
    if save_vis:
        out.release()
        print(f"✓ Saved: {output_path}")


def main():
    """Main inference function"""
    args = parse_args()

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create detector
    detector = FireDetector(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        max_detections=args.max_detections
    )

    input_path = Path(args.input)

    # Process input
    if input_path.is_file():
        # Check if image or video
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        video_exts = {'.mp4', '.avi', '.mov', '.mkv'}

        if input_path.suffix.lower() in image_exts:
            print(f"Processing image: {input_path}")
            process_image(detector, input_path, output_dir, args.save_vis, args.show)

        elif input_path.suffix.lower() in video_exts:
            print(f"Processing video: {input_path}")
            process_video(detector, input_path, output_dir, args.save_vis)

        else:
            print(f"Unknown file type: {input_path.suffix}")

    elif input_path.is_dir():
        # Process all images in directory
        print(f"Processing directory: {input_path}")

        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(list(input_path.glob(f'*{ext}')))
            image_files.extend(list(input_path.glob(f'*{ext.upper()}')))

        print(f"Found {len(image_files)} images")

        for img_path in tqdm(image_files, desc="Processing images"):
            process_image(detector, img_path, output_dir, args.save_vis, False)

    else:
        print(f"Input not found: {input_path}")


if __name__ == '__main__':
    main()
