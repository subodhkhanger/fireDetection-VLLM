#!/bin/bash

# Quick debug script to diagnose zero metrics issue

echo "Fire-ViT Pipeline Debug"
echo "======================="
echo ""

# Activate virtual environment if it exists
if [ -d "fire_vit_env" ]; then
    echo "Activating virtual environment..."
    source fire_vit_env/bin/activate
fi

echo "Running comprehensive debug..."
echo ""

python3 debug_pipeline.py \
    --checkpoint experiments/quick_test/checkpoints/best_model.pth \
    --annotation data/fire_detection/val/_annotations.coco.json \
    --device mps

echo ""
echo "Debug complete!"
