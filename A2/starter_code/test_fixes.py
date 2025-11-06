#!/usr/bin/env python3
"""
Test script to verify the fixes for the Linux server deployment
"""

import torch
import torch.nn as nn
from scene_classification import HardwareDetector, MyConv

def test_hardware_detection():
    """Test hardware detection"""
    print("Testing hardware detection...")
    detector = HardwareDetector()
    config = detector.get_optimal_config()
    print("✅ Hardware detection working")
    return config

def test_model_creation(config):
    """Test model creation and device compatibility"""
    print("Testing model creation...")
    
    device = torch.device(config['device'])
    model = MyConv(num_classes=100)
    model = model.to(device)
    
    # Test DataParallel setup (only if multiple GPUs available)
    if config['use_ddp'] and torch.cuda.device_count() > 1:
        if not isinstance(model, nn.DataParallel):
            model = nn.DataParallel(model)
            print(f"✅ DataParallel setup with {torch.cuda.device_count()} GPUs")
    
    # Test mixed precision setup
    scaler = None
    if config['use_mixed_precision'] and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
        print("✅ Mixed precision setup")
    
    return model, scaler, device

def test_forward_pass(model, scaler, device, config):
    """Test forward pass with dummy data"""
    print("Testing forward pass...")
    
    # Create dummy batch
    batch_size = min(config['batch_size'], 32)  # Use smaller batch for testing
    dummy_input = torch.randn(batch_size, 3, 128, 128, device=device)
    dummy_labels = torch.randint(0, 100, (batch_size,), device=device)
    
    model.train()
    
    # Test forward pass
    if scaler is not None:
        with torch.amp.autocast('cuda'):
            output = model(dummy_input)
    else:
        output = model(dummy_input)
    
    print(f"✅ Forward pass successful - Output shape: {output.shape}")
    
    # Test loss computation
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, dummy_labels)
    print(f"✅ Loss computation successful - Loss: {loss.item():.4f}")
    
    return True

def main():
    print("🔍 Running diagnostic tests for Linux server deployment...\n")
    
    try:
        # Test 1: Hardware detection
        config = test_hardware_detection()
        print()
        
        # Test 2: Model creation
        model, scaler, device = test_model_creation(config)
        print()
        
        # Test 3: Forward pass
        test_forward_pass(model, scaler, device, config)
        print()
        
        print("🎉 All tests passed! The fixes should work on the Linux server.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()