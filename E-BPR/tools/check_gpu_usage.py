#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查GPU使用情况的诊断脚本
"""
import torch
import sys

def check_cuda():
    """检查CUDA是否可用"""
    print("=" * 60)
    print("CUDA Availability Check")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  Current memory allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Current memory cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    else:
        print("\n⚠️  WARNING: CUDA is not available!")
        print("   The model will run on CPU, which is very slow.")
        return False
    
    return True

def test_gpu_tensor():
    """测试GPU tensor操作"""
    print("\n" + "=" * 60)
    print("GPU Tensor Test")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test")
        return False
    
    try:
        # 创建测试tensor
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        
        print("✓ GPU tensor operations work correctly")
        print(f"  Test tensor device: {x.device}")
        print(f"  Result tensor device: {z.device}")
        
        # 清理
        del x, y, z
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"✗ GPU tensor test failed: {e}")
        return False

def check_model_device(model):
    """检查模型参数所在的设备"""
    print("\n" + "=" * 60)
    print("Model Device Check")
    print("=" * 60)
    
    if model is None:
        print("Model is None")
        return
    
    # 检查第一个参数的设备
    try:
        first_param = next(model.parameters())
        device = first_param.device
        print(f"Model device: {device}")
        
        if device.type == 'cuda':
            print(f"✓ Model is on GPU: {device}")
        else:
            print(f"⚠️  Model is on CPU: {device}")
            print("   This will be very slow!")
    except Exception as e:
        print(f"Error checking model device: {e}")

if __name__ == '__main__':
    print("\nGPU Usage Diagnostic Tool\n")
    
    # 检查CUDA
    cuda_available = check_cuda()
    
    # 测试GPU
    if cuda_available:
        test_gpu_tensor()
    
    # 检查环境变量
    print("\n" + "=" * 60)
    print("Environment Variables")
    print("=" * 60)
    import os
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    if not cuda_available:
        print("\n" + "=" * 60)
        print("Troubleshooting")
        print("=" * 60)
        print("1. Check if NVIDIA drivers are installed:")
        print("   nvidia-smi")
        print("\n2. Check if PyTorch was installed with CUDA support:")
        print("   python -c 'import torch; print(torch.cuda.is_available())'")
        print("\n3. Reinstall PyTorch with CUDA if needed:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    sys.exit(0 if cuda_available else 1)

