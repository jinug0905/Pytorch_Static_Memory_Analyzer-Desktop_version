import argparse
import torch
import torchvision.models as models

def mb(x): return x / (1024 ** 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50", choices=["alexnet", "mobilenet_v3_small", "resnet50", "resnet101", "vgg16"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--h", type=int, default=224)
    ap.add_argument("--w", type=int, default=224)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")

    torch.backends.cudnn.benchmark = False

    # Use torchvision
    if args.model == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)    
    elif args.model == "resnet101":
        m = models.resnet101(weights=models.ResNet101_Weights.DEFAULT) 
    elif args.model == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.DEFAULT)   
    elif args.model == "alexnet":
        m = models.alexnet(weights=models.AlexNet_Weights.DEFAULT) 
    elif args.model == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    m.eval().to(device)

    x = torch.randn(args.batch, 3, args.h, args.w, device=device)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    baseline = torch.cuda.memory_allocated(device)

    with torch.no_grad():
        _ = m(x)
        torch.cuda.synchronize()

    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    print(f"=== Runtime Peak {args.model} b{args.batch} ===")
    print(f"baseline allocated (weights+input): {mb(baseline):.2f} MB")
    print(f"peak allocated:                    {mb(peak_alloc):.2f} MB")
    print(f"peak delta (approx activ+ws):      {mb(peak_alloc-baseline):.2f} MB")
    print(f"peak reserved (caching):           {mb(peak_reserved):.2f} MB")

if __name__ == "__main__":
    main()
