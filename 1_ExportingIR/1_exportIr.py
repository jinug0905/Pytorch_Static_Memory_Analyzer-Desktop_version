import argparse
from pathlib import Path
import torch
import torchvision.models as models

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50", choices=["alexnet", "mobilenet_v3_small", "resnet50", "resnet101", "vgg16"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--h", type=int, default=224)
    ap.add_argument("--w", type=int, default=224)
    ap.add_argument("--out_dir", default="../data")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    m.eval().cpu()

    x = torch.randn(args.batch, 3, args.h, args.w, dtype=torch.float32)

    with torch.no_grad():
        ts = torch.jit.trace(m, x, strict=False)

    ts_path = out_dir / f"{args.model}b{args.batch}_traced.pt"
    ir_path = out_dir / f"{args.model}b{args.batch}_traced.ir.txt"

    ts.save(str(ts_path))
    ir_path.write_text(str(ts.inlined_graph))

    print(f"Saved: {ts_path}")
    print(f"Saved IR: {ir_path}")

if __name__ == "__main__":
    main()
