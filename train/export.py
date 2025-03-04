import argparse
from train_utils import get_model
from brevitas.export import export_qonnx
import argparse
import torch
import yaml

def main():
    parser = argparse.ArgumentParser(description='FINN ONNX export')
    parser.add_argument('-config', '--config', type=str, required=True)
    parser.add_argument('-i', '--input_size', type=int, nargs=4, help='Model input size in the NCWH format', required=True)
    parser.add_argument('-o', '--output_path', type=str, help='Output path', default='out.onnx')
    parser.add_argument('-c', '--ckpt', type=str, help='Path to checkpoint')

    args = parser.parse_args()

    f=open(args.config, 'r')
    config = yaml.full_load(f)
    net = get_model(config)
    device = 'cpu'
    net = net.to(device)

    x = torch.randn(tuple(args.input_size)).to(device)

    if args.ckpt is not None:
        with open(args.ckpt, 'rb') as f:
                    checkpoint = torch.load(f, map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['model'])
    
    export_qonnx(net, x,  export_path=args.output_path)

if __name__ == '__main__':
    main()
