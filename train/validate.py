import torch
import yaml
import torch.cuda.amp as amp
import random
import numpy as np
from train_utils import get_dataset_loaders,get_model
from precise_bn import compute_precise_bn_stats
import argparse

class ConfusionMatrix(object):
    def __init__(self, num_classes, exclude_classes):
        self.num_classes = num_classes
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        self.exclude_classes=exclude_classes

    def update(self, a, b):
        a=a.cpu()
        b=b.cpu()
        n = self.num_classes
        k = (a >= 0) & (a < n)
        inds = n * a + b
        inds=inds[k]
        self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))

        acc_global=acc_global.item() * 100
        acc=(acc * 100).tolist()
        iu=(iu * 100).tolist()
        return acc_global, acc, iu
    def __str__(self):
        acc_global, acc, iu = self.compute()
        acc_global=round(acc_global,2)
        IOU=[round(i,2) for i in iu]
        mIOU=sum(iu)/len(iu)
        mIOU=round(mIOU,2)
        reduced_iu=[iu[i] for i in range(self.num_classes) if i not in self.exclude_classes]
        mIOU_reduced=sum(reduced_iu)/len(reduced_iu)
        mIOU_reduced=round(mIOU_reduced,2)
        return f"IOU: {IOU}\nmIOU: {mIOU}, mIOU_reduced: {mIOU_reduced}, accuracy: {acc_global}"

def evaluate(model, data_loader, device, confmat,mixed_precision,print_every,max_eval):
    model.eval()
    assert(isinstance(confmat,ConfusionMatrix))
    with torch.no_grad():
        for i,(image, target) in enumerate(data_loader):
            if (i+1)%print_every==0:
                print(i+1)
            image, target = image.to(device), target.to(device)
            with amp.autocast(enabled=mixed_precision):
                output = model(image)
            output = torch.nn.functional.interpolate(output, size=target.shape[-2:], mode='bilinear', align_corners=False)
            confmat.update(target.flatten(), output.argmax(1).flatten())
            if i+1==max_eval:
                break
    return confmat

def setup_env(config):
    torch.backends.cudnn.benchmark=True
    seed=0
    if "RNG_seed" in config:
        seed=config["RNG_seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed) # might remove dependency on np later

def validate_multiple(configs):
    confmats=[]
    for config in configs:
        confmat=validate_one(config)
        confmats.append(confmat)
    return confmats
def validate_one(config):
    setup_env(config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_loader, val_loader,train_set=get_dataset_loaders(config)
    model=get_model(config).to(device)
    with open(config["pretrained_path"], 'rb') as f:
                checkpoint = torch.load(f, map_location=device)
    model.load_state_dict(checkpoint['model'])
    mixed_precision=config["mixed_precision"]
    print_every=config["eval_print_every"]
    num_classes=config["num_classes"]
    exclude_classes=config["exclude_classes"]
    confmat = ConfusionMatrix(num_classes,exclude_classes)
    max_eval=100000
    if "max_eval" in config:
        max_eval=config["max_eval"]
    loader=val_loader
    if "validate_train_loader" in config and config["validate_train_loader"]==True:
        loader=train_loader
    if config["bn_precise_stats"]:
        print("calculating precise bn stats")
        compute_precise_bn_stats(model,train_loader,config["bn_precise_num_samples"])
    print("evaluating")
    confmat = evaluate(model, loader, device,confmat,mixed_precision,
                       print_every,max_eval)
    print(confmat)
    return confmat

def validate_main(config, pretrained_path, dataset_path, batchsize):
    config_filename=config
    with open(config_filename) as file:
        config=yaml.full_load(file)
    config["dataset_dir"]=dataset_path
    config["class_uniform_pct"]=0 # since we're only evalutaing, not training
    config["pretrained_path"]=pretrained_path
    config["batch_size"]=batchsize
    confmat=validate_one(config)
    return confmat


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config', type=str, required=True)
    parser.add_argument('-p','--pretrained_path', type=str, required=True)
    parser.add_argument('-d','--dataset_path', type=str, required=True)
    parser.add_argument('-b','--batchsize', type=int, default=1)
    args = parser.parse_args()
    validate_main(args.config, args.pretrained_path, args.dataset_path, args.batchsize)
