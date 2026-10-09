"""3D Swin Transformer classifier using MONAI's SwinUNETR encoder."""
from __future__ import annotations
import json, os, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from monai.networks.nets import SwinUNETR
from Options.Options_Swin3D import parse_swin_options
from train_ResNet50_3D import MultimodalClassificationDataset
from classification_sampling import ClassBalancedSampler

class SwinClassifier(nn.Module):
    def __init__(self, num_classes=4, feature_size=24):
        super().__init__()
        swin = SwinUNETR(
            in_channels=3,
            out_channels=num_classes,
            feature_size=feature_size,
            spatial_dims=3,
            use_checkpoint=False,
        )
        self.swin = swin.swinViT
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(feature_size * 16, num_classes)

    def forward(self, x):
        features = self.swin(x, normalize=True)
        return self.head(self.pool(features[-1]).flatten(1))

def compute_metrics(ys, probs, classes):
    probs = np.asarray(probs)
    ps = probs.argmax(axis=1)
    precision, recall, _, _ = precision_recall_fscore_support(
        ys, ps, labels=list(range(classes)), average="macro", zero_division=0
    )
    try:
        auc = roc_auc_score(
            ys, probs, labels=list(range(classes)), multi_class="ovr", average="macro"
        )
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(ys, ps)),
        "precision": float(precision),
        "recall": float(recall),
        "auc": float(auc),
    }


def evaluate(model, loader, device, criterion, classes):
    model.eval(); ys=[]; probs=[]; loss=0.0
    with torch.no_grad():
        for x,y,_ in loader:
            logits=model(x.to(device)); loss += criterion(logits,y.to(device)).item()
            probs.extend(torch.softmax(logits,1).cpu().tolist()); ys.extend(y.tolist())
    metrics = compute_metrics(ys, probs, classes)
    metrics["loss"] = loss / max(len(loader), 1)
    return metrics


def rounded_metric(value):
    return round(float(value), 4) if np.isfinite(value) else None


def plot_metric(history, metric, output_dir):
    plt.figure(figsize=(9, 5))
    plt.plot(history["epoch"], history[f"train_{metric}"], "r-", label="train")
    plt.plot(history["epoch"], history[f"val_{metric}"], "b-", label="val")
    plt.xlabel("Epoch"); plt.ylabel(metric.upper()); plt.title(f"{metric.upper()} curve")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(output_dir / f"{metric}.png", dpi=150)
    plt.close()

def main():
    a=parse_swin_options(); os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu_id) if a.gpu_id>=0 else ''
    device=torch.device('cuda' if a.gpu_id>=0 and torch.cuda.is_available() else 'cpu'); paths={'PICAI':Path('nnUNet/nnUNet_raw/Dataset2302_FullPI-CAI'),'AHCDU':Path('nnUNet/nnUNet_raw/Dataset2301_FullAHCDU')}
    tr=MultimodalClassificationDataset(paths[a.dataset],'train',augment=True); va=MultimodalClassificationDataset(paths[a.dataset],'val',augment=False)
    if a.sample:
        sampler=ClassBalancedSampler([value for _, value, _ in tr.records],a.dataset)
        print('training inverse-frequency sampler counts=',sampler.class_counts,'samples_per_epoch=',len(sampler))
        tl=DataLoader(tr,a.batch_size,sampler=sampler,num_workers=a.num_workers)
    else:
        print('training sampler=disabled; using original class distribution')
        tl=DataLoader(tr,a.batch_size,shuffle=True,num_workers=a.num_workers)
    vl=DataLoader(va,a.batch_size,shuffle=False,num_workers=a.num_workers)
    model=SwinClassifier(a.num_classes).to(device); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay); crit=nn.CrossEntropyLoss(); out=Path('Checkpoint')/a.task_name; out.mkdir(parents=True,exist_ok=True); start=1
    history={"epoch":[],"train_loss":[],"val_loss":[]}
    for metric in ("accuracy","precision","recall","auc"):
        history[f"train_{metric}"]=[]; history[f"val_{metric}"]=[]
    best=-1.0
    if a.resume:
        latest=out/'latest.pt'
        if not latest.exists(): raise FileNotFoundError(f'--resume requested but checkpoint does not exist: {latest}')
        c=torch.load(latest,map_location=device); model.load_state_dict(c['model']); opt.load_state_dict(c['optimizer']); start=c['epoch']+1
        history=c.get('history',history)
        best=float(c.get('best_auc',-1.0))
        if best < 0 and (out/'best.pt').exists():
            best_c=torch.load(out/'best.pt',map_location='cpu'); best=float(best_c.get('metrics',{}).get('auc',-1.0))
        print('resuming epoch',start)
    (out/'options.json').write_text(json.dumps(vars(a),default=str,indent=2)+'\n')
    for epoch in range(start,a.epochs+1):
        model.train(); total=0; train_ys=[]; train_probs=[]
        for x,y,_ in tqdm(tl,desc=f'Epoch {epoch}/{a.epochs}'):
            opt.zero_grad(); logits=model(x.to(device)); loss=crit(logits,y.to(device)); loss.backward(); opt.step(); total+=loss.item()
            train_probs.extend(torch.softmax(logits.detach(),1).cpu().tolist()); train_ys.extend(y.tolist())
        train_m=compute_metrics(train_ys,train_probs,a.num_classes); train_m['loss']=total/max(len(tl),1)
        val_m=evaluate(model,vl,device,crit,a.num_classes)
        history['epoch'].append(epoch)
        for metric in ('loss','accuracy','precision','recall','auc'):
            history[f'train_{metric}'].append(train_m[metric]); history[f'val_{metric}'].append(val_m[metric])
            plot_metric(history,metric,out)
        improved=np.isfinite(val_m['auc']) and val_m['auc']>best
        if improved: best=val_m['auc']
        c={'epoch':epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),'metrics':val_m,
           'train_metrics':train_m,'best_auc':best,'history':history,'options':vars(a)}
        torch.save(c,out/'latest.pt')
        row={'epoch':epoch,**{f'train_{k}':rounded_metric(v) for k,v in train_m.items()},
             **{f'val_{k}':rounded_metric(v) for k,v in val_m.items()}}
        with (out/'metrics.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
        print({"train":train_m,"val":val_m})
        if improved: torch.save(c,out/'best.pt'); print('new best macro-AUC=',best)

if __name__=='__main__': main()
