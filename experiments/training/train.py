r"""Train AURORA v4 packed-Bayer -> camera-rendered-sRGB model on SIDD-CC.

Example (PowerShell, from repository root):
  python -m experiments.training.train --data-root "..\AURORA-ISP\datasets\SIDD-CC" --epochs 30 --patch-size 128 --batch-size 2 --amp

The split is by capture-instance directory before patch extraction. Validation
crops are deterministic. Checkpoints are saved atomically and include the
model/configuration and training provenance. Metrics are validation-patch
metrics, not official benchmark/test-set results.
"""
import argparse, json, random, time, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from datasets.loader import PairedRawDataset
from models.learned_isp.unet import AuroraUNet, count_params



class CaptureBatchSampler(Sampler):
    """Shuffle capture order while keeping each capture's patches adjacent.

    This makes the dataset's one-capture RAM cache effective without loading
    the complete 32.6 GB dataset into memory.
    """
    def __init__(self, dataset, batch_size, shuffle=True, seed=42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        capture_ids = list(range(len(self.dataset.records)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(capture_ids)
        for capture_id in capture_ids:
            start = capture_id * self.dataset.patches_per_capture
            indices = list(range(start, start + self.dataset.patches_per_capture))
            if self.shuffle:
                rng.shuffle(indices)
            for offset in range(0, len(indices), self.batch_size):
                yield indices[offset:offset + self.batch_size]

    def __len__(self):
        import math
        return len(self.dataset.records) * math.ceil(self.dataset.patches_per_capture / self.batch_size)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def psnr(pred, target):
    mse = F.mse_loss(pred.float(), target.float()).item()
    return float('inf') if mse == 0 else 10.0 * np.log10(1.0 / mse)


def save_checkpoint(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp); os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', required=True, help='SIDD-CC root containing Raw/ and SRGB/')
    ap.add_argument('--output', default='models/checkpoints/aurora_v4_best.pt')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patch-size', type=int, default=128)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--patches-per-capture', type=int, default=8)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--base', type=int, default=24)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--resume', default=None, help='Optional checkpoint to resume')
    args = ap.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patches_per_capture < 1:
        ap.error('epochs, batch-size and patches-per-capture must be positive')
    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = PairedRawDataset(args.data_root, 'train', args.patch_size, seed=args.seed,
                                patches_per_capture=args.patches_per_capture)
    val_ds = PairedRawDataset(args.data_root, 'val', args.patch_size, seed=args.seed,
                              patches_per_capture=max(1, args.patches_per_capture // 2))
    train_sampler = CaptureBatchSampler(train_ds, args.batch_size, shuffle=True, seed=args.seed)
    val_sampler = CaptureBatchSampler(val_ds, args.batch_size, shuffle=False, seed=args.seed)
    train_dl = DataLoader(train_ds, batch_sampler=train_sampler,
                          num_workers=args.workers, pin_memory=device.type == 'cuda')
    val_dl = DataLoader(val_ds, batch_sampler=val_sampler,
                        num_workers=args.workers, pin_memory=device.type == 'cuda')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
    model = AuroraUNet(base=args.base, meta_dim=5, use_meta=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled = bool(args.amp and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        if 'optimizer_state' in ckpt: optimizer.load_state_dict(ckpt['optimizer_state'])
        if 'scaler_state' in ckpt and amp_enabled: scaler.load_state_dict(ckpt['scaler_state'])
        start_epoch = int(ckpt.get('epoch', 0)) + 1
    out = Path(args.output); best_path = out
    best = float('inf')
    print(json.dumps({'release':'AURORA ISP v4.0.2','device':str(device),
        'cuda_device':torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        'amp':amp_enabled,'train_instances':len({r[0] for r in train_ds.records}),
        'val_instances':len({r[0] for r in val_ds.records}),
        'train_captures':len(train_ds.records),'val_captures':len(val_ds.records),
        'train_patches_epoch':len(train_ds),'val_patches_epoch':len(val_ds),
        'parameters':count_params(model),'patch_size':args.patch_size,
        'batch_size':args.batch_size,'seed':args.seed}, indent=2), flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        t0=time.perf_counter(); model.train(); train_losses=[]
        epoch_batches = len(train_dl)
        step_times = []
        print(f'EPOCH_START {epoch}/{args.epochs} batches={epoch_batches}', flush=True)
        for batch_idx, batch in enumerate(train_dl, 1):
            step_t0 = time.perf_counter()
            raw=batch['raw'].to(device,non_blocking=True)
            meta=batch['metadata'].to(device,non_blocking=True)
            target=batch['target_srgb'].to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp_enabled):
                pred=model(raw,meta)
                loss=F.l1_loss(pred,target) + 0.2*F.mse_loss(pred,target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer); scaler.update()
            train_losses.append(float(loss.detach().cpu()))
            step_times.append(time.perf_counter() - step_t0)
            if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == epoch_batches:
                avg_step = sum(step_times[-20:]) / len(step_times[-20:])
                eta_min = avg_step * (epoch_batches - batch_idx) / 60.0
                print(f' train {batch_idx}/{epoch_batches} loss={train_losses[-1]:.5f} step={avg_step:.2f}s epoch_remaining~{eta_min:.1f}min', flush=True)
        model.eval(); val_losses=[]; val_psnr=[]
        print(f'EPOCH_VALIDATION_START {epoch}/{args.epochs} batches={len(val_dl)}', flush=True)
        with torch.inference_mode():
            for val_idx, batch in enumerate(val_dl, 1):
                raw=batch['raw'].to(device,non_blocking=True)
                meta=batch['metadata'].to(device,non_blocking=True)
                target=batch['target_srgb'].to(device,non_blocking=True)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp_enabled):
                    pred=model(raw,meta)
                val_losses.append(float(F.l1_loss(pred.float(),target.float()).cpu()))
                val_psnr.append(psnr(pred,target))
                if val_idx == 1 or val_idx % 20 == 0 or val_idx == len(val_dl):
                    print(f' val {val_idx}/{len(val_dl)}', flush=True)
        scheduler.step()
        record={'epoch':epoch,'train_loss':float(np.mean(train_losses)),
            'val_l1':float(np.mean(val_losses)),'val_psnr_db':float(np.mean(val_psnr)),
            'lr':optimizer.param_groups[0]['lr'],'epoch_seconds':time.perf_counter()-t0}
        print(json.dumps(record),flush=True)
        payload={'release':'AURORA ISP v4.0.2','model_state':model.state_dict(),
            'model_config':{'architecture':'AuroraUNet','base':args.base,'meta_dim':5,'use_meta':True,
                            'input':'canonical packed Bayer RGGB,Gr,Gb,B','output':'camera-rendered sRGB'},
            'epoch':epoch,'val_l1':record['val_l1'],'val_psnr_db':record['val_psnr_db'],
            'dataset':'SIDD-CC','target':'camera-rendered sRGB PNG','seed':args.seed,
            'args':vars(args),'optimizer_state':optimizer.state_dict(),'scaler_state':scaler.state_dict()}
        save_checkpoint(out.with_name(out.stem+'_latest'+out.suffix),payload)
        if record['val_l1'] < best:
            best=record['val_l1']; save_checkpoint(best_path,payload)
            print(f'BEST_CHECKPOINT={best_path.resolve()}',flush=True)
    print(f'TRAINING_COMPLETE best_val_l1={best:.7f} checkpoint={best_path.resolve()}',flush=True)

if __name__=='__main__': main()
