# Entry point for training baseline and multitask models

import argparse
import contextlib
import json
import os
import ssl
import sys

import numpy as np
import torch
import torchvision.transforms as T
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from config import config
from dataset.data import SphereDataset
from evaluation.evaluator import evaluate_model, compute_all_metrics, compare_models
from evaluation.metrics import paired_ttest
from models.baseline_model import BaselineLightingNet
from models.multitask_model import MaterialAwareLightingNet
from training.losses import MultiTaskLoss
from training.train import Trainer
from utils.teapot_compare import (
    run_teapot_comparisons, copy_sample_inputs,
    render_for_lpips, lpips_on_renders,
)
from utils.visualizer import (
    plot_envmap_comparison,
    plot_per_material_comparison,
    plot_per_param_bucket_comparison,
    plot_per_param_mae,
    plot_log_mse_distribution,
    plot_training_curves,
)

# torchvision downloads ImageNet weights over HTTPS; skip cert verification
# for machines with a broken CA bundle.
ssl._create_default_https_context = ssl._create_unverified_context


def _filter_results(results, mask):
    """Return a new results dict containing only samples where mask is True."""
    mask = np.asarray(mask, dtype=bool)
    out = {}
    for k, v in results.items():
        if v is None:
            out[k] = None
        else:
            out[k] = v[mask]
    return out


def _shiny_mask(results, material_param_names,
                rough_max=0.2, metallic_min=0.8):
    p = np.asarray(results["target_material_params"], dtype=np.float32)
    idx = {n: i for i, n in enumerate(material_param_names)}
    return ((p[:, idx["roughness"]] < rough_max)
            & (p[:, idx["metallic"]] > metallic_min))


def make_dataloaders(config, transform):
    train_ds = SphereDataset('train', config, transform=transform)
    val_ds = SphereDataset('val', config, transform=transform)
    test_ds = SphereDataset('test', config, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=(str(config.device) == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=(str(config.device) == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=(str(config.device) == "cuda"),
    )
    print(f"Dataset splits: train={len(train_ds)}, val={len(val_ds)}, "
          f"test={len(test_ds)}")
    return train_loader, val_loader, test_loader


def train_model(config, is_multitask, train_loader, val_loader):
    tag = "multitask" if is_multitask else "baseline"
    print(f"\n{'='*60}")
    print(f"Training {tag} model ({config.backbone})")
    print(f"{'='*60}\n")

    if is_multitask:
        model = MaterialAwareLightingNet(config).to(config.device)
        criterion = MultiTaskLoss(config)
    else:
        model = BaselineLightingNet(config).to(config.device)
        criterion = None  # baseline uses plain MSE in Trainer

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    trainer = Trainer(model, criterion, optimizer, scheduler, config,
                      is_multitask=is_multitask)
    history = trainer.fit(train_loader, val_loader)

    # Save training curves
    curves_path = os.path.join(config.save_dir,
                               f"{config.experiment_name}_{tag}_curves.png")
    plot_training_curves(history, curves_path, title=f"{tag.title()} Training Curves")
    print(f"  Training curves saved to {curves_path}")

    return model


class _Tee:
    """File-like object that mirrors writes to two streams (stdout + a log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


@contextlib.contextmanager
def _tee_stdout(log_path):
    """Mirror everything printed inside the block to `log_path`."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log_file = open(log_path, "w")
    original = sys.stdout
    sys.stdout = _Tee(original, log_file)
    try:
        yield
    finally:
        sys.stdout = original
        log_file.close()


def _load_checkpoint(model, config, tag):
    """Load <experiment>_<tag>_best.pt into `model` and move it to device."""
    path = os.path.join(config.save_dir,
                        f"{config.experiment_name}_{tag}_best.pt")
    print(f"Loading {tag} checkpoint from {path}")
    ckpt = torch.load(path, map_location=config.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(config.device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Train lighting estimation models")
    parser.add_argument('--mode', choices=['baseline', 'multitask', 'both'],
                        default='both', help='Which model(s) to train')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override num_epochs from config')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch_size from config')
    parser.add_argument('--backbone', type=str, default=None,
                        help='Override backbone from config')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training; load saved checkpoints and run '
                             'evaluation + visualizations only.')
    parser.add_argument('--no-teapot-lpips', action='store_true',
                        help='Skip the render-LPIPS pass on the shiny subset '
                             '(the slow Blender block).')
    parser.add_argument('--no-teapot-showcase', action='store_true',
                        help='Skip the final teapot-render showcase strips '
                             '(the visualization-only renders at the end).')
    args = parser.parse_args()

    # Apply overrides
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.backbone is not None:
        config.backbone = args.backbone
    config.skip_teapot_lpips = bool(args.no_teapot_lpips)
    config.skip_teapot_showcase = bool(args.no_teapot_showcase)

    # Device
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {config.device}")

    # Seed
    torch.manual_seed(config.seed)

    # Transforms
    transform = T.Compose([
        T.Resize((config.image_size, config.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    train_loader, val_loader, test_loader = make_dataloaders(config, transform)

    baseline_model = None
    multitask_model = None

    if args.eval_only:
        if args.mode in ('baseline', 'both'):
            baseline_model = _load_checkpoint(BaselineLightingNet(config), config,
                                              tag='baseline')
        if args.mode in ('multitask', 'both'):
            multitask_model = _load_checkpoint(MaterialAwareLightingNet(config),
                                               config, tag='multitask')
    else:
        if args.mode in ('baseline', 'both'):
            baseline_model = train_model(config, is_multitask=False,
                                         train_loader=train_loader, val_loader=val_loader)

        if args.mode in ('multitask', 'both'):
            multitask_model = train_model(config, is_multitask=True,
                                          train_loader=train_loader, val_loader=val_loader)

    # --- Evaluation on test set (mirrored to <experiment>_eval_log.txt) ---
    eval_log_path = os.path.join(config.save_dir,
                                 f"{config.experiment_name}_eval_log.txt")

    with _tee_stdout(eval_log_path):
        run_evaluation(config, test_loader, baseline_model, multitask_model)
    print(f"Eval log saved to {eval_log_path}")


def run_evaluation(config, test_loader, baseline_model, multitask_model):
    """Run the full eval + reporting + visualization + JSON-save pipeline.

    Lives in its own function so main() can wrap the entire stdout stream
    with `_tee_stdout(...)` in one line.
    """
    print(f"\n{'='*60}")
    print("Evaluating on test set")
    print(f"{'='*60}\n")

    baseline_metrics = None
    multitask_metrics = None

    baseline_results = None
    multitask_results = None

    if baseline_model is not None:
        baseline_results = evaluate_model(baseline_model, test_loader, config.device,
                                          is_multitask=False)
        baseline_metrics = compute_all_metrics(
            baseline_results,
            material_param_names=config.material_param_names,
            lpips_device=config.device,
        )

    if multitask_model is not None:
        multitask_results = evaluate_model(multitask_model, test_loader, config.device,
                                           is_multitask=True)
        multitask_metrics = compute_all_metrics(
            multitask_results,
            material_param_names=config.material_param_names,
            lpips_device=config.device,
        )

    # Side-by-side comparison (only when both were trained)
    if baseline_metrics is not None and multitask_metrics is not None:
        compare_models(baseline_metrics, multitask_metrics,
                       material_param_names=config.material_param_names)

    # --- Shiny-subset evaluation: roughness < 0.2 AND metallic > 0.8 ---
    # Tests the project hypothesis on the materials where envmap detail
    # matters most (sharp specular reflections).
    if (baseline_results is not None and multitask_results is not None):
        mask = _shiny_mask(multitask_results, config.material_param_names,
                           rough_max=0.2, metallic_min=0.8)
        n_shiny = int(mask.sum())
        print(f"\n{'='*60}")
        print(f"Shiny subset (roughness < 0.2 AND metallic > 0.8): "
              f"n = {n_shiny} / {len(mask)}")
        print(f"{'='*60}\n")
        if n_shiny < 2:
            print("[shiny subset too small to compare — skipping]")
        else:
            b_sub = compute_all_metrics(
                _filter_results(baseline_results, mask),
                material_param_names=config.material_param_names,
                lpips_device=config.device,
            )
            m_sub = compute_all_metrics(
                _filter_results(multitask_results, mask),
                material_param_names=config.material_param_names,
                lpips_device=config.device,
            )
            compare_models(b_sub, m_sub,
                           material_param_names=config.material_param_names)

            if getattr(config, 'skip_teapot_lpips', False):
                print("\n[render-LPIPS skipped via --no-teapot-lpips]")
            else:
                # ---- Rendered-teapot LPIPS on the full shiny subset ----
                # Renderer caches each PNG, so this is a one-time cost —
                # subsequent runs become deterministic and near-instant.
                shiny_idxs = np.where(mask)[0]
                picked = np.sort(shiny_idxs).tolist()
                n_render = len(picked)

                print(f"\n{'='*60}")
                print(f"Rendered-teapot LPIPS on {n_render} shiny samples "
                      f"(transparent envmap bg)")
                print(f"{'='*60}\n")
                render_dir = os.path.join("result", config.experiment_name,
                                          "teapot_lpips")
                records = render_for_lpips(
                    config, baseline_results, multitask_results,
                    subset_indices=picked, out_dir=render_dir,
                    render_resolution=256, render_samples=16,
                )
                if records:
                    pair = lpips_on_renders(records, device=str(config.device))
                    if pair is not None:
                        b_lp, m_lp = pair
                        tt = paired_ttest(b_lp, m_lp)
                        sign = ("multitask better" if tt["mean_diff"] > 0
                                else "baseline better")
                        print()
                        print(f"  baseline render-LPIPS  mean = "
                              f"{np.mean(b_lp):.4f}")
                        print(f"  multitask render-LPIPS mean = "
                              f"{np.mean(m_lp):.4f}")
                        print(f"  paired t-test (baseline - multitask)")
                        print(f"    n            = {tt['n']}")
                        print(f"    mean diff    = {tt['mean_diff']:+.4f} "
                              f"(95% CI ± {tt['ci95_half_diff']:.4f}) — {sign}")
                        print(f"    t-statistic  = {tt['t_stat']:+.4f}")
                        print(f"    p-value      = {tt['p_value']:.4g}")

    # --- Visualizations ---
    vis_dir = os.path.join("result", config.experiment_name)

    # Per-material comparison chart (discrete categories)
    if baseline_metrics is not None and multitask_metrics is not None:
        chart_path = os.path.join(vis_dir, "per_material_comparison.png")
        plot_per_material_comparison(baseline_metrics, multitask_metrics, chart_path)
        print(f"Per-material chart saved to {chart_path}")

        # Per-attribute (continuous BSDF property) bucket chart
        bucket_path = os.path.join(vis_dir, "per_param_bucket_comparison.png")
        plot_per_param_bucket_comparison(baseline_metrics, multitask_metrics,
                                         bucket_path)
        print(f"Per-attribute bucket chart saved to {bucket_path}")

        # Per-sample log_mse histogram (visualizes the paired-test spread)
        dist_path = os.path.join(vis_dir, "log_mse_distribution.png")
        plot_log_mse_distribution(baseline_metrics, multitask_metrics, dist_path)
        print(f"Per-sample distribution saved to {dist_path}")

    # Multitask material-parameter regression MAE
    if multitask_metrics is not None:
        mae_path = os.path.join(vis_dir, "material_param_mae.png")
        plot_per_param_mae(multitask_metrics, mae_path,
                           param_names=config.material_param_names)
        print(f"Per-parameter MAE chart saved to {mae_path}")

    # Pick a fresh random subset of test positions per run so the showcase
    # figures and teapot renders cycle through different samples. Same
    # positions are used for envmap-comparison PNGs, the input image copies,
    # and run_teapot_comparisons below — so sample_<i>_*.png across all three
    # paths reference the same test image.
    sample_source = multitask_results or baseline_results
    showcase_positions = []
    if sample_source is not None:
        n_vis = min(5, len(sample_source["pred_env"]))
        showcase_positions = sorted(
            np.random.default_rng().choice(
                len(sample_source["pred_env"]), n_vis, replace=False
            ).tolist()
        )
        print(f"\n[showcase] selected eval positions: {showcase_positions}")

    # Envmap GT vs predicted comparisons
    for tag, raw_results in [("baseline", baseline_results),
                              ("multitask", multitask_results)]:
        if raw_results is None:
            continue
        for i, pos in enumerate(showcase_positions):
            pred = raw_results["pred_env"][pos]
            target = raw_results["target_env"][pos]
            env_path = os.path.join(vis_dir, f"{tag}_envmap_{i}.png")
            plot_envmap_comparison(pred, target, env_path,
                                   title=f"{tag.title()} Sample {i}")

    # Drop the source sphere image alongside the envmap comparison figures.
    if sample_source is not None and "target_indices" in sample_source and showcase_positions:
        metadata_path = os.path.join(config.metadata_root, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                _metadata = json.load(f)
            picked = [int(sample_source["target_indices"][p])
                      for p in showcase_positions]
            copy_sample_inputs(config.images_root, _metadata,
                               picked, vis_dir, len(picked))

    # Save metrics to JSON. Drop per-sample arrays (they're large and only
    # needed in-memory for the paired t-test, which we persist separately).
    os.makedirs(config.save_dir, exist_ok=True)
    results_path = os.path.join(config.save_dir,
                                f"{config.experiment_name}_results.json")

    def _strip_per_sample(m):
        if m is None:
            return None
        out = {k: v for k, v in m.items() if k != "per_sample"}
        return out

    saved = {}
    if baseline_metrics is not None:
        saved["baseline"] = _strip_per_sample(baseline_metrics)
    if multitask_metrics is not None:
        saved["multitask"] = _strip_per_sample(multitask_metrics)

    # Persist the paired t-test on per-sample angular errors
    if baseline_metrics is not None and multitask_metrics is not None:
        b_log = baseline_metrics.get("per_sample", {}).get("log_mse")
        m_log = multitask_metrics.get("per_sample", {}).get("log_mse")
        if b_log is not None and m_log is not None and len(b_log) == len(m_log):
            saved["paired_ttest_log_mse"] = paired_ttest(b_log, m_log)

    with open(results_path, 'w') as f:
        json.dump(saved, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if getattr(config, 'skip_teapot_showcase', False):
        print("\n[teapot showcase renders skipped via --no-teapot-showcase]")
        return

    # Utah teapot insertion comparison renders on Blender
    print(f"\n{'='*60}")
    print("Teapot insertion comparison renders")
    print(f"{'='*60}\n")
    run_teapot_comparisons(
        config,
        baseline_results=baseline_results,
        multitask_results=multitask_results,
        vis_dir=vis_dir,
        n_samples=getattr(config, "teapot_n_samples", 5),
        render_resolution=getattr(config, "teapot_resolution", 512),
        render_samples=getattr(config, "teapot_samples", 64),
        positions=showcase_positions or None,
    )


if __name__ == "__main__":
    main()
