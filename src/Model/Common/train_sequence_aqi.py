"""Train Mamba, LSTM, or Transformer AQI forecasting models from the Gold dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (str(_REPO_ROOT), str(_SRC_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.core.data_structs import AQIDataset
from src.core.utils import resolve_device, set_seed, setup_logger
from src.Model.LSTM.lstm_model import TimeSeriesLSTMRegressor
from src.Model.Mamba.mamba_model import TimeSeriesMambaRegressor
from src.Model.Common.mlflow_utils import log_artifact_dir, start_mlflow_run
from src.Model.Common.train_common_aqi import (
    _mamba_cuda_path_ready,
    _mamba_fastpath_ready,
    collect_predictions,
    evaluate,
    load_prepared_dataset_from_minio,
    persistence_baseline_metrics,
    prepare_training_targets,
    run_epoch,
    save_prediction_results,
    upload_training_outputs_to_minio,
    _cfg_bool,
    _resolve_single_province,
)
from src.Model.Transformer.transformer_model import TimeSeriesTransformerRegressor
from src.common.config import MINIO_GOLD_BUCKET, load_project_config
from src.common.minio_io import get_client, load_pickle
from src.common.time_utils import now_local


def build_model(args, num_features: int):
    if args.model_type == "mamba":
        return TimeSeriesMambaRegressor(
            num_features=num_features,
            d_model=args.d_model,
            n_layers=args.n_layers,
            horizon=args.horizon,
            dropout=args.dropout,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
        )
    if args.model_type == "lstm":
        return TimeSeriesLSTMRegressor(
            num_features=num_features,
            hidden_size=args.hidden_size,
            n_layers=args.n_layers,
            horizon=args.horizon,
            dropout=args.dropout,
        )
    if args.model_type == "transformer":
        return TimeSeriesTransformerRegressor(
            num_features=num_features,
            d_model=args.d_model,
            n_layers=args.n_layers,
            horizon=args.horizon,
            dropout=args.dropout,
            n_heads=args.n_heads,
            dim_feedforward=args.dim_feedforward,
        )
    raise ValueError(f"Unsupported model_type: {args.model_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mamba/LSTM/Transformer AQI forecasting model")
    parser.add_argument("--model-type", type=str, required=True, choices=["mamba", "lstm", "transformer"])
    parser.add_argument("--dataset-prefix", type=str, default=None, help="MinIO Gold prefix from prepare_training_dataset.py")
    parser.add_argument("--run-id", type=str, default=None, help="Training dataset run id, maps to training_dataset/run_id=<run-id>")
    parser.add_argument("--config", type=str, default=None, help="Path to project YAML config")
    parser.add_argument("--target-col", type=str, default="aqi")
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--model-run-id", type=str, default=None, help="Artifact run id on MinIO; default is inferred from --run-id")
    parser.add_argument("--keep-local", action="store_true", help="Keep temporary local artifacts after uploading to MinIO")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--loss", type=str, default="huber", choices=["mse", "huber"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--min-train-samples", type=int, default=0)
    parser.add_argument("--lr-t-max", type=int, default=None, help="T_max for CosineAnnealingLR; default is --epochs")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking for this run")
    parser.add_argument("--mlflow-experiment", type=str, default="AQI Forecasting")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--auto-cpu-for-slow-mamba", action="store_true")
    args = parser.parse_args()
    args._cli_args = set(sys.argv[1:])
    return args


def _cfg_get(cfg: dict, *keys: str, default=None):
    for key in keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return default


def _metric_float(metrics: dict, key: str) -> float:
    value = metrics.get(key, float("nan"))
    try:
        return float(value)
    except Exception:
        return float("nan")


def _cli_has(args: argparse.Namespace, option: str) -> bool:
    return option in getattr(args, "_cli_args", set())


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    project_cfg = load_project_config(args.config)
    data_cfg = project_cfg.get("data", {})
    dataset_cfg = project_cfg.get("dataset", {})
    model_cfg = project_cfg.get(args.model_type, project_cfg.get("model", {}))
    training_cfg = project_cfg.get("training", {})
    run_cfg = project_cfg.get("run", {})
    scaling_cfg = project_cfg.get("scaling", {})

    if model_cfg and not _cfg_bool(model_cfg.get("enabled", True), True):
        raise ValueError(f"Model '{args.model_type}' is disabled in config.")

    args.target_col = args.target_col or data_cfg.get("target_col", "aqi")
    args.window_size = int(_cfg_get(model_cfg, "window_size", default=dataset_cfg.get("seq_len", args.window_size)))
    args.horizon = int(_cfg_get(model_cfg, "horizon", default=dataset_cfg.get("pred_len", args.horizon)))
    args.sample_stride = int(_cfg_get(model_cfg, "sample_stride", default=dataset_cfg.get("sample_stride", args.sample_stride)))
    if not _cli_has(args, "--epochs"):
        args.epochs = int(_cfg_get(model_cfg, "epochs", default=training_cfg.get("epochs", args.epochs)))
    if not _cli_has(args, "--batch-size"):
        args.batch_size = int(_cfg_get(model_cfg, "batch_size", default=training_cfg.get("batch_size", args.batch_size)))
    if not _cli_has(args, "--lr"):
        args.lr = float(_cfg_get(model_cfg, "lr", "learning_rate", default=training_cfg.get("learning_rate", args.lr)))
    if not _cli_has(args, "--weight-decay"):
        args.weight_decay = float(_cfg_get(model_cfg, "weight_decay", default=training_cfg.get("weight_decay", args.weight_decay)))
    args.patience = int(_cfg_get(model_cfg, "patience", "early_stop_patience", default=training_cfg.get("patience", args.patience)))
    args.min_delta = float(_cfg_get(model_cfg, "min_delta", default=training_cfg.get("min_delta", args.min_delta)))
    args.min_train_samples = int(training_cfg.get("min_train_samples", args.min_train_samples))
    args.loss = str(_cfg_get(model_cfg, "loss", default=training_cfg.get("loss", args.loss)))
    args.seed = int(_cfg_get(model_cfg, "seed", default=training_cfg.get("seed", args.seed)))
    if not _cli_has(args, "--num-workers"):
        args.num_workers = int(_cfg_get(model_cfg, "num_workers", default=training_cfg.get("num_workers", args.num_workers)))
    args.amp = _cfg_bool(_cfg_get(model_cfg, "amp", "use_amp", default=training_cfg.get("amp")), args.amp)
    if not _cli_has(args, "--grad-accum-steps"):
        args.grad_accum_steps = int(_cfg_get(model_cfg, "grad_accum_steps", default=training_cfg.get("grad_accum_steps", args.grad_accum_steps)))
    if not _cli_has(args, "--max-grad-norm"):
        args.max_grad_norm = float(_cfg_get(model_cfg, "max_grad_norm", default=training_cfg.get("max_grad_norm", args.max_grad_norm)))
    lr_t_max_cfg = _cfg_get(model_cfg, "lr_t_max", "t_max", default=training_cfg.get("lr_t_max", training_cfg.get("t_max", args.lr_t_max)))
    args.lr_t_max = int(lr_t_max_cfg) if lr_t_max_cfg is not None else None
    args.min_lr = float(_cfg_get(model_cfg, "min_lr", default=training_cfg.get("min_lr", args.min_lr)))
    args.device = str(_cfg_get(model_cfg, "device", default=training_cfg.get("device", args.device)))
    args.target_mode = str(scaling_cfg.get("target_mode", project_cfg.get("target_mode", "absolute"))).lower()

    if not _cli_has(args, "--hidden-size"):
        args.hidden_size = int(_cfg_get(model_cfg, "hidden_size", "d_model", default=args.hidden_size))
    if not _cli_has(args, "--d-model"):
        args.d_model = int(_cfg_get(model_cfg, "d_model", default=args.d_model))
    if not _cli_has(args, "--n-layers"):
        args.n_layers = int(_cfg_get(model_cfg, "n_layers", "num_layers", default=args.n_layers))
    if not _cli_has(args, "--d-state"):
        args.d_state = int(_cfg_get(model_cfg, "d_state", default=args.d_state))
    if not _cli_has(args, "--d-conv"):
        args.d_conv = int(_cfg_get(model_cfg, "d_conv", default=args.d_conv))
    if not _cli_has(args, "--expand"):
        args.expand = int(_cfg_get(model_cfg, "expand", default=args.expand))
    if not _cli_has(args, "--n-heads"):
        args.n_heads = int(_cfg_get(model_cfg, "n_heads", default=args.n_heads))
    if not _cli_has(args, "--dim-feedforward"):
        dim_ff = _cfg_get(model_cfg, "dim_feedforward", "d_ff", default=args.dim_feedforward)
        args.dim_feedforward = int(dim_ff) if dim_ff is not None else None
    if not _cli_has(args, "--dropout"):
        args.dropout = float(_cfg_get(model_cfg, "dropout", default=args.dropout))
    args.mlflow_experiment = str(training_cfg.get("mlflow_experiment", args.mlflow_experiment))
    args.mlflow_tracking_uri = training_cfg.get("mlflow_tracking_uri", args.mlflow_tracking_uri)
    mlflow_enabled = _cfg_bool(
        _cfg_get(model_cfg, "mlflow_enabled", default=training_cfg.get("mlflow_enabled", True)),
        True,
    )
    args.mlflow_enabled = bool(mlflow_enabled) and not args.no_mlflow
    args.auto_cpu_for_slow_mamba = _cfg_bool(
        _cfg_get(model_cfg, "auto_cpu_for_slow_mamba", default=training_cfg.get("auto_cpu_for_slow_mamba")),
        args.auto_cpu_for_slow_mamba,
    )
    args.output_root = run_cfg.get("output_root", "runs")
    args.dataset_prefix_template = dataset_cfg.get("prefix_template", "training_dataset/run_id={run_id}")
    return args


def main() -> None:
    args = apply_config(parse_args())
    if args.run_id and not args.dataset_prefix:
        args.dataset_prefix = str(args.dataset_prefix_template).format(run_id=args.run_id)

    train_run_id = args.model_run_id or args.run_id or now_local().strftime("%Y%m%d_%H%M%S")
    cleanup_out_dir = False
    if args.out_dir is None:
        if args.keep_local:
            args.out_dir = str((_REPO_ROOT / args.output_root / args.model_type / train_run_id).resolve())
        else:
            args.out_dir = tempfile.mkdtemp(prefix=f"{args.model_type}_aqi_")
            cleanup_out_dir = True
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    logger = setup_logger(args.out_dir, name=f"train_{args.model_type}_aqi")
    set_seed(args.seed)
    logger.info("Training %s AQI model | run_id=%s", args.model_type, train_run_id)

    dataset_meta: dict = {}
    feature_cols: list[str] = []
    if not args.dataset_prefix:
        raise ValueError("Training requires a Gold dataset. Provide --run-id or --dataset-prefix from src/Gold/prepare_training_dataset.py.")

    logger.info("Loading prepared dataset from MinIO: s3://%s/%s", MINIO_GOLD_BUCKET, args.dataset_prefix)
    train, val, test, dataset_meta = load_prepared_dataset_from_minio(args.dataset_prefix)
    feature_cols = dataset_meta.get("feature_cols", [])
    args.window_size = int(dataset_meta.get("seq_len", train.x_seq.shape[1]))
    args.horizon = int(dataset_meta.get("pred_len", train.y.shape[1] if train.y.ndim > 1 else 1))
    loc_count = len(dataset_meta.get("location_to_id", {}))
    max_loc_id = int(max(train.loc_ids.max(), val.loc_ids.max(), test.loc_ids.max()) + 1) if train.loc_ids is not None else 0
    num_locations = loc_count or max_loc_id
    train, val, test, y_mean, y_std, target_mode = prepare_training_targets(train, val, test, args.target_mode)
    scaler = load_pickle(get_client(), MINIO_GOLD_BUCKET, f"{args.dataset_prefix.rstrip('/')}/scaler.pkl")
    if scaler is not None:
        for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
            baseline = persistence_baseline_metrics(split_data, dataset_meta, scaler, y_mean, y_std, target_mode)
            if baseline:
                logger.info(
                    "Persistence baseline %s | mae=%.4f | rmse=%.4f | r2=%.4f",
                    split_name,
                    baseline["mae"],
                    baseline["rmse"],
                    baseline["r2"],
                )

    logger.info("Split - train: %d | val: %d | test: %d", len(train.y), len(val.y), len(test.y))
    if args.min_train_samples > 0 and len(train.y) < args.min_train_samples:
        logger.warning("Train split has only %d samples < min_train_samples=%d.", len(train.y), args.min_train_samples)

    if args.model_type == "mamba":
        mamba_fastpath_ready = _mamba_fastpath_ready()
        mamba_cuda_path_ready = _mamba_cuda_path_ready()
        if args.device == "cuda" and torch.cuda.is_available() and not mamba_cuda_path_ready:
            logger.warning(
                "Mamba CUDA fast path chua san sang (causal_conv1d/selective_scan_cuda missing). "
                "Neu tiep tuc dung CUDA, model se chay selective_scan_ref cham hon nhieu."
            )
        elif args.device == "cuda" and torch.cuda.is_available() and mamba_cuda_path_ready and not mamba_fastpath_ready:
            logger.warning(
                "Mamba fused fast path thieu causal_conv1d fwd/bwd function; se dung CUDA non-fused path thay the."
            )

    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"
    use_amp = args.amp and device.type == "cuda"
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(AQIDataset(train), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(AQIDataset(val), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(AQIDataset(test), shuffle=False, **loader_kwargs)

    model = build_model(args, num_features=train.x_seq.shape[-1]).to(device)
    if args.model_type == "mamba":
        logger.info("Mamba fused fast path active: %s", bool(getattr(model, "use_fast_path", False)))
    criterion = nn.HuberLoss(delta=1.0) if args.loss == "huber" else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_t_max = args.lr_t_max or args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr_t_max, eta_min=args.min_lr)
    logger.info("Device: %s | AMP: %s | scheduler=CosineAnnealingLR(T_max=%d, eta_min=%.2e)", device, use_amp, lr_t_max, args.min_lr)

    mlflow = None
    if args.mlflow_enabled:
        mlflow, _run = start_mlflow_run(
            logger=logger,
            experiment_name=args.mlflow_experiment,
            run_name=f"{args.model_type}_{train_run_id}",
            tracking_uri=args.mlflow_tracking_uri,
            tags={
                "model_type": args.model_type,
                "dataset_prefix": args.dataset_prefix,
                "province": _resolve_single_province(dataset_meta),
            },
        )
    else:
        logger.info("MLflow tracking disabled for this training run.")
    if mlflow:
        mlflow.log_params(
            {
                "model_type": args.model_type,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "window_size": args.window_size,
                "horizon": args.horizon,
                "num_features": int(train.x_seq.shape[-1]),
                "hidden_size": args.hidden_size,
                "d_model": args.d_model,
                "n_layers": args.n_layers,
                "d_state": args.d_state,
                "d_conv": args.d_conv,
                "expand": args.expand,
                "n_heads": args.n_heads,
                "dropout": args.dropout,
                "lr_scheduler": "CosineAnnealingLR",
                "lr_t_max": lr_t_max,
                "min_lr": args.min_lr,
            }
        )

    best_val_loss = float("inf")
    best_filename = f"best_{args.model_type}_aqi.pt"
    best_path = os.path.join(args.out_dir, best_filename)
    history_path = os.path.join(args.out_dir, "metrics_history.csv")
    epochs_without_improvement = 0
    best_epoch = 0
    final_train_loss = float("nan")
    final_val_loss = float("nan")
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "learning_rate", "mae", "rmse", "mae_norm", "rmse_norm", "val_r2", "train_sec"]
        )

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_sec = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                logger,
                epoch,
                args.epochs,
                args.log_interval,
                use_amp,
                args.grad_accum_steps,
                args.max_grad_norm,
            )
            val_metrics = evaluate(model, val_loader, criterion, device, use_amp, y_mean, y_std, target_mode, val.y_base)
            final_train_loss = float(train_loss)
            final_val_loss = float(val_metrics["loss"])
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %02d/%02d | train=%.6f | val_loss=%.6f | mae=%.4f | rmse=%.4f | r2=%.4f | lr=%.2e | %.1fs",
                epoch,
                args.epochs,
                train_loss,
                val_metrics["loss"],
                val_metrics["mae"],
                val_metrics["rmse"],
                val_metrics["r2"],
                current_lr,
                train_sec,
            )
            if mlflow:
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_metrics["loss"],
                        "val_mae": val_metrics["mae"],
                        "val_rmse": val_metrics["rmse"],
                        "val_r2": val_metrics["r2"],
                        "learning_rate": current_lr,
                    },
                    step=epoch,
                )

            with open(history_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [
                        epoch,
                        f"{train_loss:.8f}",
                        f"{val_metrics['loss']:.8f}",
                        f"{current_lr:.12g}",
                        f"{val_metrics['mae']:.8f}",
                        f"{val_metrics['rmse']:.8f}",
                        f"{val_metrics.get('mae_norm', float('nan')):.8f}",
                        f"{val_metrics.get('rmse_norm', float('nan')):.8f}",
                        f"{val_metrics['r2']:.8f}",
                        f"{train_sec:.2f}",
                    ]
                )

            if val_metrics["loss"] < best_val_loss - args.min_delta:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                torch.save(model.state_dict(), best_path)
                epochs_without_improvement = 0
                logger.info("New checkpoint: %s", best_path)
            else:
                epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                logger.info("Early stopping at epoch %d/%d | best_val_loss=%.6f", epoch, args.epochs, best_val_loss)
                break

        model.load_state_dict(torch.load(best_path, map_location=device))
        val_best_metrics = evaluate(model, val_loader, criterion, device, use_amp, y_mean, y_std, target_mode, val.y_base)
        test_metrics = evaluate(model, test_loader, criterion, device, use_amp, y_mean, y_std, target_mode, test.y_base)
        test_preds, test_targets = collect_predictions(model, test_loader, device, use_amp, y_mean, y_std, target_mode, test.y_base)
        prediction_path = save_prediction_results(
            args.out_dir,
            test,
            test_preds,
            test_targets,
            dataset_meta,
            horizon=args.horizon,
        )
        logger.info(
            "TEST | loss=%.6f | mae=%.4f | rmse=%.4f | r2=%.4f",
            test_metrics["loss"],
            test_metrics["mae"],
            test_metrics["rmse"],
            test_metrics["r2"],
        )
        if mlflow:
            mlflow.log_metrics(
                {
                    "test_loss": test_metrics["loss"],
                    "test_mae": test_metrics["mae"],
                    "test_rmse": test_metrics["rmse"],
                    "test_r2": test_metrics["r2"],
                    "best_val_loss": best_val_loss,
                }
            )

        metrics_summary = {
            "val_norm_mae": _metric_float(val_best_metrics, "mae_norm"),
            "val_norm_mse": _metric_float(val_best_metrics, "mse_norm"),
            "val_norm_rmse": _metric_float(val_best_metrics, "rmse_norm"),
            "val_norm_r2": _metric_float(val_best_metrics, "r2_norm"),
            "test_norm_mae": _metric_float(test_metrics, "mae_norm"),
            "test_norm_mse": _metric_float(test_metrics, "mse_norm"),
            "test_norm_rmse": _metric_float(test_metrics, "rmse_norm"),
            "test_norm_r2": _metric_float(test_metrics, "r2_norm"),
            "best_epoch": int(best_epoch),
            "final_gap": float(final_val_loss - final_train_loss),
        }
        metrics_json_path = Path(args.out_dir) / "metrics.json"
        metrics_json_path.write_text(
            json.dumps(metrics_summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        if mlflow:
            mlflow.log_metrics(metrics_summary)

        metadata_path = Path(args.out_dir) / "training_metadata.json"
        training_metadata = {
            "train_run_id": train_run_id,
            "dataset_prefix": args.dataset_prefix,
            "province": _resolve_single_province(dataset_meta),
            "scope": "single_province" if _resolve_single_province(dataset_meta) else "multi_province",
            "model_type": args.model_type,
            "target_col": args.target_col,
            "feature_cols": feature_cols,
            "window_size": args.window_size,
            "horizon": args.horizon,
            "num_locations": num_locations,
            "num_features": int(train.x_seq.shape[-1]),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "metrics_file": str(metrics_json_path),
            "target_normalization": {
                "method": "standard",
                "fit_on": "train",
                "target_mode": target_mode,
                "mean": y_mean,
                "std": y_std,
            },
            "test_metrics": test_metrics,
            "prediction_file": str(prediction_path),
            "model": {
                "hidden_size": args.hidden_size,
                "d_model": args.d_model,
                "n_layers": args.n_layers,
                "d_state": args.d_state,
                "d_conv": args.d_conv,
                "expand": args.expand,
                "n_heads": args.n_heads,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,
            },
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "final_learning_rate": optimizer.param_groups[0]["lr"],
                "lr_scheduler": {
                    "name": "CosineAnnealingLR",
                    "T_max": lr_t_max,
                    "eta_min": args.min_lr,
                },
                "device": str(device),
                "mlflow_enabled": args.mlflow_enabled,
                "mlflow_experiment": args.mlflow_experiment,
            },
            "created_at": now_local().isoformat(),
        }
        metadata_path.write_text(json.dumps(training_metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        upload_training_outputs_to_minio(
            args.out_dir,
            train_run_id,
            args.dataset_prefix,
            logger,
            dataset_meta,
            model_name=args.model_type,
            best_model_filename=best_filename,
        )
        log_artifact_dir(mlflow, args.out_dir, logger)
    finally:
        if mlflow:
            mlflow.end_run()
        if cleanup_out_dir:
            tmp_dir = args.out_dir
            logger.info("Removing temporary local artifacts: %s", tmp_dir)
            for handler in list(logger.handlers):
                handler.flush()
                handler.close()
                logger.removeHandler(handler)
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()