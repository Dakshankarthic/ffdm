"""Diagnose saved SDXL development results and test tile-head scoring."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score

from .calibrate_and_test_protocol import optimize_threshold
from .data.canonicalize import load_canonical_rgb
from .data.tiles import blend_tiles_probability, get_tile_grid_coords, pad_if_needed
from .models.super_frame_v2_detector import SuperFRAMEV2Detector
from .policy import compute_image_score
from .utils import create_heatmap_overlay


@torch.inference_mode()
def inspect_image(path, model):
    image, _ = load_canonical_rgb(path)
    padded, _ = pad_if_needed(np.asarray(image, dtype=np.float32) / 255, 384)
    coords = get_tile_grid_coords(padded.shape[1], padded.shape[0], 384, 256)

    def forward(rgb):
        return model(torch.from_numpy(rgb).permute(2, 0, 1)[None].cuda())

    logits, tile_scores = [], []
    for x, y in coords:
        output = forward(padded[y:y + 384, x:x + 384])
        logits.append(output["mask_logits"][0, 0].cpu().numpy())
        tile_scores.append(output["det_logits"].sigmoid().item())
    heatmap = blend_tiles_probability(logits, coords, *image.size)
    whole = forward(np.asarray(image.resize((384, 384), Image.BILINEAR),
                               dtype=np.float32) / 255)["det_logits"].sigmoid().item()
    pool = float(compute_image_score(heatmap, "top_k", k=2048))
    tile = float(np.mean(sorted(tile_scores)[-2:]))
    return image, heatmap, coords, tile_scores, (pool, whole, tile)


def main():
    torch.set_num_threads(4)
    base = Path(__file__).resolve().parents[2]
    root = base.parent
    out = base / "reports/sdxl_error_audit"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(base / "runs/super_frame_v2_run/best.pt", map_location="cpu", weights_only=False)
    model = SuperFRAMEV2Detector(pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    del ckpt
    model.cuda().eval()
    records = json.loads((base / "runs/real_development_20261114/manifest_multigen.json").read_text())
    validation = [r for r in records if r["split"] == "val"]
    validation = [r for r in validation if r["label"] == 1][:50] + [r for r in validation if r["label"] == 0][:50]
    labels = np.array([r["label"] for r in validation])
    scores = np.array([inspect_image(Path(r["image"]), model)[-1] for r in validation])
    # Compare replacements for the local evidence, with a fixed short weight grid.
    alpha = max(np.linspace(0, 1, 21), key=lambda a: roc_auc_score(labels, a*scores[:,1]+(1-a)*scores[:,2]))
    tau, _ = optimize_threshold(labels, alpha*scores[:,1]+(1-alpha)*scores[:,2])
    baseline = json.loads((base / "reports/calibrated_fused_protocol_results.json").read_text())["v2_cross_attn"]["calibration"]
    test = json.loads((root / "baseline_work/data_manifests/test_sdxl_manifest.json").read_text())["entries"]
    rows, panels = [], []
    for r in test:
        image, heatmap, coords, tile_scores, (pool, whole, tile) = inspect_image(root / r["image_path"], model)
        label = int(r["label"])
        mask = np.zeros(heatmap.shape, dtype=bool)
        if label:
            with Image.open(root / r["mask_path"]) as im:
                if im.size != image.size:
                    raise ValueError(f"Mask shape mismatch: {r['sample_id']}")
                mask = np.asarray(im.convert("RGB")).mean(2) > 127.5
        old = baseline["best_alpha"]*whole + (1-baseline["best_alpha"])*pool
        new = alpha*whole + (1-alpha)*tile
        row = dict(id=r["sample_id"], label=label, pool=pool, whole=whole, tile=tile,
                   hybrid=old, candidate=new, baseline_pred=int(old >= baseline["tau_hybrid"]),
                   candidate_pred=int(new >= tau), edited_fraction=float(mask.mean()),
                   edit_mean=float(heatmap[mask].mean()) if mask.any() else None,
                   background_mean=float(heatmap[~mask].mean()) if (~mask).any() else None,
                   tiles=[dict(x=x, y=y, score=s, edit_pixels=int(mask[y:y+384,x:x+384].sum()))
                          for (x,y),s in zip(coords,tile_scores)])
        rows.append(row)
        if row["baseline_pred"] != label:
            marked = image.copy()
            hottest = np.unravel_index(heatmap.argmax(), heatmap.shape)
            y, x = hottest
            ImageDraw.Draw(marked).ellipse((x-10,y-10,x+10,y+10), outline="yellow", width=3)
            panels.append((row, marked, Image.fromarray(mask.astype("uint8")*255).convert("RGB"),
                           create_heatmap_overlay(image, heatmap, threshold=.35)))
    sheet = Image.new("RGB", (960, 244*len(panels)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (row, *images) in enumerate(panels):
        title = "Missed edit" if row["label"] else "False alarm"
        draw.text((7,244*i+3), f"{title}  {row['id']}   whole {row['whole']:.3f}   pool {row['pool']:.3f}   hybrid {row['hybrid']:.3f}", fill="black")
        for col, (picture, caption) in enumerate(zip(images, ("Image + peak", "Reference mask", "Predicted mask overlay"))):
            picture.thumbnail((313,200))
            sheet.paste(picture,(320*col,244*i+21))
            draw.text((320*col+5,244*i+224),caption,fill="black")
    sheet.save(out / "errors.png")
    results = dict(scope="Previously inspected SDXL subset; development only", candidate_alpha=float(alpha),
                   candidate_threshold=tau, baseline_correct=sum(r["baseline_pred"] == r["label"] for r in rows),
                   candidate_correct=sum(r["candidate_pred"] == r["label"] for r in rows), images=rows)
    (out / "scores.json").write_text(json.dumps(results, indent=2))
    print({k:v for k,v in results.items() if k != "images"})


if __name__ == "__main__":
    main()
