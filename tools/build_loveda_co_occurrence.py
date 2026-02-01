import argparse
import glob
import json
import os
import sys

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_local_dinov3
from segearthov3_segmentor import get_cls_idx


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build LoveDA co-occurrence priors from unlabeled images."
    )
    parser.add_argument("--image-root", required=True, help="Folder with images.")
    parser.add_argument("--class-path", default="./configs/cls_loveda.txt")
    parser.add_argument("--prototype-path", required=True)
    parser.add_argument("--dinov3-path", required=True)
    parser.add_argument("--output", default="data/co_occurrence_loveda_auto.json")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--score-mode", choices=["presence", "area"], default="presence")
    parser.add_argument("--sam3-checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoothing", type=float, default=1e-3)
    return parser.parse_args()


def collect_images(image_root):
    patterns = ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg")
    images = []
    for pattern in patterns:
        images.extend(glob.glob(os.path.join(image_root, pattern)))
    return sorted(images)


def build_scene_weights(dino_model, img_tensor, prototypes, temperature):
    out = dino_model.forward_features(img_tensor)
    global_feat = F.normalize(out["x_norm_clstoken"], p=2, dim=1)
    sims = torch.mm(global_feat, prototypes.t())
    weights = F.softmax(sims / temperature, dim=1)
    return weights.squeeze(0), global_feat


def compute_presence_scores(
    processor,
    image_pil,
    query_words,
    device,
    score_mode="presence",
):
    inference_state = processor.set_image(image_pil)
    scores = []
    width, height = image_pil.size
    for query_word in query_words:
        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(state=inference_state, prompt=query_word)
        if score_mode == "area":
            masks = inference_state.get("semantic_mask_logits")
            if masks is None:
                score = 0.0
            else:
                logits = masks
                if logits.shape[-2:] != (height, width):
                    logits = F.interpolate(
                        logits, size=(height, width), mode="bilinear", align_corners=False
                    )
                prob = torch.sigmoid(logits).mean().item()
                score = prob
        else:
            score = float(inference_state["presence_score"])
        scores.append(score)
    return np.asarray(scores, dtype=np.float64)


def main():
    args = parse_args()
    images = collect_images(args.image_root)
    if args.max_images > 0:
        images = images[: args.max_images]
    if len(images) == 0:
        raise RuntimeError(f"No images found under {args.image_root}.")

    device = torch.device(args.device)
    dino_model = load_local_dinov3(args.dinov3_path, device=device)
    dino_model.eval()

    prototype_data = joblib.load(args.prototype_path)
    prototypes = torch.tensor(
        prototype_data["centers"], dtype=torch.float32, device=device
    )
    prototypes = F.normalize(prototypes, p=2, dim=1)
    num_scenes = prototypes.shape[0]

    query_words, _ = get_cls_idx(args.class_path)
    num_queries = len(query_words)
    accum = np.zeros((num_scenes, num_queries), dtype=np.float64)
    counts = np.zeros((num_scenes,), dtype=np.float64)

    preprocess = transforms.Compose(
        [
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    sam3_model = build_sam3_image_model(
        bpe_path="./sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        checkpoint_path=args.sam3_checkpoint,
        device=args.device,
    )
    processor = Sam3Processor(sam3_model, confidence_threshold=0.5, device=device)

    for idx, path in enumerate(images, start=1):
        image = Image.open(path).convert("RGB")
        img_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            weights, _ = build_scene_weights(
                dino_model, img_tensor, prototypes, args.temperature
            )

        scores = compute_presence_scores(
            processor, image, query_words, device, score_mode=args.score_mode
        )

        for scene_idx in range(num_scenes):
            weight = weights[scene_idx].item()
            accum[scene_idx] += weight * scores
            counts[scene_idx] += weight

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(images)} images...")

    priors = {}
    for scene_idx in range(num_scenes):
        denom = max(counts[scene_idx], 1e-8)
        scene_scores = accum[scene_idx] / denom
        scene_scores = (scene_scores + args.smoothing).astype(np.float64)
        scene_scores = scene_scores / scene_scores.max()
        priors[str(scene_idx)] = {
            query_words[i]: float(scene_scores[i]) for i in range(num_queries)
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(priors, f, ensure_ascii=False, indent=2)
    print(f"Saved co-occurrence priors to {args.output}")


if __name__ == "__main__":
    main()
