import torch
from torch import nn
import torch.nn.functional as F
from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image
import os

# SAM3 相关引入
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

try:
    from global_context import GlobalContextModulator
    HAS_GCM = True
except ImportError:
    print("Warning: global_context module not found.")
    HAS_GCM = False

def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)
    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(',')
        names_i = [i.strip() for i in names_i]
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices

@MODELS.register_module()
class SegEarthOV3Segmentation(BaseSegmentor):
    def __init__(self, classname_path,
                 device=torch.device('cuda'),
                 prob_thd=0.0,
                 bg_idx=0,
                 slide_stride=0,
                 slide_crop=0,
                 confidence_threshold=0.5,
                 use_sem_seg=True,
                 use_presence_score=True,
                 use_transformer_decoder=True,
                 use_global_prior=True, 
                 prototype_path='weights/inria_prototypes.pkl',
                 dinov3_path='weights/dinov3/model.safetensors',
                 co_occurrence_path='data/co_occurrence_inria.json',
                 gcm_alpha=1.0, # <--- [NEW] 默认强度 1.0
                 use_adaptive_alpha=False,
                 alpha_max=1.0,
                 gcm_temperature=0.1,
                 use_local_prior=False,
                 local_prior_lambda=0.5,
                 local_prior_use_similarity=True,
                 **kwargs):
        super().__init__()
        
        self.device = device

        # === 1. 自动读取配置中的 test_cfg 参数 ===
        if 'test_cfg' in kwargs and kwargs['test_cfg'] is not None:
            test_cfg = kwargs['test_cfg']
            if test_cfg.get('mode') == 'slide':
                slide_crop = test_cfg.get('crop_size', slide_crop)
                slide_stride = test_cfg.get('stride', slide_stride)
                print(f"[SegEarth-OV3] 自动启用滑动窗口模式: Crop={slide_crop}, Stride={slide_stride}")
        
        # === 2. 允许从 Config 的 kwargs 里覆盖 gcm_alpha ===
        if 'gcm_alpha' in kwargs:
            self.gcm_alpha = kwargs['gcm_alpha']
        else:
            self.gcm_alpha = gcm_alpha

        self.use_adaptive_alpha = use_adaptive_alpha
        self.alpha_max = alpha_max
        self.gcm_temperature = gcm_temperature
        self.use_local_prior = use_local_prior
        self.local_prior_lambda = local_prior_lambda
        self.local_prior_use_similarity = local_prior_use_similarity
        
        print(f"[SegEarth-OV3] Global Prior Strength (Alpha): {self.gcm_alpha}")
        if self.use_adaptive_alpha:
            print(f"[SegEarth-OV3] Adaptive Alpha ENABLED (alpha_max={self.alpha_max})")
        if self.use_local_prior:
            print(f"[SegEarth-OV3] Local Prior ENABLED (lambda={self.local_prior_lambda}, use_similarity={self.local_prior_use_similarity})")

        # 3. SAM3
        print("Initializing SAM3...")
        model = build_sam3_image_model(
            bpe_path=f"./sam3/assets/bpe_simple_vocab_16e6.txt.gz", 
            checkpoint_path='weights/sam3/sam3.pt', 
            device="cuda"
        )
        self.processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)
        
        # 4. Class Names
        self.query_words, self.query_idx = get_cls_idx(classname_path)
        self.num_cls = max(self.query_idx) + 1
        self.num_queries = len(self.query_idx)
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        # 5. Params
        self.prob_thd = prob_thd
        self.bg_idx = bg_idx
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.confidence_threshold = confidence_threshold
        self.use_sem_seg = use_sem_seg
        self.use_presence_score = use_presence_score
        self.use_transformer_decoder = use_transformer_decoder
        
        # 6. GCM
        self.use_global_prior = use_global_prior
        if self.use_global_prior and HAS_GCM:
            print("Initializing Global Context Modulator...")
            self.gcm = GlobalContextModulator(
                device=device, 
                prototype_path=prototype_path,
                dinov3_path=dinov3_path,
                co_occurrence_path=co_occurrence_path,
                temperature=gcm_temperature
            )
        else:
            self.gcm = None
            if self.use_global_prior:
                print("Warning: GCM init failed (missing module?).")
            else:
                print("Global Context Modulator DISABLED.")

    def _compute_alpha(self, scene_info):
        if not self.use_adaptive_alpha or scene_info is None:
            return self.gcm_alpha

        num_scenes = scene_info.get("num_scenes", 0)
        entropy = scene_info.get("entropy")
        if entropy is None or num_scenes <= 1:
            return self.gcm_alpha

        max_entropy = torch.log(torch.tensor(float(num_scenes), device=entropy.device))
        normalized = torch.clamp(entropy / (max_entropy + 1e-8), 0.0, 1.0)
        alpha = self.alpha_max * (1.0 - normalized)
        return float(alpha)

    def _build_prior_bundle(self, image):
        if self.gcm is None:
            return None
        priors, feat, scene_info = self.gcm.get_global_prior(
            image, self.query_words, return_scene_info=True
        )
        alpha = self._compute_alpha(scene_info)
        return {
            "priors": priors,
            "feat": feat,
            "alpha": alpha,
            "scene_info": scene_info
        }

    def _mix_prior_bundles(self, global_bundle, local_bundle):
        if global_bundle is None:
            return local_bundle
        if local_bundle is None:
            return global_bundle

        lambda_global = self.local_prior_lambda
        if (
            self.local_prior_use_similarity
            and global_bundle.get("feat") is not None
            and local_bundle.get("feat") is not None
        ):
            sim = F.cosine_similarity(global_bundle["feat"], local_bundle["feat"]).clamp(-1.0, 1.0)
            lambda_global = float((sim + 1.0) / 2.0)

        mixed_priors = {}
        for query_word in self.query_words:
            global_factor = global_bundle["priors"].get(query_word, 1.0)
            local_factor = local_bundle["priors"].get(query_word, 1.0)
            mixed_priors[query_word] = lambda_global * global_factor + (1.0 - lambda_global) * local_factor

        mixed_alpha = (
            lambda_global * global_bundle.get("alpha", self.gcm_alpha)
            + (1.0 - lambda_global) * local_bundle.get("alpha", self.gcm_alpha)
        )
        return {
            "priors": mixed_priors,
            "feat": global_bundle.get("feat"),
            "alpha": mixed_alpha,
            "scene_info": global_bundle.get("scene_info")
        }

    def _inference_single_view(self, image, prior_bundle=None):
        w, h = image.size
        seg_logits = torch.zeros((self.num_queries, h, w), device=self.device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)
            
            for query_idx, query_word in enumerate(self.query_words):
                self.processor.reset_all_prompts(inference_state)
                inference_state = self.processor.set_text_prompt(state=inference_state, prompt=query_word)

                if self.use_transformer_decoder:
                    if inference_state['masks_logits'].shape[0] > 0:
                        inst_len = inference_state['masks_logits'].shape[0]
                        for inst_id in range(inst_len):
                            instance_logits = inference_state['masks_logits'][inst_id].squeeze()
                            instance_score = inference_state['object_score'][inst_id]
                            if instance_logits.shape != (h, w):
                                instance_logits = F.interpolate(
                                    instance_logits.view(1, 1, *instance_logits.shape), 
                                    size=(h, w), mode='bilinear', align_corners=False
                                ).squeeze()
                            seg_logits[query_idx] = torch.max(seg_logits[query_idx], instance_logits * instance_score)
                    
                if self.use_sem_seg:
                    semantic_logits = inference_state['semantic_mask_logits']
                    if semantic_logits.shape != (h, w):
                            semantic_logits = F.interpolate(
                                semantic_logits, size=(h, w), mode='bilinear', align_corners=False
                            ).squeeze()
                    seg_logits[query_idx] = torch.max(seg_logits[query_idx], semantic_logits)
                
                # --- Presence Score Modulation ---
                if self.use_presence_score:
                    presence = inference_state["presence_score"]
                    
                    if prior_bundle is not None and query_word in prior_bundle["priors"]:
                        # 核心修改：引入 Alpha 指数调节
                        raw_factor = prior_bundle["priors"][query_word]
                        
                        # Apply Alpha: factor ^ alpha
                        # 如果 alpha > 1，会放大 factor 的效果 (e.g. 0.9 -> 0.81, 1.1 -> 1.21)
                        # 如果 alpha < 1，会平滑 factor 的效果
                        tuned_factor = raw_factor ** prior_bundle["alpha"]
                        
                        if not hasattr(self, "_debug_printed"):
                            print(f"\n[GCM LIVE] '{query_word}': Raw={raw_factor:.4f} -> Tuned(alpha={prior_bundle['alpha']:.4f})={tuned_factor:.4f}")
                             
                        presence = presence * tuned_factor
                        
                    seg_logits[query_idx] = seg_logits[query_idx] * presence
            
            if not hasattr(self, "_debug_printed") and prior_bundle is not None:
                self._debug_printed = True
                
        return seg_logits

    def slide_inference(self, image, stride, crop_size, global_bundle=None):
        w_img, h_img = image.size
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(crop_size, int): crop_size = (crop_size, crop_size)
        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        preds = torch.zeros((self.num_queries, h_img, w_img), device=self.device)
        count_mat = torch.zeros((1, h_img, w_img), device=self.device)
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = image.crop((x1, y1, x2, y2))
                if self.use_local_prior and self.gcm is not None:
                    local_bundle = self._build_prior_bundle(crop_img)
                    prior_bundle = self._mix_prior_bundles(global_bundle, local_bundle)
                else:
                    prior_bundle = global_bundle
                crop_seg_logit = self._inference_single_view(crop_img, prior_bundle=prior_bundle)
                preds[:, y1:y2, x1:x2] += crop_seg_logit
                count_mat[:, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0, "Error: Sparse sliding window coverage."
        preds = preds / count_mat
        return preds

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [dict(ori_shape=inputs.shape[2:], img_shape=inputs.shape[2:], pad_shape=inputs.shape[2:], padding_size=[0, 0, 0, 0])] * inputs.shape[0]
        
        for i, meta in enumerate(batch_img_metas):
            image_path = meta.get('img_path')
            image = Image.open(image_path).convert('RGB')
            ori_shape = meta['ori_shape']

            # Step 1: Global Prior
            global_bundle = None
            if self.gcm is not None:
                global_bundle = self._build_prior_bundle(image)
            
            # Step 2: Inference
            should_slide = False
            if isinstance(self.slide_crop, (list, tuple)):
                should_slide = all(c > 0 for c in self.slide_crop)
            elif isinstance(self.slide_crop, int):
                should_slide = self.slide_crop > 0
            
            if should_slide:
                crop_h = self.slide_crop[0] if isinstance(self.slide_crop, (list, tuple)) else self.slide_crop
                crop_w = self.slide_crop[1] if isinstance(self.slide_crop, (list, tuple)) else self.slide_crop
                
                if image.size[0] > crop_w or image.size[1] > crop_h:
                    seg_logits = self.slide_inference(image, self.slide_stride, self.slide_crop, global_bundle=global_bundle)
                else:
                    seg_logits = self._inference_single_view(image, prior_bundle=global_bundle)
            else:
                seg_logits = self._inference_single_view(image, prior_bundle=global_bundle)

            # Resize
            if seg_logits.shape[-2:] != ori_shape:
                seg_logits = F.interpolate(seg_logits.unsqueeze(0), size=ori_shape, mode='bilinear', align_corners=False).squeeze(0)
            
            # Post-process
            if self.num_cls != self.num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(self.num_cls, len(self.query_idx), 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]
                seg_pred = seg_logits.argmax(0, keepdim=False)
            else:
                seg_pred = torch.argmax(seg_logits, dim=0)
            
            max_vals = seg_logits.max(0)[0]
            seg_pred[max_vals < self.prob_thd] = self.bg_idx

            data_samples[i].set_data({
                'seg_logits': PixelData(**{'data': seg_logits}),
                'pred_sem_seg': PixelData(**{'data': seg_pred.unsqueeze(0)})
            })
            
        return data_samples
    
    def _forward(data_samples): pass
    def inference(self, img, batch_img_metas): pass
    def encode_decode(self, inputs, batch_img_metas): pass
    def extract_feat(self, inputs): pass
    def loss(self, inputs, data_samples): pass
