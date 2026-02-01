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
                 prototype_path='weights/scene_prototypes.pkl',
                 dinov3_path='weights/dinov3/model.safetensors',
                 co_occurrence_path='data/co_occurrence.json', # <--- 确保这里有默认值
                 **kwargs):
        super().__init__()
        
        self.device = device
        
        # 1. SAM3
        print("Initializing SAM3...")
        model = build_sam3_image_model(
            bpe_path=f"./sam3/assets/bpe_simple_vocab_16e6.txt.gz", 
            checkpoint_path='weights/sam3/sam3.pt', 
            device="cuda"
        )
        self.processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)
        
        # 2. Class Names
        self.query_words, self.query_idx = get_cls_idx(classname_path)
        self.num_cls = max(self.query_idx) + 1
        self.num_queries = len(self.query_idx)
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        # 3. Params
        self.prob_thd = prob_thd
        self.bg_idx = bg_idx
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.confidence_threshold = confidence_threshold
        self.use_sem_seg = use_sem_seg
        self.use_presence_score = use_presence_score
        self.use_transformer_decoder = use_transformer_decoder
        
        # 4. GCM
        self.use_global_prior = use_global_prior
        if self.use_global_prior and HAS_GCM:
            print("Initializing Global Context Modulator...")
            self.gcm = GlobalContextModulator(
                device=device, 
                prototype_path=prototype_path,
                dinov3_path=dinov3_path,
                co_occurrence_path=co_occurrence_path # <--- 传入路径
            )
        else:
            self.gcm = None
            if self.use_global_prior:
                print("Warning: GCM init failed (missing module?).")
            else:
                print("Global Context Modulator DISABLED.")

    def _inference_single_view(self, image, global_priors=None):
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
                    
                    if global_priors is not None and query_word in global_priors:
                        g_prior_score = global_priors[query_word]
                        
                        # [DEBUG PRINT]
                        # 这是一个只在第一次触发的 print，防止刷屏，但能证明它在工作
                        if not hasattr(self, "_debug_printed"):
                             print(f"\n[GCM LIVE] Applying prior for '{query_word}': Factor={g_prior_score}")
                             
                        presence = presence * g_prior_score
                        
                    seg_logits[query_idx] = seg_logits[query_idx] * presence
            
            if not hasattr(self, "_debug_printed") and global_priors is not None:
                self._debug_printed = True
                
        return seg_logits

    def slide_inference(self, image, stride, crop_size, global_priors=None):
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
                crop_seg_logit = self._inference_single_view(crop_img, global_priors=global_priors)
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
            global_priors = None
            if self.gcm is not None:
                global_priors, _ = self.gcm.get_global_prior(image, self.query_words)
            
            # Step 2: Inference
            if self.slide_crop > 0 and (self.slide_crop < image.size[0] or self.slide_crop < image.size[1]):
                seg_logits = self.slide_inference(image, self.slide_stride, self.slide_crop, global_priors=global_priors)
            else:
                seg_logits = self._inference_single_view(image, global_priors=global_priors)

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