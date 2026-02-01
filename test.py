import torch
import mmcv
import mmseg
from mmcv.ops import get_compiler_version, get_compiling_cuda_version

print("=== 环境检查 ===")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"显卡型号: {torch.cuda.get_device_name(0)}")
print(f"显卡数量: {torch.cuda.device_count()}")
print(f"MMCV 版本: {mmcv.__version__}")
print(f"MMSegmentation 版本: {mmseg.__version__}")
print(f"MMCV 编译 CUDA 版本: {get_compiling_cuda_version()}")
print(f"当前运行 CUDA 版本: {torch.version.cuda}")

# 简单的显存测试
x = torch.randn(1000, 1000).cuda()
print("显存写入测试成功！")
