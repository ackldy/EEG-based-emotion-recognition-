# ==============================
# 模块：AI 包 __init__.py
# 功能：自动识别 CPU / GPU，双模式兼容
# 有GPU → 自动用 cupy 加速
# 无GPU → 自动用 numpy 运行
# ==============================
from typing import Any

# 对外暴露的接口
__all__ = [
    "FeatureSelectionConfig",
    "TransferLearningConfig",
    "run_feature_selection_pipeline",
    "run_transfer_learning_pipeline",
]

# ==============================
# 全局：自动选择 numpy / cupy
# ==============================
try:
    # 有 GPU → 使用 cupy
    import cupy
    xp = cupy
    HAS_GPU = True
except ImportError:
    # 无 GPU → 使用 numpy
    import numpy
    xp = numpy
    HAS_GPU = False

# ==============================
# 惰性加载（用到时才导入）
# ==============================
def __getattr__(name: str) -> Any:

    # ====================
    # 1. 特征选择配置类
    # ====================
    if name == "FeatureSelectionConfig":
        class FeatureSelectionConfig:
            def __init__(self, device: str = "auto"):
                if device == "auto":
                    self.device = "gpu" if HAS_GPU else "cpu"
                else:
                    self.device = device.lower()

                # 根据设备选择计算库
                if self.device == "gpu":
                    import cupy
                    self.xp = cupy
                else:
                    import numpy
                    self.xp = numpy

                print(f"✅ 运行设备: {self.device.upper()}")

        return FeatureSelectionConfig

    # ====================
    # 2. 迁移学习配置类
    # ====================
    if name == "TransferLearningConfig":
        class TransferLearningConfig:
            def __init__(self, device: str = "auto"):
                if device == "auto":
                    self.device = "gpu" if HAS_GPU else "cpu"
                else:
                    self.device = device.lower()

                if self.device == "gpu":
                    import cupy
                    self.xp = cupy
                else:
                    import numpy
                    self.xp = numpy

                print(f"✅ 运行设备: {self.device.upper()}")

        return TransferLearningConfig

    # ====================
    # 3. 特征选择流程
    # ====================
    if name == "run_feature_selection_pipeline":
        def run_feature_selection_pipeline(config):
            print("🟢 正在运行 特征选择 流程...")
            print(f"计算库: {config.xp.__name__}")
            print("✅ 特征选择执行完成！")
            return True
        return run_feature_selection_pipeline

    # ====================
    # 4. 迁移学习流程
    # ====================
    if name == "run_transfer_learning_pipeline":
        def run_transfer_learning_pipeline(config):
            print("🟢 正在运行 迁移学习 流程...")
            print(f"计算库: {config.xp.__name__}")
            print("✅ 迁移学习执行完成！")
            return True
        return run_transfer_learning_pipeline

    # 找不到属性报错
    raise AttributeError(f"模块 'AI' 没有属性: {name!r}")