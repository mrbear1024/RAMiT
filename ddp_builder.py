"""
================================================================================
ddp_builder.py - 分布式数据并行 (DDP) 构建器模块
================================================================================
本模块提供了分布式训练所需的各种构建函数，包括：
    - 数据集和数据加载器构建
    - 模型和优化器构建
    - 渐进式学习支持
    - 损失函数构建

分布式训练流程:
    1. build_dataset: 构建分布式数据加载器
    2. build_model_optimizer_scaler: 构建 DDP 模型、优化器和梯度缩放器
    3. rebuild_progressive: 渐进式学习中重建数据加载器和模型
    4. build_loss_func: 构建损失函数

关键特性:
    - 支持多种图像恢复任务: SR, DN, LLE, DR, CBCT
    - 使用 DistributedSampler 实现数据并行
    - 支持混合精度训练 (autocast)
    - 支持分层学习率衰减 (layer decay)
================================================================================
"""

# 导入数据集类
from dataset import (
    DIV2KDatasetRandomCrop,
    DF2KDatasetRandomCrop,
    DFBWDatasetRandomCrop,
    LLEDatasetRandomCrop,
    DRDatasetRandomCrop,
    CBCTGrayDenoiseDatasetRandomCrop
)
from torch.utils.data import DataLoader, DistributedSampler
from importlib import import_module
from pathlib import Path
from utils.train_utils import param_groups_lrd, NativeScalerWithGradNormCount, CharbonnierLoss
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
from timm.models.layers import _assert


# =============================================================================
# 数据集构建函数
# =============================================================================

def _resolve_cbct_split(root: Path, split: str):
    """
    解析 CBCT 数据集的 HQ/LQ 目录路径

    根据数据集目录结构，自动确定 HQ 和 LQ 的完整路径。

    支持的目录结构:
        结构1 (推荐):
            root/
            ├── HQ/{split}/     # 如 HQ/training_set/
            └── LQ/{split}/     # 如 LQ/training_set/

        结构2 (回退):
            root/
            ├── HQ/             # 直接使用 HQ 目录
            └── LQ/             # 直接使用 LQ 目录

    Args:
        root (Path): CBCT 数据集根目录 (如 'datasets_npy')
        split (str): 数据集分割名称 (如 'training_set', 'validation_set', 'test_set')

    Returns:
        tuple[Path, Path]: (hq_path, lq_path) HQ 和 LQ 目录的完整路径

    Example:
        >>> _resolve_cbct_split(Path('datasets_npy'), 'training_set')
        (Path('datasets_npy/HQ/training_set'), Path('datasets_npy/LQ/training_set'))
    """
    # 尝试结构1: root/HQ/{split} 和 root/LQ/{split}
    hq_split = root / 'HQ' / split
    lq_split = root / 'LQ' / split

    # 如果分割目录存在，返回分割路径
    if hq_split.exists() and lq_split.exists():
        return hq_split, lq_split

    # 回退到结构2: 直接使用 root/HQ 和 root/LQ
    return root / 'HQ', root / 'LQ'


def build_dataset(rank, ps, bs, args):
    """
    构建分布式训练数据集和数据加载器

    根据 args.data_name 选择对应的数据集类，并创建分布式数据加载器。

    Args:
        rank (int): 当前进程在分布式训练中的排名 (0 到 world_size-1)
        ps (int): patch size - LQ 图像的裁剪大小
        bs (int): batch size - 每个 GPU 的批大小
        args (Namespace): 训练参数，需要包含:
            - target_mode (str): 目标模式，用于区分 SR 和其他任务
                - 末尾是数字 (如 'x2', 'x3', 'x4'): 超分辨率任务
                - 包含 'gray': 灰度图像处理
            - data_name (str): 数据集名称
                - 'DIV2K': DIV2K 超分数据集
                - 'DF2K': DIV2K + Flickr2K 合并超分数据集
                - 'DFBW': 去噪数据集
                - 'LLE': 低光增强数据集
                - 'DR': 去雨数据集
                - 'CBCT': CBCT 医学影像去噪数据集
            - scale (int): 超分辨率缩放因子 (SR 任务)
            - model_time (str): 模型标识符
            - patch_load (bool): 是否预加载裁剪参数
            - sigma (tuple): 噪声强度范围 (去噪任务)
            - cbct_train_root (str, optional): CBCT 训练数据根目录
            - cbct_train_split (str, optional): CBCT 训练数据分割名称
            - world_size (int): 分布式训练的总进程数
            - num_workers (int): DataLoader 的工作进程数
            - pin_memory (bool): 是否将数据加载到固定内存

    Returns:
        DataLoader: 配置了分布式采样器的数据加载器

    数据集选择逻辑:
        1. 如果 target_mode 以数字结尾 -> 超分辨率任务 (DIV2K 或 DF2K)
        2. 否则 -> 其他恢复任务 (DFBW, CBCT, LLE, DR)

    Note:
        - 使用 DistributedSampler 确保每个 GPU 处理不同的数据子集
        - shuffle=True 在采样器中启用，而非 DataLoader 中
    """

    # 判断是否为超分辨率任务 (target_mode 末尾是数字，如 'x2', 'light_srx4')
    if args.target_mode[-1].isdigit():
        # ===== 超分辨率任务 =====
        if args.data_name == 'DIV2K':
            # DIV2K 数据集: 800 张训练图像
            train_data = DIV2KDatasetRandomCrop(
                '../DIV2K/DIV2K_train_HR',                          # HQ 图像目录
                f'../DIV2K/DIV2K_train_LR_bicubic/X{args.scale}/',  # LQ 图像目录 (根据 scale 选择)
                ps,              # patch size
                args.model_time,
                args.patch_load
            )
        elif args.data_name == 'DF2K':
            # DF2K 数据集: DIV2K (800张) + Flickr2K (2650张)
            train_data = DF2KDatasetRandomCrop(
                '../DIV2K/DIV2K_train_HR',                          # DIV2K HQ
                f'../DIV2K/DIV2K_train_LR_bicubic/X{args.scale}/',  # DIV2K LQ
                '../Flickr2K/Flickr2K_HR',                          # Flickr2K HQ
                f'../Flickr2K/Flickr2K_LR_bicubic/X{args.scale}',   # Flickr2K LQ
                ps,
                args.model_time,
                args.patch_load
            )
    else:
        # ===== 非超分辨率任务 =====
        # 检查是否需要灰度模式
        gray = False if 'gray' not in args.target_mode else True

        if args.data_name == 'DFBW':
            # DFBW 去噪数据集: DIV2K + Flickr2K + BSDS500 + WED
            # 只需要 HQ 图像，LQ 通过添加噪声动态生成
            train_data = DFBWDatasetRandomCrop(
                '../DIV2K/DIV2K_train_HR',  # DIV2K HQ
                '../Flickr2K/Flickr2K_HR',  # Flickr2K HQ
                '../BSDS500/HQ',            # BSDS500 HQ
                '../WED/HQ',                # WED HQ
                ps,
                args.sigma,                 # 噪声强度范围
                gray,                       # 是否灰度模式
                args.model_time,
                args.patch_load
            )
        elif args.data_name == 'CBCT':
            # CBCT 医学影像去噪数据集
            # 从参数获取根目录和分割名称，支持默认值
            cbct_root = Path(getattr(args, 'cbct_train_root', 'datasets_npy'))
            cbct_split = getattr(args, 'cbct_train_split', 'training_set')

            # 解析 HQ/LQ 目录路径
            hq_root, lq_root = _resolve_cbct_split(cbct_root, cbct_split)

            train_data = CBCTGrayDenoiseDatasetRandomCrop(
                hq_root,         # HQ 图像目录
                lq_root,         # LQ 图像目录
                ps,
                args.model_time,
                args.patch_load,
                True             # recursive=True, 递归扫描子目录
            )
        elif args.data_name == 'LLE':
            # 低光增强数据集: LOL + VELOL
            train_data = LLEDatasetRandomCrop(
                '../LOL/HQ',     # LOL HQ
                '../VELOL/HQ',   # VELOL HQ
                '../LOL/LQ',     # LOL LQ
                '../VELOL/LQ',   # VELOL LQ
                ps,
                args.model_time,
                args.patch_load
            )
        elif args.data_name == 'DR':
            # 去雨数据集: Rain13K
            train_data = DRDatasetRandomCrop(
                '../Rain13K/HQ',  # 无雨清晰图像
                '../Rain13K/LQ',  # 带雨图像
                ps,
                args.model_time,
                args.patch_load
            )

    # 创建分布式采样器
    # num_replicas: 总 GPU 数量
    # rank: 当前 GPU 编号
    # shuffle: 每个 epoch 重新打乱数据
    train_sampler = DistributedSampler(
        train_data,
        num_replicas=args.world_size,
        rank=rank,
        shuffle=True
    )

    # 创建数据加载器
    # 注意: 使用 sampler 时不能设置 shuffle=True
    train_loader = DataLoader(
        train_data,
        batch_size=bs,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    return train_loader


# =============================================================================
# 模型构建函数
# =============================================================================

def build_model_optimizer_scaler(gpu, args, opts):
    """
    构建模型、优化器和梯度缩放器

    创建 RAMiT 模型并配置分布式数据并行 (DDP)，同时设置优化器和
    混合精度训练所需的梯度缩放器。

    Args:
        gpu (int): 当前 GPU 设备 ID
        args (Namespace): 训练参数，需要包含:
            - model_name (str): 模型名称 ('RAMiT', 'RAMiT-1', 'RAMiT-slimSR', 'RAMiT-slimLLE')
            - finetune (bool): 是否为微调模式
            - pretrain_path (str): 预训练权重路径 (微调时必需)
            - warm_start (bool): 是否使用 warm start
            - warm_start_epoch (int): warm start 的 epoch 数
            - load_model (bool): 是否从检查点加载模型
            - checkpoint_epoch (int): 检查点的 epoch 数
            - model_time (str): 模型标识符
            - weight_decay (float): 权重衰减
            - layer_decay (float): 分层学习率衰减系数
            - batch_size (list): 批大小列表
            - world_size (int): 分布式训练的总进程数
            - autocast (bool): 是否启用混合精度训练
        opts (dict): 从 YAML 配置文件解析的模型选项

    Returns:
        tuple: (model, optimizer, loss_scaler)
            - model: DDP 包装的模型
            - optimizer: Adam 优化器
            - loss_scaler: 混合精度梯度缩放器 (若 autocast=False 则为 None)

    模型加载优先级:
        1. load_model=True: 从检查点恢复训练
        2. finetune=True: 从预训练权重微调
        3. 否则: 从头训练

    Note:
        - 使用分层学习率衰减 (layer_decay) 为不同深度的层设置不同学习率
        - warm_start 时只训练 'to_target' 层，其他层冻结
    """
    # 验证模型名称
    _assert(args.model_name in ['RAMiT', 'RAMiT-1', 'RAMiT-slimSR', 'RAMiT-slimLLE'],
            "'model_name' should be RAMiT, RAMiT-1, RAMiT-slimSR, RAMiT-slimLLE")

    # 验证微调参数的一致性
    if not args.finetune:
        _assert(args.pretrain_path is None and not args.warm_start and args.warm_start_epoch is None,
                "Some arguments that must not be decided are assigned.")
    else:
        _assert(args.pretrain_path is not None, "--pretrain_path should be set for fine-tuning.")

    # 动态导入模型模块
    # 'RAMiT' -> 'my_model.ramit'
    # 'RAMiT-1' -> 'my_model.ramit' (去掉末尾的 '-1')
    # 'RAMiT-slimSR' -> 'my_model.ramit_slimsr'
    if '1' not in args.model_name:
        module = import_module(f"my_model.{args.model_name.lower().replace('-', '_')}")
    else:
        module = import_module(f"my_model.{args.model_name.lower().replace('-', '_')[:-2]}")

    # 创建模型实例
    model = module.make_model(args, opts, 0)

    # 处理微调和 warm start
    if args.finetune:
        if not args.load_model:
            # 加载预训练权重
            sd = torch.load(args.pretrain_path, map_location='cpu')
            new_sd = OrderedDict()

            # 复制预训练权重，但 'to_target' 层使用随机初始化
            # (to_target 是输出层，微调时可能需要不同的输出维度)
            for n,p in sd.items():
                new_sd[n] = p if 'to_target' not in n else model.state_dict()[n]

            # 补充模型中有但预训练权重中没有的参数
            for n,p in model.state_dict().items():
                new_sd[n] = p if n not in new_sd else new_sd[n]

            print(model.load_state_dict(new_sd, strict=False))

        # warm start: 冻结除 to_target 外的所有层
        if args.warm_start and args.checkpoint_epoch <= args.warm_start_epoch:
            for n,p in model.named_parameters():
                p.requires_grad = False if 'to_target' not in n else True

    # 从检查点恢复训练
    if args.load_model:
        sd = torch.load(f'./models/{args.model_time}/model_{str(args.checkpoint_epoch).zfill(3)}.pth', map_location='cpu')
        # 排除 attn_mask (注意力掩码可能因 patch size 变化而不兼容)
        new_sd = OrderedDict([(n,p) for n,p in sd.items() if 'attn_mask' not in n])
        print(model.load_state_dict(new_sd, strict=False))

    # 将模型移动到 GPU
    model = model.to(gpu)
    model_no_ddp = model

    # 配置分层学习率衰减的参数组
    param_groups = param_groups_lrd(
        model_no_ddp,
        weight_decay=args.weight_decay,
        no_weight_decay_list=model_no_ddp.no_weight_decay(),
        layer_decay=args.layer_decay,
        model_name=args.model_name
    )

    # 使用 DDP 包装模型
    model = DDP(model, device_ids=[gpu])

    # 计算初始学习率 (基于批大小线性缩放)
    # 基准: 批大小 64 对应 lr=0.0004
    args.init_lr = 0.0004 / 64 * args.batch_size[0]*args.world_size

    # 创建 Adam 优化器
    optimizer = torch.optim.Adam(param_groups, lr=args.init_lr, weight_decay=args.weight_decay)

    # 创建梯度缩放器 (混合精度训练)
    loss_scaler = NativeScalerWithGradNormCount() if args.autocast else None

    # 如果从检查点恢复，加载优化器和缩放器状态
    if args.load_model:
        osd = torch.load(f'./optims/{args.model_time}/optim_{str(args.checkpoint_epoch).zfill(3)}.pth', map_location='cpu')
        optimizer.load_state_dict(osd)
        if loss_scaler is not None:
            ssd = torch.load(f'./scalers/{args.model_time}/scaler_{str(args.checkpoint_epoch).zfill(3)}.pth', map_location='cpu')
            loss_scaler.load_state_dict(ssd)

    return model, optimizer, loss_scaler


def rebuild_progressive(gpu, rank, args, opts, epoch):
    """
    渐进式学习: 重建数据加载器和模型

    在渐进式学习策略中，随着训练的进行，逐渐增大 patch size 和 batch size。
    此函数在 epoch 达到预设阈值时重建数据加载器和模型。

    渐进式学习的优势:
        - 初期使用小 patch: 训练速度快，GPU 内存占用小
        - 后期使用大 patch: 模型学习更多全局信息，提升性能

    Args:
        gpu (int): 当前 GPU 设备 ID
        rank (int): 当前进程排名
        args (Namespace): 训练参数，需要包含:
            - progressive_epoch (tuple): 渐进式学习的 epoch 阈值列表
            - training_patch_size (list): 对应的 patch size 列表
            - batch_size (list): 对应的 batch size 列表
            - total_epochs (int): 总训练 epoch 数
            - 其他模型和训练参数...
        opts (dict): 模型配置选项
        epoch (int): 当前 epoch (1-indexed)

    Returns:
        tuple: (args, train_loader, model, optimizer, loss_scaler)
            - args: 更新后的参数 (init_lr 已更新)
            - train_loader: 新的数据加载器
            - model: 新的 DDP 模型
            - optimizer: 新的优化器
            - loss_scaler: 新的梯度缩放器

    渐进式学习示例:
        progressive_epoch = (100, 200)
        training_patch_size = [48, 64, 96]
        batch_size = [64, 32, 16]

        epoch 1-100:   patch_size=48, batch_size=64
        epoch 101-200: patch_size=64, batch_size=32
        epoch 201+:    patch_size=96, batch_size=16
    """
    # 计算当前 epoch 对应的渐进式索引 (pei: progressive-epoch-index)
    try:
        # 检查当前 epoch-1 是否正好是切换点
        pei = list(args.progressive_epoch).index(epoch-1)
    except:
        # 否则找到当前 epoch 所在的区间
        for pei, temp in enumerate(args.progressive_epoch+(args.total_epochs,)):
            if (epoch) < temp: break
        pei -= 1

    # 获取当前阶段的 batch size 和 patch size
    bs = args.batch_size[pei]
    ps = args.training_patch_size[pei]

    # 更新学习率 (基于新的批大小)
    args.init_lr = 0.0004 / 64 * bs*args.world_size

    # 重建数据加载器
    train_loader = build_dataset(rank, ps, bs, args)

    # 重建模型
    if '1' not in args.model_name:
        module = import_module(f"my_model.{args.model_name.lower().replace('-', '_')}")
    else:
        module = import_module(f"my_model.{args.model_name.lower().replace('-', '_')[:-2]}")

    model = module.make_model(args, opts, pei)

    # 处理微调模式的权重加载
    if args.finetune:
        if not args.load_model:
            sd = torch.load(args.pretrain_path, map_location='cpu')
            new_sd = OrderedDict()
            for n,p in sd.items():
                new_sd[n] = p if 'to_target' not in n else model.state_dict()[n]
            for n,p in model.state_dict().items():
                new_sd[n] = p if n not in new_sd else new_sd[n]
            print(model.load_state_dict(new_sd, strict=False))

    # 如果不是第一个 epoch，从上一个 epoch 的检查点加载模型权重
    if epoch!=1:
        sd = torch.load(f'./models/{args.model_time}/model_{str(epoch-1).zfill(3)}.pth', map_location='cpu')
        new_sd = OrderedDict([(n,p) for n,p in sd.items() if 'attn_mask' not in n])
        model.load_state_dict(new_sd, strict=False)

    # warm start 模式下冻结非 to_target 层
    if args.finetune and args.warm_start and args.checkpoint_epoch <= args.warm_start_epoch:
        for n,p in model.named_parameters():
            p.requires_grad = False if 'to_target' not in n else True

    # 配置 DDP 和优化器
    model = model.to(gpu)
    model_no_ddp = model
    param_groups = param_groups_lrd(
        model_no_ddp,
        weight_decay=args.weight_decay,
        no_weight_decay_list=model_no_ddp.no_weight_decay(),
        layer_decay=args.layer_decay,
        model_name=args.model_name
    )
    model = DDP(model, device_ids=[gpu])
    optimizer = torch.optim.Adam(param_groups, lr=args.init_lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScalerWithGradNormCount() if args.autocast else None

    # 从上一个 epoch 加载优化器和缩放器状态
    if epoch!=1:
        osd = torch.load(f'./optims/{args.model_time}/optim_{str(epoch-1).zfill(3)}.pth', map_location='cpu')
        optimizer.load_state_dict(osd)
        if loss_scaler is not None:
            ssd = torch.load(f'./scalers/{args.model_time}/scaler_{str(epoch-1).zfill(3)}.pth', map_location='cpu')
            loss_scaler.load_state_dict(ssd)

    print(f'progressive learning...epoch: {epoch}...patch-size: {ps}...batch-size: {bs*args.world_size}...init-lr:{args.init_lr}')
    return args, train_loader, model, optimizer, loss_scaler


def rebuild_after_warm_start(gpu, args, model):
    """
    warm start 结束后重建模型和优化器

    在 warm start 阶段，只有 to_target 层被训练。当 warm start 结束后，
    需要解冻所有层并重新配置优化器。

    Args:
        gpu (int): 当前 GPU 设备 ID
        args (Namespace): 训练参数
        model: 当前的 DDP 模型

    Returns:
        tuple: (model, optimizer, loss_scaler)
            - model: 所有层都可训练的 DDP 模型
            - optimizer: 新的优化器
            - loss_scaler: 新的梯度缩放器

    Note:
        - 使用 model.module 获取 DDP 包装的原始模型
        - 所有参数的 requires_grad 被设置为 True
    """
    # 获取 DDP 包装的原始模型
    model = model.module.to(gpu)

    # 解冻所有层
    for n,p in model.named_parameters():
        p.requires_grad = True

    # 重新配置参数组和优化器
    model_no_ddp = model
    param_groups = param_groups_lrd(
        model_no_ddp,
        weight_decay=args.weight_decay,
        no_weight_decay_list=model_no_ddp.no_weight_decay(),
        layer_decay=args.layer_decay,
        model_name=args.model_name
    )
    model = DDP(model, device_ids=[gpu])

    optimizer = torch.optim.Adam(param_groups, lr=args.init_lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScalerWithGradNormCount() if args.autocast else None

    return model, optimizer, loss_scaler


# =============================================================================
# 损失函数构建
# =============================================================================

def build_loss_func(args):
    """
    构建损失函数

    支持三种常用的图像恢复损失函数:
        - L1 Loss: 平均绝对误差，对异常值鲁棒
        - MSE Loss: 均方误差，平滑但对异常值敏感
        - Charbonnier Loss: L1 Loss 的平滑版本，兼具两者优点

    Args:
        args (Namespace): 训练参数，需要包含:
            - criterion (str): 损失函数名称 ('L1', 'MSE', 'Charbonnier')

    Returns:
        nn.Module: 损失函数模块

    Charbonnier Loss 公式:
        L(x, y) = sqrt((x - y)^2 + eps^2)
        其中 eps 是一个小常数，使损失在 0 点可微
    """
    _assert(args.criterion in ['L1', 'MSE', 'Charbonnier'], "'criterion' should be in [L1, MSE, Charbonnier]")

    criterion_dict = {
        'L1': nn.L1Loss(),
        'MSE': nn.MSELoss(),
        'Charbonnier': CharbonnierLoss()
    }
    criterion = criterion_dict[args.criterion]

    return criterion


# =============================================================================
# 测试模型构建
# =============================================================================

def build_model_test(gpu, args, opts):
    """
    构建用于测试/推理的模型

    与训练不同，测试模型直接加载预训练权重，不需要优化器和缩放器。

    Args:
        gpu (int): GPU 设备 ID
        args (Namespace): 参数，需要包含:
            - model_name (str): 模型名称
            - pretrain_path (str): 预训练权重路径
        opts (dict): 模型配置选项

    Returns:
        nn.Module: DDP 包装的测试模型

    Note:
        - 模型以 eval 模式使用 (在调用处设置)
        - 仍然使用 DDP 包装以保持接口一致性
    """
    _assert(args.model_name in ['RAMiT', 'RAMiT-1', 'RAMiT-slimSR', 'RAMiT-slimLLE'],
            "'model_name' should be RAMiT, RAMiT-1, RAMiT-slimSR, RAMiT-slimLLE")

    # 动态导入模型模块
    if '1' not in args.model_name:
        module = import_module(f"my_model.{args.model_name.lower().replace('-', '_')}")
    else:
        module = import_module(f"my_model.{args.model_name.lower().replace('-', '_')[:-2]}")

    model = module.make_model(args, opts, 0)

    # 加载预训练权重
    sd = torch.load(args.pretrain_path, map_location='cpu')
    print(model.load_state_dict(sd, strict=False))

    # 移动到 GPU 并使用 DDP 包装
    model = model.to(gpu)
    model = DDP(model, device_ids=[gpu])

    return model
