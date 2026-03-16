"""
M4 파라미터 그룹별 카운트 스크립트

실행 방법 (프로젝트 루트에서):
    python -m classification.pruning.count_params
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from classification.model.build import EfficientViT_M4
from classification.model.efficientvit import (
    EfficientViTBlock, FFN, CascadedGroupAttention,
    LocalWindowAttention, PatchMerging, Conv2d_BN, BN_Linear, Residual
)


def numel(module):
    return sum(p.numel() for p in module.parameters())


# =========================================================
# 모델 생성
# =========================================================
model = EfficientViT_M4(num_classes=1000)
model.eval()

groups = {
    'G_FFN':     0,   # FFN expand+shrink (전체 블록)
    'G_QK':      0,   # CGA Q+K projection + Q DWConv
    'G_V':       0,   # CGA V projection
    'W_out':     0,   # CGA Output Projection (W'')
    'Attn_Bias': 0,   # CGA 학습 가능 위치 편향
    'G_INV':     0,   # Subsample 1×1 expand+reduce (PatchMerging conv1+conv3)
    'PM_DW':     0,   # PatchMerging DWConv (conv2, stride=2)
    'PM_SE':     0,   # PatchMerging SqueezeExcite
    'G_PE1':     0,   # PatchEmbed Conv1 (3 → C/8)
    'G_PE2':     0,   # PatchEmbed Conv2 (C/8 → C/4)
    'G_PE3':     0,   # PatchEmbed Conv3 (C/4 → C/2)
    'G_PE4':     0,   # PatchEmbed Conv4 (C/2 → C)
    'DWConv':    0,   # Token Interaction DWConv (EVBlock dw0, dw1)
    'Head':      0,   # Classifier (BN + Linear)
}


# =========================================================
# CGA 파라미터 분리 (Q+K / V / OutProj)
# =========================================================
def process_cga(cga: CascadedGroupAttention):
    key_dim = cga.key_dim
    d       = cga.d
    total_out = key_dim * 2 + d   # QKV combined output channels per head

    for i in range(cga.num_heads):
        qkv = cga.qkvs[i]  # Conv2d_BN(C/h, key_dim*2+d)
        dw  = cga.dws[i]   # Conv2d_BN for Q DWConv

        # --- Conv weight 분리 ---
        # weight shape: [total_out, C/h, 1, 1]
        conv_total = qkv.c.weight.numel()
        qk_conv    = int(conv_total * (key_dim * 2) / total_out)
        v_conv     = conv_total - qk_conv

        # --- BN (weight + bias) 분리 ---
        bn_total = qkv.bn.weight.numel() + qkv.bn.bias.numel()  # total_out * 2
        qk_bn    = int(bn_total * (key_dim * 2) / total_out)
        v_bn     = bn_total - qk_bn

        groups['G_QK'] += qk_conv + qk_bn + numel(dw)   # Q+K proj + Q DWConv
        groups['G_V']  += v_conv + v_bn

    # Output projection: Sequential(ReLU, Conv2d_BN(d*h, dim))
    groups['W_out']     += numel(cga.proj)
    groups['Attn_Bias'] += cga.attention_biases.numel()


# =========================================================
# EfficientViTBlock 처리
# =========================================================
def process_ev_block(block: EfficientViTBlock):
    groups['DWConv'] += numel(block.dw0)    # Residual(Conv2d_BN)
    groups['DWConv'] += numel(block.dw1)
    groups['G_FFN']  += numel(block.ffn0)   # Residual(FFN)
    groups['G_FFN']  += numel(block.ffn1)

    lwa: LocalWindowAttention = block.mixer.m
    process_cga(lwa.attn)


# =========================================================
# SubsampleBlock (Sequential) 처리
# blocks2[0], blocks2[2], blocks3[0], blocks3[2]
# → Sequential( Residual(Conv2d_BN DWConv), Residual(FFN) )
# =========================================================
def process_subblock(seq: torch.nn.Sequential):
    for residual in seq:
        inner = residual.m
        if isinstance(inner, FFN):
            groups['G_FFN'] += numel(inner)
        elif isinstance(inner, Conv2d_BN):
            # Depthwise Conv (groups == channels)
            groups['DWConv'] += numel(inner)


# =========================================================
# PatchMerging 처리
# =========================================================
def process_patch_merging(pm: PatchMerging):
    groups['G_INV']  += numel(pm.conv1)   # 1×1 확장 Conv
    groups['PM_DW']  += numel(pm.conv2)   # 3×3 DWConv stride=2
    groups['PM_SE']  += numel(pm.se)      # SqueezeExcite
    groups['G_INV']  += numel(pm.conv3)   # 1×1 축소 Conv


# =========================================================
# PatchEmbed (Conv2d_BN 4개 + ReLU 3개)
# =========================================================
pe_convs = [m for m in model.patch_embed if isinstance(m, Conv2d_BN)]
groups['G_PE1'] = numel(pe_convs[0])
groups['G_PE2'] = numel(pe_convs[1])
groups['G_PE3'] = numel(pe_convs[2])
groups['G_PE4'] = numel(pe_convs[3])

# =========================================================
# Classifier Head
# =========================================================
groups['Head'] = numel(model.head)

# =========================================================
# blocks1, blocks2, blocks3 순회
# =========================================================
for block in model.blocks1:
    process_ev_block(block)

for block in model.blocks2:
    btype = type(block).__name__
    if btype == 'EfficientViTBlock':
        process_ev_block(block)
    elif btype == 'PatchMerging':
        process_patch_merging(block)
    elif btype == 'Sequential':
        process_subblock(block)

for block in model.blocks3:
    btype = type(block).__name__
    if btype == 'EfficientViTBlock':
        process_ev_block(block)
    elif btype == 'PatchMerging':
        process_patch_merging(block)
    elif btype == 'Sequential':
        process_subblock(block)


# =========================================================
# 결과 출력
# =========================================================
total_model    = sum(p.numel() for p in model.parameters())
total_computed = sum(groups.values())

labels = {
    'G_FFN':     'FFN expand+shrink (전체 블록)',
    'G_QK':      'CGA Q+K proj + Q DWConv (전체 head)',
    'G_V':       'CGA V projection (전체 head)',
    'W_out':     'CGA Output Projection (W\'\')',
    'Attn_Bias': 'CGA 학습 가능 위치 편향',
    'G_INV':     'Subsample 1×1 expand+reduce',
    'PM_DW':     'PatchMerging DWConv (stride=2)',
    'PM_SE':     'PatchMerging SqueezeExcite',
    'G_PE1':     'PatchEmbed Conv1  (3 → C/8)',
    'G_PE2':     'PatchEmbed Conv2  (C/8 → C/4)',
    'G_PE3':     'PatchEmbed Conv3  (C/4 → C/2)',
    'G_PE4':     'PatchEmbed Conv4  (C/2 → C)',
    'DWConv':    'Token Interaction DWConv (dw0+dw1)',
    'Head':      'Classifier (BN + Linear)',
}

print()
print("=" * 68)
print(f"  EfficientViT M4 — 그룹별 파라미터 수")
print("=" * 68)
print(f"  {'그룹':<12}  {'대상':<38}  {'개수':>9}  {'비율':>5}  {'MB':>6}")
print("-" * 68)

prunable_keys = {'G_FFN', 'G_QK', 'G_V'}

for key, label in labels.items():
    v   = groups[key]
    pct = v / total_model * 100
    mb  = v * 4 / 1e6
    tag = " ◀ prunable" if key in prunable_keys else ""
    print(f"  {key:<12}  {label:<38}  {v:>9,}  {pct:>4.1f}%  {mb:>5.2f}{tag}")

print("-" * 68)

unaccounted = total_model - total_computed
print(f"  {'합계 (계산)':<52}  {total_computed:>9,}  100.0%  {total_computed*4/1e6:>5.2f}")
if unaccounted != 0:
    print(f"  {'⚠ 미분류':<52}  {unaccounted:>9,}  {unaccounted/total_model*100:>4.1f}%")
print(f"  {'합계 (모델)':<52}  {total_model:>9,}  100.0%  {total_model*4/1e6:>5.2f}")
print("=" * 68)

# =========================================================
# Pruning 관점 요약
# =========================================================
prunable    = groups['G_FFN'] + groups['G_QK'] + groups['G_V']
fixed       = total_model - prunable
max_opt_rate = (total_model - fixed) / total_model * 100

print()
print("=" * 68)
print("  Pruning 관점 요약")
print("=" * 68)
print(f"  Prunable (G_FFN + G_QK + G_V):  {prunable:>9,}  ({prunable*4/1e6:.2f} MB)")
print(f"  Fixed (나머지):                  {fixed:>9,}  ({fixed*4/1e6:.2f} MB)")
print(f"  전체:                            {total_model:>9,}  ({total_model*4/1e6:.2f} MB)")
print()
print(f"  FFN+QK+V 전부 제거 시 최대 압축률:  {max_opt_rate:.1f}%")
print(f"  FFN+QK만 제거 시 최대 압축률:       "
      f"{(groups['G_FFN']+groups['G_QK'])/total_model*100:.1f}%")
print()

# FFN을 X% 제거했을 때 전체 압축률
for ffn_prune in [0.50, 0.60, 0.70, 0.76, 0.80, 0.90]:
    remaining  = total_model - groups['G_FFN'] * ffn_prune - groups['G_QK'] * 0.70
    opt_rate   = (total_model - remaining) / total_model * 100
    remain_mb  = remaining * 4 / 1e6
    print(f"  FFN {ffn_prune*100:.0f}% + QK 70% 제거 → "
          f"남은 크기 {remain_mb:.2f} MB  (압축률 {opt_rate:.1f}%)")

print("=" * 68)


if __name__ == '__main__':
    pass
