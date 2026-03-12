"""
Phase A: EfficientViT M4 Pruning Group Profiler

모든 pruning 그룹을 추출하고 메모리/파라미터 통계를 계산합니다.
결과는 콘솔 테이블과 JSON 리포트로 저장됩니다.

CLAUDE.md §8 Phase A 성공 기준:
  "Profile all layers, build group dict → Memory map complete"

Usage (서버):
  # CPU (GPU 불필요)
  cd /path/to/EfficientVIT_Compression
  python -m classification.pruning.phase_a_profile

  # GPU + pretrained weights
  python -m classification.pruning.phase_a_profile \\
      --device cuda \\
      --pretrained EfficientViT_M4 \\
      --output classification/pruning/reports/phase_a_report.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from classification.model.build import EfficientViT_M4
from classification.pruning.group_dict import build_pruning_groups, get_phase1_groups
from classification.pruning.memory_utils import profile_gpu_memory, count_zero_groups


# ─────────────────────────────────────────────────────────────────────────────
# Per-unit parameter counting (analytical)
# ─────────────────────────────────────────────────────────────────────────────

def _params_per_unit(g):
    """
    그룹 1 unit 제거 시 절약되는 파라미터 수를 해석적으로 계산합니다.
    Conv weight만 계산 (BN은 매우 작아서 무시).

    Returns
    -------
    (params_per_unit, total_group_params)
    """
    t = g['type']
    m = g['modules']
    meta = g['meta']

    if t == 'G_FFN':
        ed  = meta['ed']
        hid = meta['hidden_dim']
        # expand: [hid, ed, 1, 1]  — unit k 제거 → row k (ed params)
        # shrink: [ed, hid, 1, 1]  — unit k 제거 → col k (ed params)
        per_unit    = ed + ed
        total_group = (m['expand'].c.weight.numel() +
                       m['shrink'].c.weight.numel() +
                       m['expand'].bn.weight.numel() * 2 +
                       m['shrink'].bn.weight.numel() * 2)

    elif t == 'G_QK':
        in_ch   = meta['in_channels']
        key_dim = meta['key_dim']
        kH = m['dw'].c.weight.shape[2]
        kW = m['dw'].c.weight.shape[3]
        # qkv Q: in_ch params per dim  +  K: in_ch params per dim
        # dw:  kH*kW params per dim (DW conv, groups=key_dim)
        per_unit    = 2 * in_ch + kH * kW
        total_group = (m['qkv'].c.weight.numel() +
                       m['qkv'].bn.weight.numel() * 2 +
                       m['dw'].c.weight.numel() +
                       m['dw'].bn.weight.numel() * 2)

    elif t == 'G_V':
        in_ch = meta['in_channels']
        # qkv V: in_ch params per channel
        per_unit    = in_ch
        total_group = (m['qkv'].c.weight.numel() +
                       m['qkv'].bn.weight.numel() * 2)

    elif t == 'G_PATCH':
        w_out = m['out_conv'].c.weight   # [out_ch, in_ch, kH, kW]
        w_in  = m['in_conv'].c.weight    # [out_ch_next, in_ch_next, kH2, kW2]
        in_ch = w_out.shape[1]
        kH = w_out.shape[2]; kW = w_out.shape[3]
        out_ch_next = w_in.shape[0]
        kH2 = w_in.shape[2]; kW2 = w_in.shape[3]
        # 필터 1개 제거: out_conv row + in_conv col
        per_unit    = in_ch * kH * kW + out_ch_next * kH2 * kW2
        total_group = (m['out_conv'].c.weight.numel() +
                       m['out_conv'].bn.weight.numel() * 2)
    else:
        per_unit = 0; total_group = 0

    return per_unit, total_group


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Phase A: Profile EfficientViT M4 pruning groups')
    parser.add_argument('--device', default='cpu',
                        help='Device: cpu or cuda (default: cpu)')
    parser.add_argument('--pretrained', default='',
                        help='Pretrained weight name (e.g. EfficientViT_M4). '
                             '비워두면 랜덤 초기화 (구조 분석에는 충분)')
    parser.add_argument('--output',
                        default='classification/pruning/reports/phase_a_report.json',
                        help='JSON 리포트 저장 경로')
    args = parser.parse_args()

    device = torch.device(args.device)
    W = 90  # 테이블 너비

    print(f'\n{"="*W}')
    print('Phase A — EfficientViT M4 Pruning Group Profiler')
    print(f'{"="*W}\n')

    # ── 모델 로드 ────────────────────────────────────────────────────────────
    pretrained = args.pretrained if args.pretrained else False
    model = EfficientViT_M4(pretrained=pretrained)
    model = model.to(device).eval()

    total_p = sum(p.numel() for p in model.parameters())
    total_mb = total_p * 4 / 1e6
    print(f'  Model    : EfficientViT_M4')
    print(f'  Pretrained: {args.pretrained if args.pretrained else "No (random init)"}')
    print(f'  Device   : {device}')
    print(f'  Total params : {total_p:,}  ({total_mb:.2f} MB)\n')

    # ── 그룹 추출 ────────────────────────────────────────────────────────────
    groups = build_pruning_groups(model)
    print(f'  Total pruning groups : {len(groups)}')
    print(f'  Phase 1 groups (G_FFN+G_QK+G_V): {len(get_phase1_groups(groups))}\n')

    # ── 그룹별 통계 계산 ─────────────────────────────────────────────────────
    report_groups = []
    type_agg = {t: {'n_groups': 0, 'total_units': 0, 'total_params': 0}
                for t in ['G_FFN', 'G_QK', 'G_V', 'G_PATCH']}

    hdr = f'{"Group ID":<48} {"Type":<8} {"Units":>6} {"Params/Unit":>11} {"Group Params":>13} {"Mem KB":>8}'
    print(f'{"─"*W}')
    print(hdr)
    print(f'{"─"*W}')

    for g in groups:
        per_unit, total_gp = _params_per_unit(g)
        mem_kb = total_gp * 4 / 1024
        note   = '  ← Phase2+' if g['type'] == 'G_PATCH' else ''

        print(f'{g["id"]:<48} {g["type"]:<8} {g["unit_count"]:>6,} '
              f'{per_unit:>11,} {total_gp:>13,} {mem_kb:>8.1f}{note}')

        t = g['type']
        type_agg[t]['n_groups']    += 1
        type_agg[t]['total_units'] += g['unit_count']
        type_agg[t]['total_params']+= total_gp

        report_groups.append({
            'id':           g['id'],
            'type':         t,
            'lambda_rec':   g['lambda_rec'],
            'unit_count':   g['unit_count'],
            'params_per_unit': per_unit,
            'group_params': total_gp,
            'group_mem_kb': round(mem_kb, 2),
            'meta': {
                k: (v if not isinstance(v, slice) else f'{v.start}:{v.stop}')
                for k, v in g['meta'].items()
            },
        })

    print(f'{"─"*W}\n')

    # ── Type별 요약 ──────────────────────────────────────────────────────────
    lambda_map = {'G_FFN': 0.005, 'G_QK': 0.010, 'G_V': 0.001, 'G_PATCH': 0.005}
    print('Summary by Group Type:')
    print(f'{"─"*70}')
    print(f'{"Type":<12} {"Groups":>7} {"Total Units":>12} {"Total Params":>14} '
          f'{"Mem MB":>8} {"λ_rec":>7}')
    print(f'{"─"*70}')
    prunable_p1_params = 0
    for t, agg in type_agg.items():
        mem_mb = agg['total_params'] * 4 / 1e6
        lam    = lambda_map[t]
        note   = '  (Phase2+)' if t == 'G_PATCH' else ''
        print(f'{t:<12} {agg["n_groups"]:>7} {agg["total_units"]:>12,} '
              f'{agg["total_params"]:>14,} {mem_mb:>8.3f} {lam:>7.3f}{note}')
        if t != 'G_PATCH':
            prunable_p1_params += agg['total_params']

    print(f'{"─"*70}\n')

    # ── 압축 가능성 분석 ─────────────────────────────────────────────────────
    print('Compression Analysis:')
    print(f'  Phase 1 prunable params (G_FFN+G_QK+G_V) : {prunable_p1_params:>12,}  '
          f'({100*prunable_p1_params/total_p:.1f}% of total)')
    target_params = int(total_p * 0.24)                # 76% compression → 24% remain
    target_mb     = target_params * 4 / 1e6
    print(f'  76% optimization target → A ≤ {target_params:,} params  ({target_mb:.2f} MB)')
    print(f'  M_max (μ penalty threshold) = {target_mb:.4f} MB\n')

    # CLAUDE.md §5.3: 권장 하이퍼파라미터 요약
    print('Recommended Hyperparameters (CLAUDE.md §5.3):')
    print(f'  λ_FFN = 0.005   λ_QK = 0.010   λ_V = 0.001')
    print(f'  μ     = 1.0     M_max = {target_mb:.4f} MB')
    print(f'  lr(η) = 1e-4    epochs = 100 (Phase B ablation)\n')

    # ── λ 진단 기준 ──────────────────────────────────────────────────────────
    print('λ Diagnosis (CLAUDE.md §5.3):')
    print('  After 10 epochs, zero-group ratio should be 0–10%.')
    print('  If >20% → λ is too large, reduce it.\n')

    # ── GPU 메모리 프로파일링 ─────────────────────────────────────────────────
    gpu_info = None
    if device.type == 'cuda':
        print('GPU Memory Profiling (CLAUDE.md §7):')
        gpu_info = profile_gpu_memory(model, device)
        print(f'  Parameter memory  : {gpu_info["param_memory_bytes"]/1e6:.2f} MB')
        print(f'  Peak fwd memory   : {gpu_info["peak_forward_bytes"]/1e6:.2f} MB')
        print(f'  Activation memory : {gpu_info["activation_bytes"]/1e6:.2f} MB\n')

    # ── 파라미터 중요도 요약 (CLAUDE.md §9) ─────────────────────────────────
    print('Parameter Importance (CLAUDE.md §9):')
    print('  REMOVE aggressively : Q / K  (λ=0.010, small d_qk=16)')
    print('  REMOVE moderately   : FFN hidden  (λ=0.005)')
    print('  PRESERVE mostly     : V  (λ=0.001)')
    print('  NEVER REMOVE        : Output proj')
    print('  PHASE 2+ only       : PatchEmbed intermediate, Subsample channels\n')

    # ── JSON 리포트 저장 ──────────────────────────────────────────────────────
    report = {
        'phase':            'A',
        'model':            'EfficientViT_M4',
        'pretrained':       args.pretrained if args.pretrained else False,
        'total_params':     total_p,
        'total_param_mb':   round(total_mb, 4),
        'n_groups_total':   len(groups),
        'n_groups_phase1':  len(get_phase1_groups(groups)),
        'type_summary':     {
            t: {**agg, 'mem_mb': round(agg['total_params']*4/1e6, 4), 'lambda_rec': lambda_map[t]}
            for t, agg in type_agg.items()
        },
        'compression': {
            'target_opt_rate':   0.76,
            'target_param_count': target_params,
            'target_param_mb':   round(target_mb, 4),
            'm_max_mb':          round(target_mb, 4),
        },
        'hyperparams_rec': {
            'lambda_ffn': 0.005,
            'lambda_qk':  0.010,
            'lambda_v':   0.001,
            'mu':         1.0,
            'lr':         1e-4,
            'epochs_phase_b': 100,
        },
        'gpu_memory': (
            {k: round(v/1e6, 4) for k, v in gpu_info.items()} if gpu_info else None
        ),
        'groups': report_groups,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f'{"="*W}')
    print(f'  Phase A Complete.')
    print(f'  Report saved → {out_path}')
    print(f'{"="*W}\n')


if __name__ == '__main__':
    main()
