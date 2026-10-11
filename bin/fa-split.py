#!/usr/bin/env python3
"""Pinned llama.cpp FA launch model; standard library, no CUDA runtime required.

Assumes one sequence, equal DKQ/DV, K and V of the same type, an aligned
(non-pruning) FP16 mask, max_bias=softcap=0, FA enabled, all quantised KV
instances enabled, and native compiled architectures for each requested cc.
Pinned llama.cpp: 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6.
Resource CSV is from nvcc 12.8 --use_fast_math --extended-lambda -std=c++17 -arch=sm_XX
-cubin, with ggml/include and ggml/src include paths; softcap=false instances.
VEC is FP32 on NVIDIA; np reports its warp/V-column accumulator groups.
Its actual key loop uses 128 keys even when launch_fattn receives nbatch_fa=D.
 'other' means the selected Volta MMA kernel, which
is outside this Pascal accumulator model and is never compiled here.
"""
import argparse
import csv
import io
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / 'docs/research/fa-tile-resources-sm60-sm70-2026-10-10.csv'
FIELDS = ('cc,nsm,d,heads_q,heads_kv,n_q,seq_kv,kv_type,kernel,ncols1,ncols2,'
          'nwarps,np,nbatch_fa,occupancy,ntiles_dst,ntiles_KV,parallel_blocks,'
          'keys_per_fp16_accumulator,accumulator_dtype,resource_source').split(',')

# fattn-tile.cuh:21-107: (threads, launch-bounds minimum blocks, batch_FA, batch_K)
# Entries are ordered by total Q columns 2,4,8,16,32; DKQ=DV only.
CONFIGS = {
    40: [(128,3,128,40),(128,2,128,40),(128,3,128,40),(256,2,64,40),(256,2,64,40)],
    64: [(64,2,64,64),(128,2,64,64),(256,3,128,64),(256,2,64,64),(256,2,64,64)],
    72: [(128,2,64,72),(128,3,128,72),(256,2,64,72),(256,2,64,72),(256,2,64,72)],
    80: [(128,2,64,40),(128,2,64,40),(256,2,64,40),(256,2,64,40),(256,2,128,40)],
    96: [(64,2,64,48),(128,2,64,48),(256,2,64,48),(256,2,64,48),(256,2,128,24)],
    112: [(128,3,64,56),(128,2,64,56),(256,3,32,112),(256,2,32,112),(256,2,128,56)],
    128: [(64,2,64,64),(128,2,64,64),(256,2,64,64),(256,2,64,64),(256,2,64,64)],
    256: [(64,2,64,64),(128,2,64,64),(256,2,64,64),(256,2,64,64),(256,2,64,64)],
}


def ceildiv(a, b):
    return (a + b - 1) // b


def resources():
    if not RESOURCES.exists():
        return {}
    with RESOURCES.open(newline='') as stream:
        return {(int(r['cc']), r['kernel'], int(r['d']), int(r['ncols1']),
                 int(r['ncols2']), r['kv_type']): r for r in csv.DictReader(stream)}


def decide(cc, nsm, d, heads_q, heads_kv, n_q, seq_kv, kv_type, measured=None):
    if d not in CONFIGS or min(nsm, heads_q, heads_kv, n_q, seq_kv) < 1 or heads_q % heads_kv:
        raise ValueError('positive sizes, divisible heads and d in ' + str(sorted(CONFIGS)) + ' required')
    gqa = heads_q // heads_kv
    # fattn.cu:559-572; fattn-common.cuh:9: mask/alignment assumptions above.
    gqa_opt = gqa >= 2 and seq_kv % 256 == 0
    # fattn.cu:635,667-671: effective GQA is the largest power-of-two divisor <=8.
    vector_ok = d % 64 == 0 and seq_kv % 256 == 0
    gqa_eff = 1
    while gqa % (2 * gqa_eff) == 0 and gqa_eff < 8:
        gqa_eff *= 2
    # common.cuh:360-362; fattn.cu:673-680: native Volta build assumption.
    if cc == 700 and d not in (40, 72):
        kernel = ('VEC' if vector_ok and n_q * gqa_eff <= 2 else
                  'TILE' if n_q * gqa_eff <= 16 else 'other')
    # fattn.cu:705-719: Pascal quantised decode uses VEC at <=2 Q columns.
    else:
        kernel = ('VEC' if vector_ok and ((kv_type != 'f16' and n_q <= 2) or
                                         (kv_type == 'f16' and n_q == 1 and not gqa_opt)) else 'TILE')
    row = dict(zip(FIELDS[:9], (cc, nsm, d, heads_q, heads_kv, n_q, seq_kv, kv_type, kernel)))
    if kernel == 'other':
        row.update({field: 0 for field in FIELDS[9:19]})
        row.update(accumulator_dtype='not_modeled', resource_source='not_applicable_MMA_F16')
        return row
    if kernel == 'TILE':
        # fattn-tile.cuh:1242-1257,1298-1317: NVIDIA GQA limit for ratios <=4.
        use_gqa = seq_kv % 256 == 0 and (gqa > 4 or n_q <= 16)
        ncols2 = next((x for x in (8, 4, 2) if use_gqa and gqa % x == 0), 1)
        # fattn-tile.cuh:1172-1231: strict Q-column thresholds, DKQ<=256.
        ncols = next((x for x in (32, 16, 8, 4) if ncols2 <= x and n_q > (x // 2) // ncols2), 2)
        ncols1 = ncols // ncols2
        threads, bounds, nbatch, batch_k = CONFIGS[d][int(math.log2(ncols)) - 1]
        # fattn-tile.cuh:584,872: parallel warps per Q column.
        np = max(1, (threads // 32) // ncols)
        # common.cuh:399-409; fattn-tile.cuh:885-887: static shared arrays.
        cpy_ne = 2 if cc == 600 else 4
        padded_d = ceildiv(d, 64) * 64
        smem = ncols*d*2 + (nbatch*(batch_k//2 + cpy_ne) + padded_d-d)*4 + ncols*nbatch*2
        resource_type = 'f16'  # launch_fattn converts quantised KV: fattn-tile.cuh:1182.
        dtype = 'f16'
    else:
        # fattn-vec.cuh:4-10,535-541,552-566: 128 threads, one/two Q columns.
        ncols1, ncols2, threads, bounds, nbatch = min(n_q, 2), 1, 128, 1, d
        # fattn-vec.cuh:78-96,323-325: 4 warps times V column groups.
        nthreads_v = (16 if cc == 600 else 8) if kv_type == 'f16' else min(d//4, 32)
        np = (threads // 32) * (32 // nthreads_v)
        # fattn-vec.cuh:123-130,412-413; common.cuh:760-762: FP32 on NVIDIA.
        smem = max(ncols1*d, np*d)*4 + 2*ncols1*32*4
        resource_type, dtype = kv_type, 'f32'
    data = (resources() if measured is None else measured).get((cc, kernel, d, ncols1, ncols2, resource_type))
    # fattn-common.cuh:1136-1139: runtime occupancy, approximated by static resources.
    # CUDA Pascal/Volta limits: 2048 threads, 32 blocks, 65536 registers/SM;
    # cuda_occupancy.h (CUDA 12.8):613-628,671-694,700-720,1484-1539:
    # registers allocated in 256-register units per warp across 2 (sm_60) or
    # 4 (sm_70) subpartitions; shared allocated in 256-byte units. Assumes
    # partitioned global caching off and zero driver-reserved shared memory.
    # fattn-tile.cuh:792; fattn-vec.cuh:20: launch bounds are a compiler
    # MINIMUM, not an occupancy ceiling. When measured
    # registers exist, do not cap the predicted occupancy by that minimum.
    if data:
        smem = int(data['static_smem_bytes'])
        regs_per_warp = ceildiv(int(data['registers_per_thread'])*32, 256)*256
        subpartitions = 2 if cc == 600 else 4
        warps_per_partition = (65536//subpartitions)//regs_per_warp
        reg_limit = (warps_per_partition*subpartitions)//(threads//32)
        assumed_regs = regs_per_warp * ceildiv(threads//32, subpartitions)*subpartitions
        if assumed_regs > 65536 or int(data['registers_per_thread']) > (255 if cc == 600 else 256):
            reg_limit = 0
        source = 'measured_nvcc12.8_static_prediction'
    else:
        reg_limit = bounds
        source = 'fallback_launch_bounds_no_measured_registers'
    per_sm_smem = (64 if cc == 600 else 96)*1024
    occupancy = min(reg_limit, 2048//threads, 32, per_sm_smem//(ceildiv(smem,256)*256))
    if smem > 48*1024 or occupancy < 1:
        raise ValueError('configuration exceeds shared memory/block or has zero occupancy')
    # fattn-common.cuh:1095-1098: single sequence destination tiles.
    ntiles_dst = ceildiv(n_q, ncols1)*ceildiv(gqa, ncols2)*heads_kv
    # fattn-common.cuh:1141-1142: note VEC passes D although its loop steps by 128.
    ntiles_kv = ceildiv(seq_kv, nbatch)
    parallel = min(occupancy, ntiles_kv)
    # fattn-common.cuh:1192-1215: integer-percent wave-efficiency search.
    blocks_per_wave = nsm*occupancy
    best_eff, best_waves = 0, 0
    for candidate in range(parallel, ntiles_kv+1):
        blocks = ntiles_dst*candidate
        waves = ceildiv(blocks, blocks_per_wave)
        efficiency = 100*blocks//(waves*blocks_per_wave)
        if best_eff >= 95 and waves > best_waves:
            break
        if efficiency > best_eff:
            best_eff, best_waves, parallel = efficiency, waves, candidate
    # fattn-tile.cuh:727-753,954-977: cyclic KV tiles and j%np warp routing.
    # Count tail keys explicitly; do not divide total keys by the split approximately.
    counts = [0]*(parallel*np)
    if dtype == 'f16':
        for tile in range(ntiles_kv):
            keys = min(nbatch, seq_kv-tile*nbatch)
            for lane in range(np):
                counts[(tile % parallel)*np+lane] += max(0, ceildiv(keys-lane, np))
    row.update(ncols1=ncols1, ncols2=ncols2, nwarps=threads//32, np=np,
               nbatch_fa=nbatch, occupancy=occupancy, ntiles_dst=ntiles_dst,
               ntiles_KV=ntiles_kv, parallel_blocks=parallel,
               keys_per_fp16_accumulator=max(counts), accumulator_dtype=dtype, resource_source=source)
    return row


def table(cc=None, nsm=None):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader()
    measured = resources()
    for arch in ([cc] if cc else [600, 700]):
        for d, hq, hkv in ((128,32,8),(256,24,4),(128,40,8)):
            for nq in (1,2,4,8,16,512):
                for nk in (4096,16384,32768):
                    for typ in ('f16','q8_0'):
                        writer.writerow(decide(arch,(56 if arch==600 else 80) if nsm is None else nsm,d,hq,hkv,nq,nk,typ,measured))
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', type=int, choices=(600,700))
    parser.add_argument('--nsm', type=int)
    for name in ('d','heads-q','heads-kv','n-q','seq-kv'):
        parser.add_argument('--'+name, type=int, help='equal DKQ/DV in '+str(sorted(CONFIGS)) if name == 'd' else None)
    parser.add_argument('--kv-type', choices=('f16','q8_0','q4_0'))
    parser.add_argument('--table', action='store_true')
    parser.add_argument('--check', type=Path, help='with --table: compare generated CSV byte for byte')
    args = parser.parse_args()
    try:
        if args.table:
            result = table(args.cc, args.nsm)
            if args.check:
                if args.check.read_bytes() != result.encode():
                    raise ValueError('generated split CSV differs from ' + str(args.check))
                print('PASS fa_split_table (216 rows)' if not args.cc else 'PASS fa_split_table')
            else:
                sys.stdout.write(result)
        else:
            if args.check or any(getattr(args, name) is None for name in ('cc','d','heads_q','heads_kv','n_q','seq_kv','kv_type')):
                parser.error('--cc --d --heads-q --heads-kv --n-q --seq-kv --kv-type required')
            row = decide(args.cc,(56 if args.cc==600 else 80) if args.nsm is None else args.nsm,args.d,args.heads_q,args.heads_kv,args.n_q,args.seq_kv,args.kv_type)
            print(' '.join(f'{key}={row[key]}' for key in FIELDS))
            print('assumptions: one sequence; aligned present non-pruning mask; max_bias=softcap=0; native architecture; static occupancy prediction (GPU runtime not verified)')
    except (ValueError, OSError) as error:
        parser.exit(1, str(error)+'\n')


if __name__ == '__main__':
    main()
