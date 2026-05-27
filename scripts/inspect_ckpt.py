#!/usr/bin/env python3
"""快速检查 checkpoint 的 key 结构（用于定位加载失败原因）"""
import sys, torch

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/Timer_forecast_1.0.ckpt"
sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# 如果是完整 ckpt（包含 optimizer、epoch 等），只取 state_dict
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]

print(f"总 key 数量: {len(sd)}")

# 分析第一层前缀分布
prefixes = {}
for k in sd.keys():
    top = k.split(".")[0]
    prefixes[top] = prefixes.get(top, 0) + 1

print("\n第一层前缀分布:")
for p, cnt in sorted(prefixes.items(), key=lambda x: -x[1]):
    print(f"  '{p}': {cnt} 个 key")

# 展示每种前缀下的前5个 key
print("\n各前缀下的 key 示例:")
seen = set()
for k in sorted(sd.keys()):
    top = k.split(".")[0]
    if top not in seen:
        seen.add(top)
        print(f"\n  [{top}] 前缀下的前3个 key:")
        cnt = 0
        for k2 in sorted(sd.keys()):
            if k2.split(".")[0] == top:
                print(f"    {k2}")
                cnt += 1
                if cnt >= 3:
                    break
