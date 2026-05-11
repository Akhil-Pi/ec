#!/usr/bin/env python3
"""
演示脚本：测试新的 tilt → rotate 逻辑

这个脚本展示在各种场景下，系统如何：
1. 计算头部倾斜角度
2. 决定腕部旋转方向
3. 执行干预策略
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from pss_calculator import PSSCalculator
from intervention_policy import InterventionPolicy
import config


def create_mock_landmarks(head_offset_x=0, shoulder_width=0.15,
                         trunk_angle=0, forward_lean=False):
    """
    创建模拟的 MediaPipe landmarks。

    Args:
        head_offset_x: 头部相对肩膀中点的水平偏移（正值=向右）
        shoulder_width: 肩膀宽度（归一化坐标）
        trunk_angle: 躯干倾斜角（度数）
        forward_lean: 是否向前倾斜
    """
    shoulder_mid_x = 0.5
    shoulder_mid_y = 0.6

    # 应用躯干倾斜
    import math
    angle_rad = math.radians(trunk_angle)
    hip_offset_x = math.sin(angle_rad) * 0.1

    # 格式: (x, y, z, visibility)
    return {
        "LEFT_EAR": (shoulder_mid_x - shoulder_width/2 + head_offset_x,
                     shoulder_mid_y - 0.2, 0.1, 0.95),
        "RIGHT_EAR": (shoulder_mid_x + shoulder_width/2 + head_offset_x,
                      shoulder_mid_y - 0.2, 0.1, 0.95),
        "LEFT_SHOULDER": (shoulder_mid_x - shoulder_width/2,
                          shoulder_mid_y, 0.0, 0.95),
        "RIGHT_SHOULDER": (shoulder_mid_x + shoulder_width/2,
                           shoulder_mid_y, 0.0, 0.95),
        "LEFT_HIP": (shoulder_mid_x - shoulder_width/2 + hip_offset_x,
                     shoulder_mid_y + 0.25, -0.1, 0.95),
        "RIGHT_HIP": (shoulder_mid_x + shoulder_width/2 + hip_offset_x,
                      shoulder_mid_y + 0.25, -0.1, 0.95),
        "NOSE": (shoulder_mid_x + head_offset_x,
                 shoulder_mid_y - 0.25, 0.05, 0.95),
        "LEFT_ELBOW": (shoulder_mid_x - 0.15, shoulder_mid_y + 0.1, 0.0, 0.9),
        "RIGHT_ELBOW": (shoulder_mid_x + 0.15, shoulder_mid_y + 0.1, 0.0, 0.9),
    }


def test_scenario(name, head_offset_x, trunk_angle, description):
    """
    测试一个特定场景。
    """
    print(f"\n{'='*70}")
    print(f"场景: {name}")
    print(f"说明: {description}")
    print(f"{'='*70}")

    # 创建计算器和策略
    pss_calc = PSSCalculator()
    policy = InterventionPolicy(condition="experimental")

    # 创建模拟 landmarks
    landmarks = create_mock_landmarks(head_offset_x=head_offset_x,
                                     trunk_angle=trunk_angle)

    # 计算 PSS
    pss_result = pss_calc.compute(landmarks)

    print(f"\n📊 PSS 计算结果:")
    print(f"  - 头部倾斜角: {pss_result['tilt_angle_deg']:+.1f}°")
    print(f"  - 倾斜分数: {pss_result['tilt_score']:.3f}")
    print(f"  - 躯干分数: {pss_result['trunk_score']:.3f}")
    print(f"  - PSS (平滑): {pss_result['pss_smooth']:.3f}")

    # 计算干预（直接测试，不依赖PSS阈值）
    print(f"\n🤖 干预计算（基于当前PSS分数）:")
    interventions = policy._compute_interventions(pss_result)

    if interventions:
        for action, magnitude in interventions:
            if action == "rotate":
                direction = "RIGHT (逆时针)" if magnitude > 0 else "LEFT (顺时针)"
                print(f"  ✓ {action.upper()}: {magnitude:+.3f} rad → {direction}")
            elif action == "raise":
                direction = "UP" if magnitude > 0 else "DOWN"
                print(f"  ✓ {action.upper()}: {magnitude:+.3f} m → {direction}")
            elif action == "forward":
                direction = "FORWARD" if magnitude > 0 else "BACKWARD"
                print(f"  ✓ {action.upper()}: {magnitude:+.3f} m → {direction}")
    else:
        print(f"  ✓ 无干预")

    # 检查PSS状态
    if pss_result['pss_smooth'] >= config.PSS_THRESHOLD:
        print(f"\n✅ PSS 超过阈值 ({config.PSS_THRESHOLD})，干预会被触发")
    else:
        print(f"\n⚠️  PSS ({pss_result['pss_smooth']:.3f}) 未超过阈值 ({config.PSS_THRESHOLD})，在实际使用中干预不会触发")


def main():
    print("\n" + "="*70)
    print("测试: Tilt → Rotate 逻辑验证")
    print("="*70)

    # 测试场景 1: 头部向左倾斜 + 躯干倾斜（触发干预）
    test_scenario(
        name="头部左倾 + 躯干倾斜",
        head_offset_x=-0.12,
        trunk_angle=45,
        description="修复家向左倾斜 + 躯干倾斜 → 预期腕部右旋(+drz) + 升降(+dz)"
    )

    # 测试场景 2: 头部向右倾斜 + 躯干倾斜（触发干预）
    test_scenario(
        name="头部右倾 + 躯干倾斜",
        head_offset_x=+0.12,
        trunk_angle=45,
        description="修复家向右倾斜 + 躯干倾斜 → 预期腕部左旋(-drz) + 升降(+dz)"
    )

    # 测试场景 3: 头部正常（无干预）
    test_scenario(
        name="头部正常",
        head_offset_x=0.0,
        trunk_angle=20,
        description="修复家头部正常、轻微躯干倾斜 → 无倾斜干预"
    )

    print("\n" + "="*70)
    print("📋 逻辑验证总结")
    print("="*70)
    print("""
✓ 左倾 (-offset) → 正倾角 → 正drz → 右旋
✓ 右倾 (+offset) → 负倾角 → 负drz → 左旋
✓ 正常 (0 offset) → 零倾角 → 无旋转

关键特性:
  • 头部倾斜与腕部旋转方向相反（补偿性）
  • 升降逻辑独立于倾斜逻辑
  • 可见性检查确保鲁棒性（0.6+置信度）
  • 低头时仍会触发升降反应（躯干分数）
    """)


if __name__ == "__main__":
    main()
