# 🤖 Cobot 腕部旋转控制 - 快速参考

## 核心逻辑 (30 秒理解)

```
修复家的动作              机械臂的反应
─────────────────────────────────────
向左倾斜 (-angle)    →   腕部向右旋转 (+drz)
向右倾斜 (+angle)    →   腕部向左旋转 (-drz)
低头                 →   升降 (+dz)
向前靠               →   前移 (+dy)
```

## 物理直觉

想象修复家和机器人对着面坐：
- 修复家头向左歪，想看左边 → 机器人向右转工件
- 修复家头向右歪，想看右边 → 机器人向左转工件
- **结果**: 工件始终在修复家的最佳视角

## 关键参数速查

| 参数 | 值 | 含义 |
|------|-----|------|
| `MIN_VISIBILITY` | 0.6 | 可见性阈值（低=容错强） |
| `tilt_score > 0.3` | 触发值 | 倾斜检测灵敏度 |
| `ROTATE_ADJUST_STEP` | 0.10 rad | 单次旋转幅度（~6°） |
| `RAISE triggered` | `trunk > 0.5` | 升降触发条件 |

## 代码位置快速定位

| 功能 | 文件 | 函数 |
|------|------|------|
| 头部倾斜计算 | `pss_calculator.py` | `head_tilt_angle()` |
| 干预决策 | `intervention_policy.py` | `_compute_interventions()` |
| 腕部旋转执行 | `ur3_controller.py` | `adjust_rotate()` |
| 参数配置 | `config.py` | `ROTATE_ADJUST_STEP` |

## 调试流程

### 症状: 旋转方向反了
```
修复家左倾 → 机械臂向左转（错误）
解决: 检查 intervention_policy.py 中的符号
    magnitude = -step * sign(tilt_angle)
           ↑
         这个负号控制方向反转
```

### 症状: 旋转不够灵敏
```
倾斜角度大但不旋转 → 可能 MIN_VISIBILITY 太高
解决: 降低 MIN_VISIBILITY 从 0.6 → 0.5
     或增加 tilt_score 阈值从 0.3 → 0.2
```

### 症状: 误触发（无倾斜也旋转）
```
修复家正常头位还在旋转 → 阈值太低
解决: 提高 tilt_score 阈值从 0.3 → 0.4
     或提高 MIN_VISIBILITY 从 0.6 → 0.7
```

## 测试命令

```bash
# 运行逻辑验证测试
python3 test_tilt_to_rotate.py

# 预期输出: 
# ✓ 左倾 → rotate RIGHT
# ✓ 右倾 → rotate LEFT
# ✓ 正常 → no rotation
```

## 关键公式

### 头部倾斜角计算
```
head_offset = head_x - shoulder_x
tilt_angle = arctan(head_offset / shoulder_width) × (180/π)
tilt_score = |tilt_angle| / 20°  (范围: 0-1)
```

### 腕部旋转量
```
drz = -ROTATE_ADJUST_STEP × tilt_score × sign(tilt_angle)

负号: 反向旋转（补偿）
符号: 角度方向（负=左倾，正=右倾）
```

### PSS 权重（新）
```
PSS = 0.40 × trunk_score
    + 0.30 × tilt_score
    + 0.30 × lean_score
```

## 常见问题 (FAQ)

**Q: 为什么要用反向旋转？**
A: 修复家倾斜是为了看工件，机械臂跟着反向旋转保持工件在最佳角度。

**Q: 升降逻辑还要吗？**
A: 要的。升降是独立的，根据躯干倾斜（低头时抬升）。

**Q: 如何处理机械臂遮挡胳膊？**
A: 可见性阈值只需 0.6（非常低），容错能力强。

**Q: 可以同时升降和旋转吗？**
A: 可以。干预策略会生成多个动作，机器人逐个执行。

**Q: 旋转幅度是固定的吗？**
A: 否。幅度 = 基础步长 × 倾斜分数，倾斜越厉害，旋转越大。

## 关键文件对比

### 旧系统
```
躯干倾斜 → raise
颈椎位移 → tilt （工件倾斜）
向前靠 → forward
```

### 新系统 ✨
```
躯干倾斜 → raise （同上）
头部左右倾斜 → rotate （腕部旋转）✨NEW
向前靠 → forward （同上）
```

## 验收标准

- [ ] 左倾斜 → 右旋
- [ ] 右倾斜 → 左旋
- [ ] 升降仍正常
- [ ] 机械臂不误触
- [ ] 可见性检查有效

## 紧急停止

如有异常旋转：
```bash
# 1. 按机械臂E-Stop
# 2. 禁用 rotate 动作:
#    在 _compute_interventions() 中注释 rotate 部分
# 3. 排查 tilt_angle 计算
```

---

**最后更新**: 2026-05-11  
**版本**: 1.0  
**测试状态**: ✅ 已验证
