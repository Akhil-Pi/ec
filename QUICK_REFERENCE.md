# Quick Reference Guide - Empathetic Conservator v2

**Last Updated**: 2026-05-08  
**Version**: 2.0 (Dynamic Cooldown Support)

---

## 🚀 Quick Start

```bash
# Control Group (Static table, no intervention)
python main.py --participant P01 --condition control --simulate

# Experimental Group (UR3 dynamic assistance)
python main.py --participant P02 --condition experimental --simulate

# Real execution (requires UR3 connection)
python main.py --participant P03 --condition experimental
```

---

## 📋 Core Workflow

### Control Group
```
Start → Calibration(15s) → Task(45min) → End
      ↓                   ↓
  Record baseline     📊 Record PSS only
                      ❌ No UR3 intervention
```

### Experimental Group  
```
Start → Calibration(15s) → Transition(~1min) → Task(45min) → End
      ↓                   ↓                    ↓
  Record baseline     UR3 moves to        PSS > 0.4
                      task position       and sustained 2s
                                          ↓
                                        🤖 Intervene(raise/tilt)
                                          ↓
                                        ⏱️ Dynamic cooldown
                                          ↓
                                        📊 Record latency
```

---

## 🎯 Intervention Logic

```
PSS exceeds 0.4 threshold?
  ├─ No → Continue monitoring
  ├─ Yes, but < 2s sustained → Continue monitoring
  └─ Yes, and ≥ 2s sustained → Trigger intervention
       ↓
    Execute actions:
    • Trunk > 0.4  → Raise (Z+)
    • Cervical > 0.4 → Tilt (Rx+)
       ↓
    Wait for motion complete → Record latency
       ↓
    Dynamic cooldown (0.5~3.0s)
```

---

## ⏱️ Dynamic Cooldown Schedule

| Condition | Cooldown | Logged As | Scenario |
|-----------|----------|-----------|----------|
| PSS improvement > 0.1 | **1.0s** | fast_recovery | Participant self-correcting |
| PSS > 0.60 | **1.5s** | high_pss | Situation deteriorating, need quick response |
| 0.40-0.60 | **3.0s** | standard | Normal exceeding threshold |
| PSS < 0.40 | **0.5s** | recovering | Recovering state |

---

## 📊 Output Files

### frames.csv (Per-frame data)
```
timestamp_s, trunk_angle_deg, trunk_score, cervical_cm, 
cervical_score, pss_raw, pss_smooth

One row every 10 frames (10Hz sampling rate)
```

### events.csv (Intervention records)
```
timestamp_s, event_type, pss_at_event, actions, latency_s, details

Intervention events: intervention
├─ latency_s: 0.5-1.5 seconds (ensured by wait_for_motion_complete())
├─ actions: format "raise:0.008|tilt:0.025"
└─ details: "id=3" (intervention sequence number)
```

### meta.txt (Session metadata)
```
participant_id: P01
condition: experimental
start_time: 2026-05-08T14:30:00
duration_s: 2700
frames_logged: 450
events_logged: 8
pss_threshold: 0.40
pss_hysteresis: 0.10
```

---

## 🔧 Configuration Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| PSS_THRESHOLD | 0.40 | Intervention trigger point |
| PSS_HYSTERESIS | 0.10 | Recovery threshold (0.30) |
| sustained_seconds | 2.0 | Monitoring duration before trigger |
| Z_ADJUST_STEP | 0.02m | Single raise magnitude |
| TILT_ADJUST_STEP | 0.05rad | Single tilt magnitude |
| cooldown_seconds | (dynamic) | 1.0-3.0 seconds |

---

## 🧪 Verification Checklist

Before running:
- [ ] UR3 connected and at HOME position
- [ ] Camera working (seeing real-time video)
- [ ] MediaPipe detecting human body
- [ ] `data/sessions/` directory writable

During execution:
- [ ] See PSS real-time display bar (0.0-1.0)
- [ ] Trunk and Cervical angles displaying reasonably
- [ ] Timer countdown working normally

After execution:
- [ ] Generated `*_frames.csv` (should have ~450 rows)
- [ ] Generated `*_events.csv`
- [ ] Generated `*_meta.txt`

Data validation:
- **Control Group**: events.csv has no intervention rows ✓
- **Experimental Group**: events.csv has intervention rows + latency_s column has values ✓

---

## 🐛 Frequently Asked Questions

**Q: Why is events.csv empty for control group?**  
A: That's correct! Control group doesn't intervene, so no intervention events. PSS is only recorded in frames.csv.

**Q: PSS never exceeds 0.4?**  
A: ✓ That's also good! Means participant has excellent posture. Data is valid.

**Q: latency_s is empty?**  
A: ❌ Means wait_for_motion_complete() timed out. Check UR3 connection or increase timeout.

**Q: Always showing "transitioning"?**  
A: Normal. Experimental group needs ~1 minute to complete gradual_transition to task position.

**Q: Interventions too frequent/too sparse?**  
A: Check PSS_THRESHOLD configuration. Default 0.40 is reasonable.

---

## 📈 Data Analysis Tips

### Control Group
```python
# Calculate mean PSS
df_control = pd.read_csv('*_control_*_frames.csv')
mean_pss = df_control['pss_smooth'].mean()
```

### Experimental Group
```python
# Intervention effectiveness
df_frames = pd.read_csv('*_experimental_*_frames.csv')
df_events = pd.read_csv('*_experimental_*_events.csv')

# Number of interventions
n_interventions = df_events[df_events['event_type']=='intervention'].shape[0]

# Average latency (for Spearman analysis)
latencies = df_events['latency_s'].dropna()
mean_latency = latencies.mean()

# PSS changes before and after intervention
# (analyze effectiveness, see pss_at_event in *_events.csv)
```

---

## 🎓 Key Research Metrics

| Metric | Location | Purpose |
|--------|----------|---------|
| PSS score sequence | frames.csv | Primary observation variable |
| Intervention count | events.csv row count | Problem severity |
| **Latency** | latency_s column | Spearman correlation analysis |
| Trunk angle | trunk_angle_deg | Trunk inclination degree |
| Cervical displacement | cervical_cm | Forward head displacement |
| Dynamic cooldown reason | cooldown() log | Participant response pattern |

---

## 📞 Debugging Commands

```bash
# View latest session
ls -lrt data/sessions/ | tail -5

# Check control group has no interventions
grep "intervention" data/sessions/*_control_*_events.csv

# Check experimental group has latency
grep "intervention" data/sessions/*_experimental_*_events.csv | cut -d, -f5 | head -10

# View PSS statistics
python3 -c "
import pandas as pd
df = pd.read_csv('data/sessions/P01_experimental_*_frames.csv')
print(df['pss_smooth'].describe())
"
```

---

## 🎯 Guarantees After Fixes

✅ **Latency Accuracy**: wait_for_motion_complete() ensures actual motion completion  
✅ **Control Group Isolation**: condition parameter prevents unintended interventions  
✅ **Fast Response**: Dynamic cooldown re-evaluates high PSS within 1.5s  
✅ **Defensive Programming**: Missing PSS fields handled gracefully  
✅ **Research Validity**: Clear intervention decision logs for subsequent analysis  

---

**Ready to run the experiment! 🚀**
