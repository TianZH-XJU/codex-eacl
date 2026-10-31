# Error analysis (all gates)

## ProcessBench Gate 1 (math Choice)
- Tournament correct set identical to single; wide net-harmful; oracle +0.017.
- SINGLE predictions collapse to early steps (113/120 on steps 0-5, zero sentinel).
## Code Gate 1
- Upgrades only reshuffle errors (each fixes 1, breaks 15); LR OOF below majority.
## FinDVer Gate 1
- Switch +0.05 raw over single but unpredictable (LR OOF below majority).
## Gate 4 (config sensitivity, math)
- Only significant effect: wide_max1024 HURTS (delta CI below zero).
- order_reverse flips 18 but net delta CI covers zero. No repeatable quality gain.
## Laya vs Vela baselines
- Both models near/below chance on math Choice; both ~chance on code; Laya ahead on finance but weak. No credible model gap claim.
