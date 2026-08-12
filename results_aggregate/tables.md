### Multi-task success rate (10 seeds, mean [95% CI])

| Task | Augmented (learned) | Baseline | Delta (pp) |
|---|---|---|---|
| Morris Water Maze | 0.664 [0.487, 0.837] | 0.533 [0.329, 0.736] | 13.1 [-15.6, 43.3] |
| Visual Foraging | 0.803 [0.684, 0.914] | 0.690 [0.579, 0.809] | 11.3 [-5.1, 27.8] |
| Dynamic Obstacles | 0.870 [0.846, 0.893] | 0.663 [0.592, 0.738] | 20.7 [13.2, 27.1] |
| Visual Search | 0.314 [0.174, 0.482] | 0.087 [0.057, 0.120] | 22.7 [9.3, 38.0] |
| **Overall** | **0.663 [0.564, 0.764]** | **0.493 [0.406, 0.586]** | **16.9 [3.8, 30.3]** |
| **Overall (IQM)** | **0.732** | **0.505** | **+22.7** |

### Selection-mechanism ablation

| Selection mechanism | Seeds | Multi-task success | vs. learned (pp) | Trained params |
|---|---|---|---|---|
| `learned` | 10 | 0.663 [0.564, 0.763] | -- | 231,050 |
| `random` | 10 | 0.591 [0.567, 0.614] | -7.2 [-17.8, 3.6] | 152,518 |
| `fixed-explore` | 5 | 0.657 [0.603, 0.703] | -4.7 [-16.7, 7.3] | 152,518 |
| `fixed-approach` | 5 | 0.769 [0.714, 0.825] | 6.6 [-9.9, 23.1] | 152,518 |
| `fixed-exploit` | 5 | 0.461 [0.340, 0.596] | -24.3 [-31.9, -15.7] | 152,518 |
| `uniform-sum` | 5 | 0.863 [0.814, 0.909] | 15.9 [3.6, 29.8] | 152,518 |

### Zero-shot transfer to unseen task variants

| Arm | Mean success [95% CI] |
|---|---|
| Augmented (learned) | 0.658 [0.561, 0.755] |
| Baseline | 0.495 [0.405, 0.587] |
| Delta (pp) | 16.3 [2.7, 29.2] |
