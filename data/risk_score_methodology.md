# Climate mismatch risk

For every crop, the training split supplies a rainfall mean and standard deviation. For a row with rainfall `r`, crop-specific mismatch is:

`risk = clip(abs(r - crop_mean) / (2 * crop_std) * 100, 0, 100)`

A score of 0 means the rainfall is near that crop's observed center; 100 means it is at least two training standard deviations away. The two-sigma scale is a transparent heuristic, not an agronomic threshold. The regression model learns this engineered target from the seven numeric inputs and is evaluated on a held-out test split. Crop-specific rainfall statistics are fit on training data only to avoid leakage.
