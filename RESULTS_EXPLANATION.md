# Understanding the Results

## Why Results Vary

The experiments use **random projections** (no training), so results naturally vary between runs. This is expected and normal.

## Key Findings

From multiple independent trials:
- **Average improvement: +2-6%** on MNIST
- **Best cases: up to +19.8%** 
- Some trials show negative results (normal for random initialization)

## Statistical Significance

With random projections:
- Standard deviation is typically 2-3%
- Improvements of 2-6% are meaningful
- Results become more stable with:
  - More trials (we use 10)
  - Larger datasets
  - Proper training (future work)

## The Important Insight

Ring Convolution Networks show **selective advantage on structured data**:
- ✅ Images (MNIST): Positive improvement
- ❌ Random noise: Negative improvement  
- ❌ Smooth signals: Negative improvement

This suggests RCN acts as an implicit feature detector for spatial patterns.

## Speed Comparison

- `experiments.py`: Full OOP implementation (~600s for full test)
- `fast_experiments.py`: Vectorized implementation (~0.01s per trial)

Both give similar results, but the fast version is recommended for experimentation.

## Next Steps

1. Implement proper training (backpropagation)
2. Test on larger datasets (CIFAR-10, ImageNet)
3. Explore different ring configurations
4. Hardware optimization

## Reproducibility

Results depend on:
- Random seed
- Data split
- Sample size
- NumPy version

To get exactly the same results, fix the random seed:
```python
np.random.seed(42)
```

## Citation

If you use these results, please cite:
```
Akbarali. (2025). Ring Convolution Networks: A Novel Neural Architecture 
Inspired by Quantum Superposition. Zenodo. https://doi.org/10.5281/zenodo.15776775
```
