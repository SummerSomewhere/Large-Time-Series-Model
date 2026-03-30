aggregate/: mean/std HSIC over batches + summary plots.
samples/sample_XXXXXX/: mi_per_patch_by_layer.png = each row is one layer, each point is one patch's MI proxy (HSIC or cosine if n_vars<4).
hsic_per_patch.npy, cos_align_per_patch.npy, summary.json.
Default --max_samples=10; use --max_samples 0 for every test sample.
