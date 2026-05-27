aggregate/: batch-mean HSIC (layer_*_hsic.png, all_layers_hsic_overlay.png).
  layer_LL_mi_all_samples.png = one figure per layer: all saved samples' MI vs patch + dashed batch mean.
  layer_LL_mi_sample_x_patch.png = heatmap sample×patch for that layer.
samples/sample_XXXXXX/: per-sample layer_LL_mi.png, layer_patch_mi_heatmap.png, npy.
Prefer joint encode on cat(history,true future); else split encode if patch crosses boundary.
z_y = concat of future patch token vectors (no mean); HSIC(h_x[t], z_y) per layer.
Default --max_samples=10; use --max_samples 0 for every test sample.
