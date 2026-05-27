aggregate/ (population-level, over all batches that passed filters):
  hsic_mean.npy, hsic_std.npy, summary.json — batch-mean/std HSIC per (layer, input patch t).
  layer_LL_hsic.png — for layer L: mean HSIC vs t, shaded band = batch std; red = IQR peak line.
  all_layers_hsic_overlay.png — every layer's batch-mean HSIC vs t on one axes.
  layer_LL_mi_all_samples.png — thin curves = each saved sample's MI vs t; black dashed = batch mean.
  layer_LL_mi_sample_x_patch.png — heatmap rows=samples, cols=t, for that layer.
samples/sample_XXXXXX/: layer_LL_mi.png, layer_patch_mi_heatmap.png, mi_all_layers_patches_summary.png, npy.
Prefer joint encode on cat(history,true future); else split encode if patch crosses boundary.
z_y = concat of future patch token vectors (no mean); HSIC(h_x[t], z_y) per layer.
Default --max_samples=100; use --max_samples 0 for every test sample.
