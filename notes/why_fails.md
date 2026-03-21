## Classical CV failure analysis — 21 March 2026

### Result
Classical thresholding fails on this dataset.

### Why
Background (light table) and cylinder (aluminium) occupy
the same pixel value range: 50–150.

No threshold cleanly separates object from background.

pixels > 50:  ~89% — almost entire image

pixels > 100: ~61% — still most of the image  

pixels > 150: ~27% — getting better but still noisy

pixels > 200:  ~2% — only specular highlights, not cylinder

### Conclusion
Need feature-based detection, not intensity-based.

YOLOv8 learns shape + texture features, not just brightness.

This is exactly why deep learning is required for Vision-Link.