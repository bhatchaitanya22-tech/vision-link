## Dark background experiment — batch_02

### Result
Partially better but still inconsistent.

### Key finding
Pixel distribution varies dramatically based on lighting:
- Image 2 (light on cloth): pixels > 50 = 96.2%  FAIL
- Image 6 (darker area):    pixels > 50 = 29.7%  BETTER
- Image 7 (darkest):        pixels > 50 = 18.7%  BEST

### Conclusion
Even with controlled background, inconsistent factory
lighting makes thresholding unreliable. A fixed threshold
that works in one lighting condition fails in another.
YOLOv8 learns features robust to lighting variation.
Classical CV cannot. Case closed.