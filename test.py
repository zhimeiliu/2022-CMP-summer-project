
from lyzeum.patch_extraction import PatchExtractor
f=PatchExtractor(patch_size=1024, stride=1024, min_patch_mag=5.0, max_patch_mag=5.0, cleanup_workers=1,
 qupath_binary="/Applications/QuPath.app/Contents/MacOS/QuPath", zip_patches=False)
f(wsi="TestSample.svs", parent_dir="test-patches")
