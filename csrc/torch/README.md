# Temporary torch-supa stable Header Patch

This directory temporarily shadows headers from the installed PyTorch package.
The build adds `csrc` to the include path, so files under this directory must
preserve their original `torch/...` include paths.
