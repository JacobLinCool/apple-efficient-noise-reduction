# Third-party notices

## DeepFilterNet3 weights

`Sources/EfficientNoiseReduction/Resources/DeepFilterNet3.safetensors` is the
DeepFilterNet3 checkpoint from https://github.com/Rikorose/DeepFilterNet (as distributed by
the `deepfilternet` 0.5.6 Python package), converted to safetensors by
`reference/golden_dfn3.py`. It is licensed under either the Apache License 2.0 or the MIT
license, at your option; the MIT license text follows.

```
The MIT License (MIT)
Copyright (c) 2021 Hendrik Schröter

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Real-time RE-USE weights (not included)

The Real-time RE-USE checkpoint (https://huggingface.co/nvidia/Real-time_RE-USE) is
released by NVIDIA under the NVIDIA One-Way Noncommercial License (NSCLv1). This
repository does not contain it; `noise-reduction download` fetches it on request, and its
use is subject to that license (research and development only, noncommercial).

## Dependencies

- [mlx-swift](https://github.com/ml-explore/mlx-swift), MIT
- [swift-numerics](https://github.com/apple/swift-numerics), Apache-2.0
- [swift-argument-parser](https://github.com/apple/swift-argument-parser), Apache-2.0
