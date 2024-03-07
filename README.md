# ellipse_tangent

[![Linux](https://github.com/burbokop/ellipse_tangent/actions/workflows/linux.yml/badge.svg)](https://github.com/burbokop/ellipse_tangent/actions/workflows/linux.yml)
[![Android](https://github.com/burbokop/ellipse_tangent/actions/workflows/android.yml/badge.svg)](https://github.com/burbokop/ellipse_tangent/actions/workflows/android.yml)

## run test & benchmark on android
```bash
PATH=$PATH:~/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/bin cargo test --target armv7-linux-androideabi
```
```bash
PATH=$PATH:~/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/bin cargo bench --target armv7-linux-androideabi
```
