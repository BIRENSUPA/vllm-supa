# vLLM-SUPA

vLLM-SUPA 是面向壁仞 BR200/SUPA 硬件的 vLLM 插件，提供平台注册、运行时补丁以及 SUPA 扩展。

## 系统要求

- Ubuntu 22.04、Ubuntu 24.04 或兼容 Linux
- Python 3.10+
- BR2XX 硬件和 BIRENSUPA SDK
- 使用 Docker 时需要 Docker 20.10.7+

版本统一定义在 [`upstream_version.txt`](upstream_version.txt)：

```bash
cat upstream_version.txt
```

## 安装方式

### Docker 镜像

使用预装 vLLM-SUPA、TorchSUPA 和运行时依赖的镜像（镜像发布情况以交付说明为准）。

### Wheel 安装

在已安装 BIRENSUPA SDK 的环境中：

```bash
source /usr/local/birensupa/all/latest/scripts/brsw_set_env.sh
suda init
. "$HOME/.gstub/suda.sh"
suda load

python3 -m pip install "vllm==<VLLM_VERSION>" --no-deps

python3 -m pip install flashattn_infer-*.whl suattention-*.whl deep_ep-*.whl \
  flash_mla-*.whl triton-*.whl tilelang-*.whl torch_supa-*.whl vllm_supa-*.whl
```

### 源码构建

适用于需要修改代码或重新编译扩展的开发环境：

```bash
MAX_JOBS=8 python3 -m pip install -e . --no-build-isolation  # 开发环境
```

构建发布 wheel：

```bash
python3 setup.py bdist_wheel
```

## 文档编译

- html

```bash
cd docs/vllm_supa
make html       # HTML: build/html/
```

- pdf

```bash
# PDF 编译依赖(Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y latexmk texlive-xetex texlive-lang-chinese

make latexpdf   # PDF: build/latex/
```

更多内容请参阅 [vLLM 文档](https://vllm.readthedocs.io/) 和 `docs/` 目录。

## 许可证

许可证信息请参阅仓库中的 [`LICENSE`](LICENSE) 文件。
