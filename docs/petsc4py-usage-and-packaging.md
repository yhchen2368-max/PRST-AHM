# PETSc4py 调用与打包指南（Windows 原生构建）

> **目标读者**：任何需要在本项目（或本机）中调用 `petsc4py` 或将程序打包成 Windows exe 的开发者 / AI 代理。
> **最后验证日期**：2026-08-27（onedir 与 onefile 打包均已实测通过）
> **主文档位置**：`C:\Users\junji\petsc\PETSC4PY_GUIDE.md`（内容与此副本一致，如冲突以此副本为准）

---

## 1. 环境概览

本机已通过 **MSYS2 (UCRT64) + MinGW** 从源码编译安装了 PETSc 3.25.4 及 petsc4py 3.25.4，可直接使用，**无需再安装任何东西**。

| 组件 | 版本 | 位置 |
|------|------|------|
| conda Python（目标解释器） | 3.14.6 | `C:\Users\junji\Desktop\github\PRSTCore\.conda\python.exe` |
| PETSc 源码树 | 3.25.4 | `C:\Users\junji\petsc\petsc-3.25.4` |
| PETSc 编译产物（arch） | — | `C:\Users\junji\petsc\petsc-3.25.4\arch-mingw64-opt\lib\libpetsc.dll` |
| petsc4py（已装入 conda 环境） | 3.25.4 | `...\.conda\Lib\site-packages\petsc4py` |
| MSYS2 UCRT64 工具链 | gcc/gfortran 16.2.0 | `C:\msys64\ucrt64\bin` |
| MS-MPI 运行时 | 10.1.3 | `C:\Program Files\Microsoft MPI` + `C:\Windows\System32\msmpi.dll` |
| **hypre (BoomerAMG)** | 2.33.0 | `C:\msys64\ucrt64`（MSYS2 包 `mingw-w64-ucrt-x86_64-hypre`） |
| PyInstaller（打包用） | 6.22.2 | conda 环境内 |

**关键编译配置**（PETSc）：`PETSC_ARCH=arch-mingw64-opt`，共享库构建（`--with-shared-libraries=1`），MS-MPI，**hypre 已启用**（`--with-hypre-dir=/ucrt64`）。

**可用预条件器**：`hypre`（BoomerAMG，实测 40000 未知数 6 次迭代）、`gamg`（内置 AMG，12 次迭代）、`bjacobi`、`ilu`、`jacobi` 等。

---

## 2. 调用 petsc4py

### 2.1 前提：DLL 搜索路径自动注册（已被处理，勿删）

`petsc4py/__init__.py` 已被打过补丁：导入时自动调用 `os.add_dll_directory()` 注册 PETSc DLL 目录，否则会报 `ImportError: DLL load failed`。

```python
# ...\.conda\Lib\site-packages\petsc4py\__init__.py 中自动执行的内容（逻辑）：
import os
# 1) 包内相对路径：<pkg>/lib/arch-mingw64-opt （开发环境与 PyInstaller 打包内均有效）
# 2) 开发机绝对路径：C:\Users\junji\petsc\petsc-3.25.4\arch-mingw64-opt\lib
# 3) 开发机绝对路径：C:\msys64\ucrt64\bin
os.add_dll_directory(每个存在的目录)
```

> ⚠️ **重装 petsc4py 会覆盖此补丁**，必须重新打上（见 §5 备用补丁代码）。

### 2.2 最小调用示例

```python
import numpy as np
from petsc4py import PETSc

# 创建 3x3 稀疏矩阵
A = PETSc.Mat().createAIJ(size=(3, 3))
A.setValue(0, 0, 4.0); A.setValue(0, 1, 1.0)
A.setValue(1, 0, 3.0); A.setValue(1, 1, 1.0)
A.setValue(2, 2, 2.0)
A.assemble()

# 右端项与解向量
b = PETSc.Vec().createSeq(3)
b.setArray(np.array([1.0, 2.0, 3.0]))
x = A.createVecRight()

# KSP 线性求解器
ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType("gmres")          # 可选: cg / gmres / bcgs / preonly ...
ksp.getPC().setType("ilu")    # 可选: jacobi / ilu / bjacobi / lu ...
ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
ksp.solve(b, x)
print("收敛:", ksp.getConvergedReason(), "迭代数:", ksp.getIterationNumber())
print("解:", x.getArray())
```

### 2.3 常用 API 速查

| 用途 | 代码 |
|------|------|
| 版本信息 | `PETSc.Sys.getVersion()` |
| MPI 信息 | `PETSc.COMM_WORLD.getSize()`, `.getRank()` |
| 稀疏矩阵 | `PETSc.Mat().createAIJ(size=(n,n), nnz=k)` |
| 稠密矩阵 | `PETSc.Mat().createDense((n,n))` |
| 向量 | `PETSc.Vec().createSeq(n)`，`vec.setArray(np_array)`，`vec.getArray()` |
| 线性求解 | `PETSc.KSP().create()`（setType / getPC / setTolerances / solve） |
| 启用 hypre | `ksp.getPC().setType("hypre")`（BoomerAMG；可与 `cg`/`gmres` 搭配） |
| 非线性求解 | `PETSc.SNES().create()`（setFunction / setJacobian / solve） |
| 命令行选项 | `ksp.setFromOptions()`，运行时 `-ksp_type gmres -pc_type ilu` |
| 清理 | 对象可 `destroy()`，或进程结束时自动释放 |

---

## 3. 打包成 exe（PyInstaller）

### 3.1 为什么必须特殊配置（四个坑）

1. **PyInstaller 没有 petsc4py 的官方 hook**（核心 hooks 与 hooks-contrib 均无）。
2. **`.pyd` 是动态加载的**：petsc4py 从 `petsc.cfg` 读取 `PETSC_ARCH` 后，用 `importlib` 按运行时路径加载 `PETSc.cp314-win_amd64.pyd`。PyInstaller 静态分析**追踪不到**它 → 默认打包后 `.pyd` 缺失。
3. **`.pyd` 不能加进 `binaries`**：PyInstaller 会把 binaries 中的 `.pyd` 当扩展模块处理，**onefile 模式下会被丢弃**。必须作为 **data 文件（datas）** 收集，且目标路径必须带 `petsc4py/` 包前缀。
4. **`msmpi.dll` 是系统 DLL**：不会被自动收集。若目标机器未装 MS-MPI，需要把它从 `C:\Windows\System32\msmpi.dll` 手动打进包里，实现完全自包含。

### 3.2 可复用的 hook 文件（已就绪）

文件位于：**`C:\Users\junji\petsc\hooks\hook-petsc4py.py`**（内容见 §5）。该 hook 自动完成：
- 收集 petsc4py 全部子模块（`collect_submodules`）
- 收集包内数据（含 `petsc.cfg`）与全部 DLL（`collect_dynamic_libs` → `libpetsc.dll`、`libopenblas.dll`、gcc 运行时等）
- 显式收集 `petsc4py/lib/<arch>/*.pyd`（作为 datas）
- 打包 `msmpi.dll` 到包根目录

### 3.3 打包命令

**构建机要求**（打包前必须设置，否则依赖分析找不到 PETSc DLL）：

```powershell
$env:PATH = "C:\Users\junji\petsc\petsc-3.25.4\arch-mingw64-opt\lib;C:\msys64\ucrt64\bin;" + $env:PATH
```

**onefile（推荐，单文件分发）**：

```powershell
cd <你的项目目录>
& "C:\Users\junji\Desktop\github\PRSTCore\.conda\python.exe" -m PyInstaller `
  --noconfirm --clean --onefile `
  --additional-hooks-dir C:\Users\junji\petsc\hooks `
  --name myapp `
  app.py
```

**onedir（目录模式，启动更快）**：把 `--onefile` 换成 `--onedir` 即可。

### 3.4 验证清单

打包后应检查：

```
1. dist/myapp.exe 能直接运行（不设任何额外 PATH）
2. 输出 KSP 求解结果正确
3. 归档包含（可用 pyi-archive_viewer 检查）：
   - petsc4py/lib/arch-mingw64-opt/PETSc.cp314-win_amd64.pyd
   - petsc4py/lib/arch-mingw64-opt/libpetsc.dll （及 libopenblas/libgcc/libgfortran 等）
   - msmpi.dll
4. 拷贝到一台【未安装】PETSc/MS-MPI/MSYS2 的干净 Windows x64 机器上仍能运行
```

> 已实测：onedir 与 onefile 均通过，目标机无需任何额外安装。

---

## 4. 注意事项（坑）

| # | 注意点 |
|---|--------|
| 1 | **重装 petsc4py 会覆盖 `__init__.py` DLL 补丁**，需按 §5 重新打补丁。 |
| 2 | **构建/打包时的 PATH** 必须含 `arch-mingw64-opt\lib` 与 `C:\msys64\ucrt64\bin`，否则 PyInstaller 收集不到 `libpetsc.dll` 依赖。 |
| 3 | **不要设置 `CC`/`CXX` 环境变量为带反斜杠的路径**（如 `C:\msys64\ucrt64\bin\gcc.exe`）。setuptools 用 `shlex.split()` 解析，反斜杠会被当转义符 → `WinError 2`。 |
| 4 | **Cython 必须 ≤ 3.0.x**（当前已装 3.0.12）。Cython 3.3 对 petsc4py 3.25.4 报 `Invalid index type 'int'`。 |
| 5 | **只能 x64 Windows**：PETSc 按 `win-amd64` 编译，`cp314-win_amd64` 扩展不可用于 32 位或其它架构。 |
| 6 | conda 环境的 `Lib\Library\bin\libgcc_s_seh-1.dll` 与 MSYS2 版本**不兼容**，不要通过 PATH 方式解决 DLL（会加载错版本）；必须依赖 `__init__.py` 补丁的 `add_dll_directory`。 |
| 7 | `petsc4py.lib._pytypes` 需要可选依赖 `pyvista`，打包时会有无害警告，可忽略。 |

---

## 5. 备用补丁 / hook 源码（重装后恢复用）

### 5.1 `petsc4py/__init__.py` DLL 注册补丁

在 `...\.conda\Lib\site-packages\petsc4py\__init__.py` 的 `__credits__` 之后插入：

```python
# --- Windows DLL search-path registration (MSYS2/MinGW-built PETSc) ---
import os as _os
if _os.name == 'nt':
    _extra_dirs = []
    _pkgdir = _os.path.dirname(_os.path.abspath(__file__))
    _arch_lib = _os.path.join(_pkgdir, 'lib', 'arch-mingw64-opt')
    if _os.path.isdir(_arch_lib):
        _extra_dirs.append(_arch_lib)
    for _d in (
        r'C:\Users\junji\petsc\petsc-3.25.4\arch-mingw64-opt\lib',
        r'C:\msys64\ucrt64\bin',
    ):
        if _os.path.isdir(_d):
            _extra_dirs.append(_d)
    for _d in _extra_dirs:
        try:
            _os.add_dll_directory(_d)
        except Exception:
            pass
    del _extra_dirs, _pkgdir, _arch_lib, _d
del _os
```

### 5.2 `confPETSc.py` 补丁（仅重编译 petsc4py 时需要）

文件：`C:\Users\junji\petsc\petsc-3.25.4\src\binding\petsc4py\conf\confPETSc.py`
在 `configure_extension()` 的"Link in extra libraries on static builds"分支后追加 Windows 分支（petsc4py 直接调用 MPI 符号，共享库构建也必须链接 MS-MPI）：

```python
        # Link in extra libraries on static builds
        if self['BUILDSHAREDLIB'] != 'yes':
            petsc_ext_lib = split_quoted(self['PETSC_EXTERNAL_LIB_BASIC'])
            petsc_lib['extra_link_args'].extend(petsc_ext_lib)
        elif sys.platform == 'win32':
            mpi_lib = split_quoted(self.get('MPI_LIB', ''))
            mpi_lib = [f for f in mpi_lib if not f.startswith('-Wl,')]
            petsc_lib['extra_link_args'].extend(mpi_lib)
```

### 5.3 PyInstaller hook 源码（`C:\Users\junji\petsc\hooks\hook-petsc4py.py`）

```python
# hook-petsc4py.py
import glob
import os
from PyInstaller.utils.hooks import (
    collect_data_files, collect_dynamic_libs, collect_submodules,
    get_package_paths,
)

hiddenimports = collect_submodules("petsc4py")
datas = collect_data_files("petsc4py")
binaries = collect_dynamic_libs("petsc4py")

# .pyd 动态加载，静态分析发现不了 → 作为 datas 收集（onefile 下 binaries 会被丢弃）
pkg_dir = get_package_paths("petsc4py")[1]
for pyd in glob.glob(os.path.join(pkg_dir, "lib", "*", "*.pyd")):
    rel = os.path.relpath(pyd, pkg_dir)
    dest = "petsc4py/" + os.path.dirname(rel).replace(os.sep, "/")
    datas.append((pyd, dest))

# 打包 MS-MPI 运行时，目标机无需安装
_msmpi = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "msmpi.dll")
if os.path.isfile(_msmpi):
    binaries.append((_msmpi, "."))
```

---

## 6. 相关文件 / 命令速查

| 用途 | 路径 / 命令 |
|------|------------|
| PETSc configure 脚本 | `C:\Users\junji\petsc\build_petsc.sh` |
| PETSc 编译脚本 | `C:\Users\junji\petsc\make_petsc.sh` |
| petsc4py 构建脚本 | `C:\Users\junji\petsc\build_petsc4py.sh` |
| petscvariables 路径转换 | `C:\Users\junji\petsc\fix_petscvars.py`（POSIX→Windows 路径） |
| 打包测试工程 | `C:\Users\junji\petsc\exetest\`（app.py + hooks\） |
| 打包 hook | `C:\Users\junji\petsc\hooks\hook-petsc4py.py` |
| 打包命令 | `python -m PyInstaller --onefile --additional-hooks-dir C:\Users\junji\petsc\hooks app.py`（需先设置 §3.3 的 PATH） |
| 归档检查 | `python -m PyInstaller.utils.cliutils.archive_viewer -l dist\myapp.exe` |
