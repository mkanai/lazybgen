"""
Setup configuration for lazybgen package with Cython extensions.
"""

import os
import platform
import subprocess
import sys
from distutils.ccompiler import new_compiler
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Determine platform-specific compile args.
# Do NOT add -ffast-math here: it implies -ffinite-math-only, which lets the
# compiler assume NaN never occurs and drop the NaN stores / isnan logic that
# the missing-genotype contract depends on (missing samples must decode to NaN).
EXTRA_COMPILE_ARGS = ["-O3", "-funroll-loops"]
EXTRA_LINK_ARGS = []

# Add SIMD optimizations for x86_64 architectures
if platform.machine() in ["x86_64", "AMD64"]:
    if sys.platform != "win32":
        EXTRA_COMPILE_ARGS += ["-mavx", "-mavx2", "-mfma"]

if sys.platform == "darwin":
    EXTRA_COMPILE_ARGS += ["-stdlib=libc++", "-std=c++14"]
elif sys.platform == "linux":
    EXTRA_COMPILE_ARGS += ["-std=c++14"]
elif sys.platform == "win32":
    EXTRA_COMPILE_ARGS = ["/O2", "/std:c++14", "/arch:AVX2"]

# Base directory for the BGEN extension sources (the package directory)
BGEN_DIR = Path("lazybgen")
BUILD_DIR = Path("build")


def build_zlib_ng():
    """Build zlib-ng from submodule."""
    zlib_dir = BGEN_DIR / "zlib-ng"
    zlib_build_dir = BUILD_DIR / "zlib-ng"

    if not zlib_dir.exists() or not any(zlib_dir.iterdir()):
        raise RuntimeError(
            f"zlib-ng submodule not found at {zlib_dir}\n\n"
            "The vendored compression libraries are missing. Please run:\n"
            "  git submodule update --init --recursive\n\n"
            "Or clone with submodules:\n"
            "  git clone --recursive https://github.com/mkanai/lazybgen.git\n"
        )

    zlib_build_dir.mkdir(parents=True, exist_ok=True)

    cmake_args = [
        "cmake",
        "-S",
        str(zlib_dir.absolute()),
        "-B",
        str(zlib_build_dir.absolute()),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DZLIB_COMPAT=ON",  # Enable zlib compatibility mode
        "-DBUILD_SHARED_LIBS=OFF",  # Build static library
        "-DZLIB_ENABLE_TESTS=OFF",  # Disable tests to avoid GTest dependency
        "-DWITH_GTEST=OFF",  # Disable GTest
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",  # Enable -fPIC for static libs
    ]

    if sys.platform == "darwin":
        # Honor MACOSX_DEPLOYMENT_TARGET (set by cibuildwheel); arm64 requires >= 11.0
        deployment_target = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "10.9")
        cmake_args.extend([f"-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment_target}"])

    try:
        subprocess.check_call(cmake_args)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"CMake configuration failed with error: {e}\n\n"
            "Please ensure CMake is installed:\n"
            "  Ubuntu/Debian: sudo apt-get install cmake\n"
            "  macOS: brew install cmake\n"
            "  pip: pip install cmake\n"
        )

    subprocess.check_call(["cmake", "--build", str(zlib_build_dir), "--config", "Release"])

    if sys.platform == "win32":
        lib_path = zlib_build_dir / "Release" / "zlibstatic.lib"
    else:
        lib_path = zlib_build_dir / "libz.a"

    if not lib_path.exists():
        raise RuntimeError(f"Failed to find built zlib-ng library at {lib_path}")

    # Return the build directory for headers since that's where zlib.h is generated
    return str(lib_path), str(zlib_build_dir)


def build_zstd():
    """Build zstd from submodule."""
    zstd_dir = BGEN_DIR / "zstd"
    zstd_lib_dir = zstd_dir / "lib"
    zstd_build_dir = BUILD_DIR / "zstd"

    if not zstd_dir.exists() or not any(zstd_dir.iterdir()):
        raise RuntimeError(
            f"zstd submodule not found at {zstd_dir}\n\n"
            "The vendored compression libraries are missing. Please run:\n"
            "  git submodule update --init --recursive\n\n"
            "Or clone with submodules:\n"
            "  git clone --recursive https://github.com/mkanai/lazybgen.git\n"
        )

    zstd_build_dir.mkdir(parents=True, exist_ok=True)

    if sys.platform == "win32":
        cmake_args = [
            "cmake",
            "-S",
            str(zstd_dir / "build" / "cmake"),
            "-B",
            str(zstd_build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DZSTD_BUILD_PROGRAMS=OFF",
            "-DZSTD_BUILD_SHARED=OFF",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        ]
        try:
            subprocess.check_call(cmake_args)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"CMake configuration failed with error: {e}\n\n"
                "Please ensure CMake is installed:\n"
                "  Ubuntu/Debian: sudo apt-get install cmake\n"
                "  macOS: brew install cmake\n"
                "  pip: pip install cmake\n"
            )
        subprocess.check_call(["cmake", "--build", str(zstd_build_dir), "--config", "Release"])
        lib_path = zstd_build_dir / "lib" / "Release" / "zstd_static.lib"
    else:
        # On Unix-like systems, compile directly
        sources = [
            "common/debug.c",
            "common/entropy_common.c",
            "common/error_private.c",
            "common/fse_decompress.c",
            "common/pool.c",
            "common/threading.c",
            "common/xxhash.c",
            "common/zstd_common.c",
            "compress/fse_compress.c",
            "compress/hist.c",
            "compress/huf_compress.c",
            "compress/zstd_compress.c",
            "compress/zstd_compress_literals.c",
            "compress/zstd_compress_sequences.c",
            "compress/zstd_compress_superblock.c",
            "compress/zstd_double_fast.c",
            "compress/zstd_fast.c",
            "compress/zstd_lazy.c",
            "compress/zstd_ldm.c",
            "compress/zstd_opt.c",
            "compress/zstdmt_compress.c",
            "decompress/huf_decompress.c",
            "decompress/zstd_ddict.c",
            "decompress/zstd_decompress.c",
            "decompress/zstd_decompress_block.c",
            "dictBuilder/cover.c",
            "dictBuilder/divsufsort.c",
            "dictBuilder/fastcover.c",
            "dictBuilder/zdict.c",
        ]

        asm_sources = []
        if platform.machine() in ["x86_64", "AMD64"] and sys.platform == "linux":
            asm_sources.append("decompress/huf_decompress_amd64.S")

        compiler = new_compiler()
        if sys.platform == "darwin":
            compiler.compiler_so[0] = "clang"
            compiler.compiler[0] = "clang"

        objects = []
        for src in sources:
            src_path = zstd_lib_dir / src
            compile_args = ["-O3", "-fPIC", "-I" + str(zstd_lib_dir), "-I" + str(zstd_lib_dir / "common")]
            if sys.platform == "darwin":
                compile_args.extend(["-stdlib=libc++"])

            obj_files = compiler.compile([str(src_path)], output_dir=str(zstd_build_dir), extra_preargs=compile_args)
            if obj_files:
                objects.extend(obj_files)

        for asm_src in asm_sources:
            asm_path = zstd_lib_dir / asm_src
            asm_obj_path = zstd_build_dir / (asm_src.replace("/", "_").replace(".S", ".o"))
            asm_compile_cmd = ["gcc", "-c", "-fPIC", str(asm_path), "-o", str(asm_obj_path)]
            subprocess.check_call(asm_compile_cmd)
            objects.append(str(asm_obj_path))

        lib_path = zstd_build_dir / "libzstd.a"
        subprocess.check_call(["ar", "rcs", str(lib_path)] + objects)

    if not lib_path.exists():
        raise RuntimeError(f"Failed to find built zstd library at {lib_path}")

    # For zstd, the headers are in the source lib directory
    return str(lib_path), str(zstd_lib_dir)


class CustomBuildExt(build_ext):
    """Custom build extension to build vendored libraries first."""

    def _write_build_config(self, backend_type):
        """Write the build-configuration module that backs lazybgen.get_build_info().

        Written to two places: the source tree (for in-place / editable installs)
        and, when building a wheel, the build output tree. build_py copies the
        package into build_lib BEFORE build_ext runs, so writing only to the
        source dir would leave the generated module out of the wheel and
        get_build_info() would fall back to "unknown".
        """
        config_content = f'''# Auto-generated during build - DO NOT EDIT
# This file records which compression backend was used during compilation

COMPRESSION_BACKEND = "{backend_type}"


def get_build_info():
    """Get build-time configuration."""
    if COMPRESSION_BACKEND == "vendored":
        return {{
            "type": "vendored",
            "zlib": "zlib-ng (zlib-compatible, optimized)",
            "zstd": "zstd 1.5.7",
            "note": "Using vendored high-performance compression libraries",
        }}
    else:
        return {{
            "type": "system",
            "zlib": "System zlib",
            "zstd": "System zstd",
            "note": "Using system compression libraries",
        }}
'''
        targets = [BGEN_DIR / "_build_config.py"]
        if getattr(self, "build_lib", None):
            targets.append(Path(self.build_lib) / "lazybgen" / "_build_config.py")
        for config_path in targets:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                f.write(config_content)
            print(f"Wrote build configuration to {config_path}")

    def run(self):
        use_system_libs = os.environ.get("LAZYBGEN_USE_SYSTEM_LIBS", "").lower() in ("1", "true", "yes")

        if use_system_libs:
            print("Using system libraries as requested via LAZYBGEN_USE_SYSTEM_LIBS")
            for ext in self.extensions:
                if "bgen" in ext.name:
                    ext.libraries.extend(["z", "zstd"])

            self._write_build_config("system")
        else:
            try:
                print("Building vendored compression libraries...")
                zlib_lib, zlib_include = build_zlib_ng()
                zstd_lib, zstd_include = build_zstd()

                for ext in self.extensions:
                    if "bgen" in ext.name:
                        ext.include_dirs = [d for d in ext.include_dirs if "zlib-ng" not in d and "zstd/lib" not in d]
                        ext.include_dirs.extend([zlib_include, zstd_include])
                        ext.extra_objects.extend([zlib_lib, zstd_lib])

                print("Successfully built vendored compression libraries")
                self._write_build_config("vendored")

            except Exception as e:
                error_msg = f"""
Failed to build vendored compression libraries: {e}

lazybgen requires building zlib-ng and zstd from source for optimal
performance and consistency. The build failed with the above error.

Possible solutions:
1. Ensure you have CMake installed: pip install cmake
2. Ensure you have a C++ compiler installed
3. Check the error message above for specific issues

If you want to use system libraries instead (not recommended), you can:
  LAZYBGEN_USE_SYSTEM_LIBS=1 pip install lazybgen
"""
                raise RuntimeError(error_msg)

        build_ext.run(self)


# Common define macros for NumPy compatibility
NUMPY_MACROS = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"), ("NPY_1_7_API_VERSION", "0x00000007")]

# Define Cython extensions
extensions = [
    # High-performance BGEN reader
    Extension(
        "lazybgen.reader",
        [
            "lazybgen/reader.pyx",
            # Core implementation
            "lazybgen/bgen_reader_impl.cpp",
            # BGI index
            "lazybgen/index/bgi_reader.cpp",
            # Format parsers
            "lazybgen/format/bgen_header.cpp",
            "lazybgen/format/variant_parser.cpp",
            "lazybgen/format/genotype_parser.cpp",
            "lazybgen/format/genotype_parser_simd.cpp",
            # IO
            "lazybgen/io/fsspec_file_reader.cpp",
            # Decompression architecture
            "lazybgen/decompress/decompressor_factory.cpp",
            "lazybgen/decompress/compression_utils.cpp",
            "lazybgen/decompress/sequential_decompressor.cpp",
        ],
        include_dirs=[
            np.get_include(),
            "lazybgen",  # Base directory
            "lazybgen/io",  # IO headers including reader_interface.h
            "lazybgen/index",  # BGI reader headers
            "lazybgen/format",  # Format headers
            "lazybgen/decompress",  # Decompressor headers
            "lazybgen/zlib-ng",  # zlib-ng headers
            "lazybgen/zstd/lib",  # zstd headers
        ],
        libraries=["sqlite3"],
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
        define_macros=NUMPY_MACROS,
        language="c++",
    ),
]

ext_modules = cythonize(
    extensions,
    compiler_directives={
        "language_level": "3",
        "boundscheck": False,
        "wraparound": False,
        "nonecheck": False,
        "cdivision": True,
    },
)

if __name__ == "__main__":
    setup(
        ext_modules=ext_modules,
        cmdclass={"build_ext": CustomBuildExt},
        zip_safe=False,
    )
