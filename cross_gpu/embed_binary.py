"""Embedded kernel payload registry, packing CLI, runtime cache, and compiler adapters."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence
import zlib
import ctypes

from cross_gpu.backends.base import bind_c_symbol, bind_optional_c_symbol, load_shared_library
from cross_gpu.exit_codes import (
    ExitCode,
    KernelCompilationError,
    KernelNotFoundError,
    UnsupportedKernelError,
)
from cross_gpu.models import DeviceCapabilities
from cross_gpu.workloads import WORKLOAD_SPECS_BY_NAME, device_supports_workload

KERNELS_DIR = Path(__file__).resolve().parent / "kernels"

NVRTC_LIB_CANDIDATES: tuple[str, ...] = ("libnvrtc.so", "libnvrtc.so.12", "libnvrtc.so.11.2")
HIPRTC_LIB_CANDIDATES: tuple[str, ...] = (
    "libhiprtc.so",
    "libhiprtc.so.6",
    "libhiprtc.so.5",
    "libamdhip64.so",
)

BACKEND_EXTENSION_FORMATS: dict[str, dict[str, str]] = {
    "cuda": {".cu": "cuda_source", ".ptx": "ptx", ".cubin": "cubin"},
    "hip": {".hip": "hip_source", ".cu": "hip_source", ".hsaco": "hsaco"},
    "levelzero": {".cl": "opencl_source", ".spv": "spirv"},
    "metal": {".metal": "msl_source", ".metallib": "metallib"},
    "opencl": {".cl": "opencl_source", ".bin": "opencl_binary"},
}

PRECOMPILED_FORMATS: frozenset[str] = frozenset(
    {"ptx", "cubin", "hsaco", "spirv", "metallib", "opencl_binary"}
)

SPIRV_MAGIC_LE = b"\x03\x02\x23\x07"
SPIRV_MAGIC_BE = b"\x07\x23\x02\x03"
METALLIB_MAGIC = b"MTLB"

# BEGIN EMBEDDED_KERNELS
EMBEDDED_KERNELS: dict[str, dict[str, Any]] = {
    "cuda:bandwidth_d2d": {
        "backend": "cuda",
        "entry_point": "kernel_bandwidth_d2d",
        "entry_points": [
            "kernel_bandwidth_d2d"
        ],
        "filename": "bandwidth_d2d.cu",
        "format": "cuda_source",
        "sha256": "5c1493d7099e464bb00502c3b2825c2ce6d869d0fdc2d1e48a75efe0996a649f",
        "size_bytes": 692,
        "workload": "bandwidth_d2d",
        "zlib_b64": "eNptUV1PwjAUfd+vuPFpfEnAZTFDHvjYA4kao/hgjGm2toyG0ZK2AxbDf/euAxG1D0t77znnnnvWbXrQBKqVMZ1sU8CKa8nzCNJEsp1gdklYn4FPC5Y0KmQsrS5ho4S0vmlERzy5gFe40XgWgV8gLGgCM7YNVElj4VgxmrbdPQyIBVmsyZbTwE0YabqELsR7y6URSkJWJJpFkOQ5TF6nI0gQICynttDcoDWV5NxQzqDXv+2kwoKxmidrITOcuSmd6PhtHr+Qp/iZxPfxQ/w4j+CmD34v7KSl5bUrQBaDFlwWdxpnVRpdz7PlhjO+gAKNZRIn5gqHuE8RBgPP42haS7iaXAEhWa7SJCcEtkqwf4PyPcBzjIQQXMdqQS1SqsRc8yK1C0gVYc0Pg3OC8PmThp0MZw/xT4RBI80VXc3Y/nqPiZwrU7HGSquu2GWVggMNfilVkxk/iWX4qJl/tGriQmnXAMH2SEIfA3e9+zZbv1vDo/LJe3Vw/3dsfiAR93TXWvXgHbwvPizYnw=="
    },
    "cuda:compute_bf16": {
        "backend": "cuda",
        "entry_point": "kernel_compute_bf16",
        "entry_points": [
            "kernel_compute_bf16"
        ],
        "filename": "compute_bf16.cu",
        "format": "cuda_source",
        "sha256": "6a00229436475d66742e956fcdc4cac9e28251e6b43c84d02a50d02e084f5c55",
        "size_bytes": 4202,
        "workload": "compute_bf16",
        "zlib_b64": "eNqVV2tvqzgQ/Z5fMZvqVtC0KY+QVEm7q9w21a20dxv1Ia20Wlk8TIpKcNZAN1d7+993bEIC1EgFKcJjn5lzPAODc37SgxPwOUvTs9Umh1fKExpPwWfrTZ5R4oXmGDQ/D1xdABdJxn/AhkVJpqX6dAcnVbSAzb/eTUELY+ZmJ8DyjHh5eAo5etkWySDJ14TGdE2TLK1MRxnlbhaxBCelL6SUBsStWZ7UMef+C5zDYpvRJEUPWOUuD6ZAyPXzzZzMH66/EQK/XsGFYYCWYNQ3CuHaHfJkKERuLRnmfvlIlosH8vTtYTG/kcO7p8XDFC5wETd+AiO4Ase0QLuAwg9c309hK1Zvv8/FyILYTWgxuv39fpmK0Oe93lEUwi8BDaOEBlpNmA4/f0J9Ci6FVL13RDlnHPrPfzw+L5f3D0+LmykIXL0inP6TRxxJH7/DxdAYQIhOqm32MWISRGGvl/3YUFQDOSZshZIA0w65bc16vTRDTx+TF9C3yKeohhAM6NMoiVE92oiDjeu/kiJqUVqIWVmal0iH/3qAl0BixWfSOD+B5dOf4L9lB0XDcBcrBbbBgicBmEILg3yDtkisF4nnoly1itWY/XtYHYoMCwY3XWv9j/G/GKfwxcSfNevDFPpXvK+hKl2Mw76Gck/lIGa6XkjlNMt5Ukh//2xKMNVlRoSJT6q4ecXNr6YkmJVy4Y3FGDumWr9WqYpm/NkH3YFUjQNXiMa7t7v7De3BJ5UXJcuTSkFJmq+1ssg0KJX7+C5mUBYbXwVCxNtK3JTISU0r8HAMxvYWr1yHy0ss0k5Y1f8lUvg33A288vqmkHaAvnJrFF93nOpf9zHQKmaeG+N23lgUqBqRJsPsmhAh+LZkPPIz9Cg70r46tYa0n630o0Oosic1Z7x6zoT/CnVdgRcz//Uu2A632E6kcROt0RhA9sKpG8ilYsvYMDThhH2rqqiMfMhKAceMNJPskhAp8bkKNWNoOdh0C7lwJgCMh1ph4xNkDJ1QUSZvow7hNUJ4MoTViCHfAnSv9grUdCqEfQB6ZMPSBljQn0oRCnhCVw34mdiFyKt0q1rovi8kNxpuxtBE5VgCY2iMLelTtDRdkwXABXySj8F0crlN82LyWfhOtqQ1O9Oa3WhNJa3VmdbqRmspae3OtHY3WltJO+pMO+pGO1LSOp1pnW60jpJ23Jl23I12rKSddKaddKOdVGmL3oonGq1ow0huzMT9stKTZzAYRFm1Nx5tuLtau/hl4yyO99P7QBs34kUoOcITVy6CCKMaRrZZo+iB5X45fpvx6y471i4ve6jZgJrtUKsBtdqhdgNqt0NHDeioHeo0oE47dNyAjtuhkwZ0UoN+LrHY2z+bWAW0LbEKaFtiFdC2xCqgbYlVQNsSq4C2JbYOfW87DaQi2R/PedxQfPdTU401VVhLjbVUWFuNtVXYkRo7UmEdNdZRYcdqrOqYmk7U2En5JBdgj7EY2CtioxT/4WFb0lIDW9hxxTYbttWwbWHXCl5eB8yo4eM07HHDnqi2lK9RJ+rDPitEDXBsibFdjEdi7BTjsRiXQXbn5L+wRf8tIuB+a2T5WtfhNxl/+uFob2wnt9dGeaJ/7/0PrPB+6Q=="
    },
    "cuda:compute_fp16": {
        "backend": "cuda",
        "entry_point": "kernel_compute_fp16",
        "entry_points": [
            "kernel_compute_fp16"
        ],
        "filename": "compute_fp16.cu",
        "format": "cuda_source",
        "sha256": "d948e03d9307026a6cd504f1cb6c60894361552fe7aa941e2aaf5594439a01a3",
        "size_bytes": 4295,
        "workload": "compute_fp16",
        "zlib_b64": "eNqVWNtu4kgQfecraokS2YEQX7CJINkVmxBNpJ1JlMvTaGX50gYrxmbbNmKUyb9vdbcBm7QljBS5q/tUnapTpmxyed6Bc/BpmmUX81UB74QmJB6Dny5XRU6ccKXboPhF4KoMOEty+gtWaZTkSqaOS7hTRTPY9O+HMShhnLr5OaRF7nhF2IcCvUzDySEplg6JyZIkeVbZjnJC3TxKE9zkvpAREjhuzfJ4HlPqL+ASZpucJBl6wLxwaTAGx7l9u5s60+fbb44Df96AZWqgJBh1TSBcugOaDELd3hg8yuPTi/M0e3Zevz3Ppnd8+fA6ex7DFR5i3ecwBAyhG6BcwcKNQwNc389gww7vv0/ZyoDYTYhY3f/z+JSxyJedzkkUwh8BCaOEBEotLRV+/4b6FlyzRNXOCaE0pdB9+/Hy9vT0+Pw6uxsDw9X7Qcl/RUSR9OU7WAOzByE6SYrsYsAkiMJOJ/+1IpgMFKjWHDMC1BwK05h8PckWKcUz3Z50OlmOQX1UNSDryCeYqOMgl0+iJMbC0MYYsHL9d4cTipZDnG5btohU+OgAfjAgLLQ+LPSJsNER7wxhuNlS6frrvMx8EOLhKYJP9UkXxtC9WXSVhaayZdhV4lRVW/jpWz9Mpuq3TNcDr3T4ONXRy/gs3WhXwdy4n2Dui4W+DUBJXtBEFPB5rE7YmlImZuFtzS6euPg7ndAIdlnCOo0xdEyUbrWxZZEsZfwz91kHPGdcuCxlvHrl1T/IPDgyb9HGItn32MmKpbJtOwma+iscQ9wJdYnoH6KCT1ZCrcf9Wtcw85LmsOPYaqZFvd3Y5FCrtO1oJ33ndNjiUIMeLwHlIjhucKt720Wd5nHquTFKtE6jQDYIlb0O54jH72tOIz9Hj+1E3DW8NhB3u5V5WJG0nImHO962ET7i+Xcb5pjXDXhx6r8/BJvBBucZN+6iJRo9yBeUuAE/EiXjyFKYE87NakbbyHtVBBwV2ROKVPAGR0q8VUNFGxgWDn2RLlwwQEpDRdjYZm1ghaXU1QjeSh7COwjh8RDGQQz+xUL3ykjClPosry84z1mlWR3LyPs8BQk6IfM6+oKVwETlXlULvXddpFrdSxvomDXKrw002+AuYmyqChcfDwoVzkC3Cl6ifjU6Fq7uRyvV27Lq7Vh1KavRltVox2pIWc22rGY7VlPKOmzLOmzHOpSyWm1ZrXaslpTVbstqt2O1payjtqyjdqyjKqsYqPgipYjZi9zahF2vK4N4Ar1elFcH4smKuvOli49ImsbxbnsXaOVGVITiq2u4KlgQZlTD8NmqicFXlkvxIYUvCXxMlarskHodqTcjjTrSaEaadaTZjBzWkcNmpFVHWs1Iu460m5GjOnJUQx4lKE7yIwWVIBsElSAbBJUgGwSVIBsElSAbBJUgGwStIz+bHvkZE/nLCyLVJM/2TJdCdRnUkEINGdSUQk0ZdCiFDmVQSwq1ZFBbCrVl0JEUOtreuQLrpWkM6TtCowx/QOL4UTJ8rT07q9j6gW0c2Caza33efvaY4YGPdWDbB/ZIVlGxxDwxP5ynLKkerg22NsV6yNaWWNtsvQ1SvgT/xFH8L4uA9dbIiqWqwl88Pvtdz/5L4LiZw1kVbTO6v9XYp1DZq/n/g9d9mQ=="
    },
    "cuda:compute_fp32": {
        "backend": "cuda",
        "entry_point": "kernel_compute_fp32",
        "entry_points": [
            "kernel_compute_fp32"
        ],
        "filename": "compute_fp32.cu",
        "format": "cuda_source",
        "sha256": "be3b57ce225bf59315cf5a9ac7b1523e1b02f730e644a0208100b26f401b926d",
        "size_bytes": 2354,
        "workload": "compute_fp32",
        "zlib_b64": "eNqNVl1v2jAUfc+vuNqkKiktzQcJFbSbWEu1Sptadd3TNFlOcGjUJEaOXVFN/e+7diANEFTyAPfa5xxfH3wdzo4tOIZE8Ko6nS8UPDNRsnwECS8WSjKSLgIf7ETNqKOB01KKV1jwrJR25YxWcNJGa9jk2+0I7DTnVB4DV5LEKj0BhazAJxJKVRCWs4KVsmoNZ5IJKjNe4qDhQsXYjNCNLDZ1TETyBGcwXUpWVsiAuaJiNgKa53D1+3oCFAGol0glWAV2ibovDG7ucTc3PydG4+7+F7mfPpDH7w/TybUJbx+nDyM4x0kvwg8fLsEPI7DPgSaJKlROJRcVLBGCKrCgmcmM5ir4cXdfafkzy5KvCzZjKSgscV6yGeBGQQX+2LIYFi5K+HT1CQiZ5zymOSHwwrNZl6W2Bfis7CQEdyRFlkhkrL01AJTetLYZbTn7LrV2d3skduCfGUsQbwqGOdZ1CXHOk+fb2bK/RG9Mcp0VmPRAPglGZ2ZqbKhZCrYmfbncqGitrB/B8Lcpa/ib1VqwLoXigmlBU9vt+yEenrpYONXTXKR2nTsn4PbD1Bnv8GOy4FWXRrylERsNf49IyeYocqoX0ZvWomOr5Zhwcdrte8hHH9y+G/kGWR9+xzYu4IRy4Ai8UK0WWZG9g8heN9k/iOx3k4ODyEE3eXAQedBNDg8ih93k6CBy1E0eHkQetsg1mwuw6y7SAmP9fdFqqTH0eplsH+3PC0HnBcXGFzzPm+FGSN8btZSJLuBcaRGdtGVMl7jrMyzcE8Cr0BzB1b4akNeAvP0gvwH5+0FBAwr2gwYNaLAfFDagcD8oakDRftCwAQ03QB8Zhb37sVEdoF2jOkC7RnWAdo3qAO0a1QHaNaoDtGvUJuht97KNOc+BPyMxq9KsxFON/mEDHLVybyv3t/JA5xuVrJ93zGCLE27l0VY+7LiNK1VgnVgfdqkuqoexr+Ogjgc6Dus40vFaZPWW/IMN/lcr4H7bi6Gu48BXoz/C96v+Q0JoRcyqtrsc3ly5+tE3wpv1H2a7cvs="
    },
    "cuda:compute_fp4": {
        "backend": "cuda",
        "entry_point": "kernel_compute_fp4",
        "entry_points": [
            "kernel_compute_fp4"
        ],
        "filename": "compute_fp4.cu",
        "format": "cuda_source",
        "sha256": "baffa5c65acc8997b64b019bbfeffaf7b68178a1089c6c082bfabfda386cc2ef",
        "size_bytes": 3728,
        "workload": "compute_fp4",
        "zlib_b64": "eNrFV/tv2zYQ/l1/xc1FCil2FJG0HddON6RJihZYVyNtgQHDIOhBuUL0cPVIXbT533c8PUwl6YYUA0rD4vH43ccj73iyjw8NOISgyMvyaLOt4VoWmUyWEOTptq6kG22nYAZ16FkKd5lVxRfY5nFWmaW1bNGuBlaosxevl2BGSe5Vh5DXlevX0QRqNBLcrSCrU1cmMpVZVWrquJKFV8V5hkqyhVLK0PUGI5/cOCuCj3AMl7tKZiVawKb2inAJrnv+4eLMPbs6f+W68OtzYI7jgPki8YLrzzJJIMMFbiRInjLwkzy4PioDL5EhvHlzRsxv1+/c9eWV+/7V1eXZBYmv319eLWGKkwv8mhwfbN6O5lN0Q3ALcKm5WOBRCd6wI2EJO5gxDi9/f7sujxMvk2qJY8N4EkfwSyijOJOhOfDZgm/fYKiCU9qFZTyRRZEXMPrwx7sP6/Xbq/eXF0tQwEGsCvmpjgtZwrs3aGc7Y237FkRIoJ/By/VUeTpC8iyMI8OovmwlegY1nusG3QOMDtSCrwyjrNAwwDMO5U0cSPTMdZEvkHGW4E5wfJPHIaSp56Zsni2u51NXLWIagI1i+BRCZ9KLbC/yvSgmhMc1wUMw9aztedtrGL/F+GyyXweCbhkIWC/xXtLsKf4qyfYD34KvNO+VKe4pwW0nstmFaiPcoV1+yQLbS+iM7G67dpF/toM8sa/jLFwu0100zW6iqU2Z5hK53SxxI4PlcvrnaE9qR4Lb6riahxrVcpoK0DBfD3BbB7ihA9zKgbidwNeDKUoz/M7xe0KaBUrPSGIEV3imDJiyGNCxqcIhymnws3642uOWMHoejczQsSatxHqJ95KwdINiZHqEVwLrBN4Jwpr0aCCV36F9NpxD7oDmlMA6gXfCfaY2oC1dF1FrZdwahsSCUWQwOh9h7m6S3PeSLm3vVzItbw8RjneqKuKgQoOupPVJNKhovVYraFpqtkXtrqZPugDxdOVgg249b8rU63Bn77De0OAiTnEwhupjIb2QpvB6KtvjQ1jjXcdru9g1FzyLfT+R5RKcHUeyMbNVeHdMyY49U/IzlI9QXkGTcM5OLFQ9sx1bVauhT5574yU4i3Rtq1d3IL67zUuCsLY9AMnkhiDP2nYPUmdx1VwZwolF86nbnTbA5vx8rySQPYvwVBybY4+FujloOFKovIjasWXBWMuYQRua+ndMfWWKGGfeomh1y1RhegpsVlutb10JclT00LcJyqyViYEhgVJyXdnqhK4TJ7NopXMyR58mCzagnjc0bEB90ugG1Iu71FyjxtgrC850Ves157qy1Qldd89rMaAmCzGgbr0WA+rGazGgbrxuuPFVZjbXTAV/pfpT7c6tYDyOq+5WqfZkW3ib1MPUKvIk6dU90daLi4aKpFOY1opEDXQa1e6/5DDcFGeKKwVy0lyW73R0S/ruIev9DdBla/VfjqiqH6iqH6iqj2F/lCMPWP+oI9yhHKKcoSR5lCMPWP+oI8KhjKMMo5R6lCMPWH/Xkf89SbBO9t1PTZKBIz8zSQaO/MwkGTjymCTpnLg1mid1+JeA3iSngx8Tet3RX3llnWKlUvmEZVG9Xcb0OlFPod5SKsBqxGiG0QwT//Lqw6ZioXCcbDjZ8JZN0IygGUEzQmg7aX8Q/YUb+Bv9ikv8a4OV2EQ3LfiNvFV/ztRfPdcrXdqD6exOXp47qtUt1S3+RvsHtvrtAg=="
    },
    "cuda:compute_fp64": {
        "backend": "cuda",
        "entry_point": "kernel_compute_fp64",
        "entry_points": [
            "kernel_compute_fp64"
        ],
        "filename": "compute_fp64.cu",
        "format": "cuda_source",
        "sha256": "8db3a406a5bfd0ae11ad301e788646266323c67aa92181dcf30b37cbde69640b",
        "size_bytes": 2439,
        "workload": "compute_fp64",
        "zlib_b64": "eNqNVWFvmzAQ/c6vOG1SBW2aEhJIlaybsjbVKnVa1XWfpsky4FBUgpGxu1RT//vOBjoSghakhDv7vcfdwzZnxxYcQyR4WZ4mhYInJnKWzSDi60JJRlZFMAE7UjF1NHCZS/ECBU9zaZfOrIaTNlrDFp9vZmDHXIUZOwauJAnVagAKaWOPSMjVmrCMrVkuy9ZwKpmgMuU5DlZkKBmLCd1OQ1PKQkSPcAbLjWR5iRxIFBXxDGiWweWPqwVQBKBiJJVgJfxO5SPkqP7M4PoOm7r+utAy3+6+k7vlPXn4cr9cXJnw5mF5P4NznBwF+OfBBXh+APY50ChSa5VRyUUJG4SgCBQ0NZmnszq4/XZX6irPLEu+FCxmK1BYZZKzGLBbUGNvblkMaxc5vLt8B4QkGQ9pRgg88zTeZ6xtAV6NqYRgV1KkkURK47BBoPa2wW+jLX9bWo3HnaHQgT9mMEKGqRkSLO0CwoxHTzfxZrhBe0xyla4xOQH5KBiNzdTcUNMV2Jr08WKrpkZZX4LhG8or+KtlbmfHcMlxqdHIvC+aaZ+w8jwxjj8xVpTARZjKEkKu8tjYCj/d4WgA3tD9BSsu2DMTQ/0K/rVQd0exh9Wa2u7Q8wd1+3AKq4xzYVepMwB36DvzLjkkBS+7AuG2QGgEvB6FnCWocIpP0AZqwbnVtl+4OI3NeD566g7dwNPAekM5tnEUJ5QDRzDyVf2Qhjw6hDzqIXuHkL0e8vgQ8riHPDmEPOkh+4eQ/R5ycAg56CFPDyFPW2TDxvUJdrUnNX+u7x9aG3QOJyepbG+T94WgyZriOSJ4lr0NvwnpY6iSMtEHOFdaRCdtGbPj3Hr9CncAeLiaFVi39YYZNZhRP8ZrMF4/Ztxgxv2YSYOZ9GP8BuP3Y4IGE/Rjpg1muoX5j0G4Y/9r0B5Mx6A9mI5BezAdg/ZgOgbtwXQM2oPpGLSNeW0f1NW5FnKeAX9CXlqu0hxXMRqHC/6olY92cm8nH+t8q5Dm+oeZ7HD8nTzYyaf7Tt9SrbFQLBC3pa7qBGNPx+MqnujYr+JAx41K/ZH9iTv6l1bAhttPQ13HgU9Gf4af54znif4RWpLq0ba7mV5fn7ut68ftLaq/Wn8BOIeMpg=="
    },
    "cuda:compute_fp8": {
        "backend": "cuda",
        "entry_point": "kernel_compute_fp8",
        "entry_points": [
            "kernel_compute_fp8"
        ],
        "filename": "compute_fp8.cu",
        "format": "cuda_source",
        "sha256": "f87830a590aa569306cb40bffdd4316775a927e7be489c2e1eca1dbd66cda4a8",
        "size_bytes": 3376,
        "workload": "compute_fp8",
        "zlib_b64": "eNq9V21P4zgQ/p5fMVfEKqElxDEtpYU7FSgC6TgqXj6dTlFenBKRJl0nYcux/PcbT5KS0L1DK51w1Xo8fuaZsceepHs7GuyAL9Ms250vC3gUMhHxCPx0sSxy4YTLIeh+EbiGwk2TXD7DMo2SXM+MUYV2GmCFmpxcjkAP49TNdyAtcscrwh4UaMRtJ4ekWDgiFguR5FlDHeVCunmUJqgkW8iECBy3NfIojIn0H2APpqtcJBlawLxwZTACxzm9P5s4k5vTC8eBX49heGiBPgncvYt0uRQSEnTwJEDsLzhcXU2I7Hp268ymN87dxc10ckbi5d30ZgT7ODnEr27jDxtUI26jZ24bgOzs0AYdFTVfBiuw+wM4//16lu3FbiKUhz1N24pC+CUQYZSIQG9FacD379BWwZGK29C2hJSphM79H7f3s9n1zd30bAQK10qOFF+LSIoMbq9gaB52IUzb6zyfDVVsECVZLgufdriD5EkQhZqWPy8FBgYFbuQco0NYDgW3x5qW5cji46YG4inyBQbmOEjuiyiJcSE4fkqjABYL11mwQTJ8xDQqj7oG2ChpXyCwemuRvYn2m8h7hEef4CKYelb1dtU3MF6F8VjvzQ/4tRvw2Vqy1xI34IXQbrbAqGNcWCzKOFXr4BrM7DnxTTemXTDrBZky/Wb6aWyGKKvFlT84gs6b+cs2ut9Gx9vocpu/9uBlex+lPn4H+D0gzRClQ5IYwRWeKQPGX8dvbCPoHIcdPbCMXiWxtWSvJW40DWRHdwmvBFYLdi1wo7dGA6m8Gu2x9hxy+zSnBFYLdi1wwxhrr5om8OrJBDqnHTwU8zj13Lg+D5s1oXEgdhCOhzWXkZ+jQV0c1tlt1Ya1tlEaGjmvysN7jVfn2kc8nWWYY1jH4MWp/3gZrMwVXmManEULHHQhf5DCDWgKz72y3cNat3L9vLxCIvHTIErm2QisFR8iV5eZmEMcWGpgmX01OFGDXRyY5Y7uwMzFqxmAXiLUlFHWCTcIkGNgWuhbJLBL4rcof4C/hUxBpkWiHEIgozA3VQ1pr8h1ntwYvalwyk8xfgfxnGWalRCr/PwAkog5QU6s8lNU6y8x5a56biYUyOyHuFeWaWOPVbHcfthVqFSG1dgwoNs4T63WNvXemXrKFDHWoEKRd0NXyfsCrF8YVWz1jVe7rWLrocwqmRgYEiil3VRWOt7U8YN+OG5yMqs5TRasRT0oaViL+qDUtaiH76ntBjUeHmVhs6aqitq2m8pKx5u6jah5i5oseIu6ipq3qMuoeYu6jLrkxseIXl4+lfyx6o8aN3EM3W6U13dNta2ldOcLFx8lMo3jtXpNtMTLUFKRdAT7hSJRgyaNapvPFEw35ZnySonslZfgXzo6/etu09oYf+RQFWlfFWlfFWlM70853LD+2KFt0ZmgM0BJ/ymHG9YfO+QWnRQ6GXQUfsrhhnV9P//PJGJ9WnefksSWw89IYsvhZySx5fBHSaydvWrlL3X4AkuV+Kj1iG7e2+YjIysWeNNVvrGsqOrcpXLcpaRhpxKjRoxmGM0w/h+PDmxqbxXOJhubbOyKjdMMpxlOM62VVK8Zf+IC/sK4ogxfxLGS6RimAb9RtOrPg/or4riZQ2vQrdXB+amlWlFRveKbzz+JInwA"
    },
    "cuda:compute_int16": {
        "backend": "cuda",
        "entry_point": "kernel_compute_int16",
        "entry_points": [
            "kernel_compute_int16"
        ],
        "filename": "compute_int16.cu",
        "format": "cuda_source",
        "sha256": "22412b5d7e6de17938046a42579ec187c5e2c28e8a847730571f7c2d4697d2ab",
        "size_bytes": 1878,
        "workload": "compute_int16",
        "zlib_b64": "eNqVVNFu2jAUfc9XXHXSlMA6IIHASJnECtV4mFax7mlaIycxEDXEkWNXVBv/vmsTSCAIjbzk+vicY9/ra7caBjQg5CzPb5eZhBfKU5oMIWTrTArqx6nouGCGMiKWYk5Twd8gY4ibuTUs+P4RXfHGX2ZDMCWOHdsXDWBS+IFcfIA9BKlc+zSha5qKvALHgnIiYpZWwZzSyCenQKB3NObhClow3Qia5qiDpSQ8GgJJErj/ORkDQQK6hkJymoOZovsrhY57G8QCJGqWKY3g23ii7b4//vAfp3P/6et8Op7ocPY0nQ9hgJNYigbYMAK7h0UZAAlDuZYJEYznsFHTMuUsSXZ+CrKBZbkybhmGeMtoRBflmpgNSMf26jP5inGc67ieYVDMjKdwc38Dvr9MWEAS34dXFkdnq28agB/aNpCOGQsehwIF+xPYTx8fwAGt1P+AFeU/HgcW/NFIiFydByxxSyMIEha+zKLNxw0WSw8m8RoHTRArTkmkpzwtjRdgKtHn0dFu9s7q4xTPLd3Rt0Z1Qaw21h4XNDG0THO3S3gP7c0DftKCv9CRlneiIVF0ognOaHa5Ip23S7beaxMJmFh782nq9FHwXJTHqrh4pbxTk9vXyO2a3LlG7tTk3Wvk3Zq8d428V5O718jdmrx/jbxfkw/+Q671C8ZRqm8DmrQ99b+rXA0Pms1YVNv0XcbJck2K+3+AD0a5oNnOSkd3+FZI5aJGVR/d8mXLYdjQXd5UfVtkd+CVvYXhBV7ZRBhe4JXdguEFXtkWGF7gleeP4QVeedDcvcQrTxTD87zt+YcCH+qyG/aVfcYKWuhwKA0CTgXoKqBXAVwF9K1iueJB/YW99VubO7aF63jG1vgHlL3zJQ=="
    },
    "cuda:compute_int32": {
        "backend": "cuda",
        "entry_point": "kernel_compute_int32",
        "entry_points": [
            "kernel_compute_int32"
        ],
        "filename": "compute_int32.cu",
        "format": "cuda_source",
        "sha256": "1e21974a06857545fec1c8cf52c39e4f78429fdb355cf4339d3670c1c755b1e0",
        "size_bytes": 1579,
        "workload": "compute_int32",
        "zlib_b64": "eNqNlN1vmzAQwN/5K07dCyTrkkJCPmgmpQ3S8jCtyrqnaUUGHIrKl4xdpdryv+9sSCBRU8HDcT7/7ny+Oxj0NOhBwPKyvI4KAS+UZTSZQ5CnheDUizNumaAHIiSGJN2MszcocrTrpTGvee8El9zybj0HXai1x3uQC+75YvsZDibIROrRhKY042XLHHPKCI/zrG0sKQ09cm7wVUZLFjzDANwdp1mJfhAJwsI5kCSB+1+rJRAEMGrABaMl6BlGf6Vgmdd+zEGgT5TREL4vVyrcj4ef3oO78R6/bdzlSqnrR3czhylu3tgoTFiAObZBnwIJApGKhPCclbCT2yJjeZJU8aTJhLwoZeCBpvG3goZ025yJtwFhmY6mUcyeZXB1fwWeFyW5TxLPg9c8Dt+tsK4BPujaQxxvxVkccHQ4VPmwfVrko7VV46OtLvHp2jfgr7IEyKpcIcKUFuAnefCyDndfdlgQtVjFKS76wJ8ZJaHacpRrvAVdOn1dnGRziCwfRrE3WYXvtbMDsb54YJUe/IMb4ZwBJAwPgF8Dx1uwIW6p4/u4gbkOdzPXmkxmdzNhwFMd1mn4m4Y3u/Bmw1tdeKvhR134UcOPu/Djhre78HbDT7rwk4affsArh23O8Beg5g29ho5837aGz4F+P+btQfhUMBKlpP6KjuZjoJLTogqltFv84oSMIlftOGqoZOtR9NQE9eWYOKeA7DWKy4BsLorLgOwmisuAbB+Ky4DsF4rLgGwQsz8AZEdQvAvs259U/XP4jd37I7uItXnC+xvopOMtUbcqfST1caXbUp8YjrbX/gN9qaoe"
    },
    "cuda:compute_int4": {
        "backend": "cuda",
        "entry_point": "kernel_compute_int4",
        "entry_points": [
            "kernel_compute_int4"
        ],
        "filename": "compute_int4.cu",
        "format": "cuda_source",
        "sha256": "dfea42fe031d7587a3faf0b27c90d989839d1b040ba3777896d3aca40acf343c",
        "size_bytes": 2637,
        "workload": "compute_int4",
        "zlib_b64": "eNqlVW1T4zYQ/u5fsQ3D1SYhsZ1wpAl0JkA6xwd6DC+fOj2PLCvBg2K7khzCcPz3riQ7do7S61w1xF6vnme12jcGBw4cABW5lIfLooRHJjLGJ0DzVVEqFqWZGoFLy4R4GjjPlHiGIke1K71JBY/aaA2bnV1OwC3xcxhG6gDyUkVxuehBrYKsXEWMsxXLlGypU8UEUWmetZWSsSQi3ypi49BM0AcYwHyjWCaRB8uSiGQCUXR+fzGLZjfnn6IIfj2F4yMf3AxtrxmsVqQvnzPaJzxdZizpr8bZ+HEYghwZo5+vb6Pr+U109+lmPrsw4uXd/GYCI9wc488NK0H/kDfAhwenMPJ/+QguKp6IKNAaXF3NJGwgCMeQF3LAScb0CQPH2UsX8FPCFime7+5468HXr7CrghPtv+fsMSFyAZ3732/vr68/39zNLyagcbv5EuyvMhVMwu0VHPePurBAUnV36xSkmVSipCbUHbSbJenCcdRzwdAnKDGWOjAIU1AOw+l2RytSrXCkQoMUA52wdUoZOhlFeA5lacbxUvi9ztNExzqqwhvJkesALjTwARK/VwlBT58BxLfv2G4Ard+BBy+GR+QKjXI8lzNrSa/Oe+nsi/ypT3PelyjLkflDe52G+bKPZ+wHrz142Q/Nc2ieI9QevU4b5AQ6p6LjJr7Xq6TAa2+ihpg9FOJaoFsh8Lyp8+o4DMtUZNA572CwljyPCa/j9A+dZK+IITlANKZTiZQqxNfdVG/vNtNW2+qlra5qpd3vuI4vRazJNyzRo1OIeU4fL5NNf4Nlbj4u0hV+dEE9CEYSs4WloLmDA7gm9BFrZryBqnpGh3GKrZ7GMWdyAt0AbfobTPehlX4DVz3lP0tTvdZ/r2/Dqq1hCSfgds8Qf+bZVqIko4wD5Yxk/Bk30ddD3YFozgOVQyHYGs1AvmZiwfOn2trTA9YMBpkVabYExOAYs44BoaYxiBmCeDMGDzhCsINNsyREkYKoh77u2t0gkWhNuL1StUrAznVtkOED+KUHJycwwuTvMuOoyOV7zPjfmRlb2uBVq4QvsEOsEmI5uoFiIhlSXJQ919Wp/VIVQi1gBXyoTJbVkbYDfV0Fmt4FTBv1g+Y7bOGCFm6IuKCFG7VwYQt3hLiwhfvYwg1buGPEDVu4cVNwM86hHrZ6tEpgG0axfbYTHidZikNphbWCxUFMpxVEqJTwQ0NLsIx1YW/Tq0ela9tHh3mq3yetXppCt5uqumP02isEWa4IHiVyzrfqraECC9maMhJmtdRG9EfbjF7fDEuMvwl6z5Zaz9aNVll9lar32IFvUvGGXem/ww59k6A37Er/HfbQN2l7w670dZH+94tj2f+Pi1v2j17csn/04pbdXLxmvjr2act+AaYzT3aGebs+mhkgy5VuZxQ93aHdWgy2YtBosRW7Ow63l4WEDTpsbAwb7TBovK7++fyBzv6JXqAv0+oer87fvjzQ7g=="
    },
    "cuda:compute_int64": {
        "backend": "cuda",
        "entry_point": "kernel_compute_int64",
        "entry_points": [
            "kernel_compute_int64"
        ],
        "filename": "compute_int64.cu",
        "format": "cuda_source",
        "sha256": "9e16d7b06aa787a172bb78c4106e6748c5440e2a8700014772693c50df4798b9",
        "size_bytes": 1750,
        "workload": "compute_int64",
        "zlib_b64": "eNqVVFFP2zAQfs+vOLGXpB2DJmkCDUwqtGhITEMMnqZhOYkbItI4cmxUtPHfd3ba4jI6rXm4nM/f9/kud85Bz4EeZIK37X7RKHhkombVCDI+b5RkpKxlFIKbqZx6GjmtpXiGhmPcbb3REk824Bo3PrscgavMmsgecCVJqmYfQYcCn0io1Zywis1ZLVsrXEomqCx5vQxqOrSM5YS+DaQmo7HIHuAApgvJ6hZ5UCgq8hHQqoLzu8kYKAJQNZNKsBbcGtWfGEThflpKUMgpapbD1/HEyH27/k6upzfk9svNdDwx7uXt9GYER7g5iND4cAr+MAL3CGiWqbmqqOSihYXeVrXgVdXp6ZAPvGm18IHjyOeG5Wz2embF66IzKgqTv/exWlCBnzgOw+pEDXvne0BIUfGUVoTAEy/zdzvgOoAPivYQjlVLUWYSCasudNuBv9mEddTqwUpo1YLNderBLxPJEGtyhQJTOoW04tnjZb74tMAPZhaTco6LPsgHwWhuthJDLWfgatLn041sVsr6EQx7V3fwF8c+EBPB748HdunBbxjcXV0lbyA0z1eQdAskpS1DjIu+h9kkzrpQcajjZr9vqFjR4eJ4GsTx8dlxfBGO4/PBEOMe3C/TSF7JA4vs70r2LXKwKzmwyOGu5NAiD3clDy1ytCs5ssjxruTYIh/9J9mwZ1xg783oo8Rhot8n1j1IoN8vpT2THxpBizldXvh1eC3UStZ0UsY7wZ+D0ip6ZeuY+dYjhqZnhrmv5zXZBOgxQrMdoEcFzXaAHgc02wG65Wi2A3Rb0WwH6NaJ6B8A3R407wJe7Nu9/E/9wJv4U7cUv8091u8hycUq0Q86P9T+sPMj7cde4rw4fwA+NdiT"
    },
    "cuda:compute_int8": {
        "backend": "cuda",
        "entry_point": "kernel_compute_int8",
        "entry_points": [
            "kernel_compute_int8"
        ],
        "filename": "compute_int8.cu",
        "format": "cuda_source",
        "sha256": "d3f7a1028714bda1226b0dbf29aeea0a15fb9d3879717667ad5bb9162a7946c9",
        "size_bytes": 2128,
        "workload": "compute_int8",
        "zlib_b64": "eNqNVF1v2jAUfc+vuKPqlEA/yAcJhXYSbZnah3UVbZ+m1QqJYVZDkjk2olv733ftAHPQNEEUfH1zzsn19YlP2xa0IeFFVR3PSwkvlOc0G0BSLEopKGG56IOdyDR2FHCcC/4KZYFpu3IGazgx0Qo2urwdgC1x6ntEtKGQgkzl7Ag2KcjlgtCMLmguKiPNBOWxYEVuJitKUxLvJqa6oBFPfsApjFeC5hXyYC5jng6AkKun6xEZTa5uCIFPFxC6XbBz1F5SSMsgPpG+p26t8vX+gdyPJ+TxZjIeXevw9nE8GUAfH7oh/vXhAtyuF4DdhzhJ5EJmsSh4BSv1XOa8yDKaamVM9aEoK6V8alkHbAYfUjpjOU3tRlkOvL1BMwXnqlDHOqCcFxxaT3cPT/f3XyeP4+sBKFxzYzj9KRmnFTx8gfDE7cAMSf9YZAsF85TNLEu8lhSLwYorNseKAHUAEUPLqgQSE+xcSpcsoVgMIaiXUJZnWDzOEadliezbKlZ7gsO0HhIHfluAPw0b6jCuFrAssFUso3bLLAkOu0dw6OLt4e0PWzCA1gVv2amjIgxi50iP0/WYOE4tyqmQPFeveLcsijuPk9ZVC8udZ8U0zrDSZcHSf5nT3hTYRjQ2TnCWCMRvDLqtv+HPbdaw5za3dmdzPt30IkGsbjDMsaILmGZF8nKbrk5WaCo9uWYLnHRA/OA0TvWjepnoG1uR0LxmNRvlv52o4dgLNZy24a7Ij39RXkAZJy+4xdu9nr4K9ArL4Rv23fWi79giWkJaiABNrV1DlxQ/8ErQ8kS5t7kEND1RmrgOu142fITuKvpcXxINjdOuW19yuEtnK2Suic/rRg2tbed4VwnrNXfAldig7ups7EfR2eUZaj8rAfWKjbLmuAbH25PjGRx/T45vcII9OYHB6e3J6RmccE9OaHCiPTmRwen/n6NJ6mSx628Amd2hGs+ND2IInQ4TpjkPSh7PF/H6cNymt0LKZLWUjs7xIJVKRc1MHW10ZY3N0cPx5NhY8QgfrY+FLdY1sW4D6+5iPRPrNbDeLtY3sX4D6+9iAxMbNLDBLrZnYnsNbG8XG5rYsIENd7GRiY0a2MjAvpvHx/og/IbG+K4cgo1/Vm1Dl2B/VOzXcaDiXh2HKlaS79Yf+fhIRg=="
    },
    "cuda:compute_tensor_bf16": {
        "backend": "cuda",
        "entry_point": "kernel_compute_tensor_bf16",
        "entry_points": [
            "kernel_compute_tensor_bf16"
        ],
        "filename": "compute_tensor_bf16.cu",
        "format": "cuda_source",
        "sha256": "fb3ec63cab56f09759a1e44092236d9a66ab0613ead273032f12b07e04172014",
        "size_bytes": 3082,
        "workload": "compute_tensor_bf16",
        "zlib_b64": "eNqtVltP60YQfvevmAZxZJPg2F4IOQm0ChB0kEpBXJ6qyvJlnWPh2KkvNIjDf+/M+BKbcIpUsZLtmdlvvh3P7I493FNgD7w0ybL9xaqAR5nGMpqAlyxXRS7tXMZZktpuYI5A9Qrf0Qg/j/P0GVZJGOdqpk0qL/sdJ0LPTi8noAZR4uR7kBS57RbBAAp0FpadQ1wsbRnJpYzzrGUOc5k6eZjEaGRfyKT0baejuRzOLPW+wxDma1oYPWBROKk/Ads+ezif2bPbs2+2Db+ewNgwQD29wHdZLh09e4493YnCRSx9fWmO4vGjOWLC65s7+2Z+a99/u53Pzlm8vJ/fTuAAJ8d4qRbekKbUUBiCsDQ4gQPjKyZKWMCrXF3NMliDaY3h4vfrm2wYObGkFYaKshMG8IsvgxBXVzuRavDjB3RNcEyxa8qOTNMkhd7DH3cPNzfXt/fz8wkQ7t2CpfLvIkxlBndXMNaNPgToG2NSn2QTXw85Yz8MFCV/XkmMBwrMIaUEsBJQCGuqKFmOTh7m05dPoScxHttGLk+GcYTxo/6UhD4l1a7zyBGoCuDgen0B3xg0orkRrY0oBozHNcGxn5xowKLL4oYJvJoIPLORrEYSGrww2smWGFeEoUeyjIRG76el19PkH91LIj0Qlk7hlzfUoLdxf9nF5Xdx4V1ccle8DuBl9wClQ7xGeB2xZYzSV5ZMhhPeJAdTvE43bBPonQQ91Te0QSWZjWQ1ktDaDmlP5ezQ9Mdy4wlsdVsIdxuBq3kcCwlmLVi1IDRtqrwqisSTlsbQO+vhRlhEietE9R74eStobYY9dMN9maehl6Nj3ROa6ndaQmNtdYTWbqi6wluLW+8CD/G8j2GB4Z2AGyXe46W/1td4clk5D5eo9CH/nkrH5ync8+Q7xFa3dry8PC0rJ0yzCRhrcTE2kKlv6lhbUg1WDf2Q1NNS3UdVp5PeDYLLgtMlC13F9A3EtVdJVkEMg653ILFcMIRWOy0hLUyZCNfJJIH0wwBfz9AtfGLvKjMG+4RK0qDSNQ36ra3QGV1X942rS66IMUYVilfXVMr3FzAPC62KrT6+lB6KbYCyWcnMYCIBGa22sbKJtk0cHQbTNqdptKfZw+xQj0oas0N9VNo61OO31FaLGitOHpbZNlVRW1bbWNlE27YVtehQs4foUFdRiw51GbXoUJdRl9zY5NXyvFDxp/Q8bh2eKfT7YV4fDxo7q9RZLB3s/GkSRY25IaKNX1KxdAwHBZGQ0qahsf0JwHJznbmuXMhB3d15o5PpLUKbfkRKXdWjrupRV8USbpFuIT4mtQyuLdeSi7dFuoX4mFQYXFWuIpdti3QLUZ+X/5lUbA2fn9SS9JOTWpJ+clJL0veSWhO+KuWdH/gbxp3quPPVae/rdkvNiiWeBMo/HjvqXn1uV31OMD4oiaSZPGPyjCn+o7XioBwRzmIfi32sik3wjOAZwTOdN6m+nH/iC/yFcYUZ/k7iSVcxTA1+42jpN5h+qm0ns/kdVGN9dHFm0Cgqqlf8qP8LcpYl2Q=="
    },
    "cuda:compute_tensor_fp16": {
        "backend": "cuda",
        "entry_point": "kernel_compute_tensor_fp16",
        "entry_points": [
            "kernel_compute_tensor_fp16"
        ],
        "filename": "compute_tensor_fp16.cu",
        "format": "cuda_source",
        "sha256": "3028142a3b8cfec8a068efbfa0934d53884d890d1057dd3789913d64939c1fd0",
        "size_bytes": 3577,
        "workload": "compute_tensor_fp16",
        "zlib_b64": "eNrdV21P40YQ/p5fMQ3iZJNgbG/eLoFWAYIOqZSIl09VZfllnbNw7NzapkFc/ntnx3ZYE0qlK/1SS8azs888OzuzO0OODlpwAL5Is+xwsSrggYuEx2Pw0+WqyLmT8yRLhROurAFofhG4usTPklw8wSqNklzL9HFl5bxhJNHT08sxaGGcuvkBpEXueEXYhQKNme3kkBRLh8d8yZM8U9RRzoWbR2mCSrKFjPPAcRsjj9yZCv8rHMFsLRdGC1gUrgjG4Dhn9+dTZ3pz9sVx4OcTGPZN0JbWIBk94H4QeHsFI8PsdKFUjird0OgT8fX81pnPbpy7Lzez6TmJl3ezmzH0zM+qPWjMhjVY9ggufr2eZ0exm3B9ArbZUygr1KCngnCZo1ZrLwrhp4CHUcIDreG2Dt+/Q1MFx3IjemuPC5EKaN//dns/n1/f3M3OxyBxb2ZP8G9FJHhWOdOBEG0TjPAjh4s5Aq6upm3kTIIobLXypxVHf6DAgC7QKcC0QMHsSenrTmRHptnKcmTzcS7gj5HPccJxcBGfR0mMG8PxYxoFsFy65JGT5XyltQAfyugnCMzuVrReRPtFZF3CoyPgOo9u3CXRI/GFCfyaCHxrK9lbienwTGg3W6JPMbod89IT+bTRQyN7SnzDjWnzRn1kDJH+afhpbITMNkIc04setF+Mn/dx8X1cdh8X3GebLjzv91Dq4zvAd0iaEUqfSbIILvGWNLDYZvLCNob2SdjWAlPvVpK1leytxHTVQLQ1io2c/md5awmk9RSEt4vA1XzyRQpWLdi1wHR90trgIYoz/r8/DqN/dRpIGtDfIY7pPEik+d+k/+00/0hqy/rAsdaKBNpnbczqIk49N64T+vfNQMnuAZphMcpF5OdoWHeFbTobTWGrVXqCkt6qL7zWeHVafcRT8YIFuncCXpz6D5fB2lhjiafBebTEQQfyr4K7AU1hoZO2R9js1q6fw+VsNjsc9nvgRYkrnrBertxIZGMw1+zMNJG2YxmYaxyOyqFp9OXwtBwe4tCQtb7pEaUHp0sW+RaTVxDPWaVZCUEq+b4BSfiCIHK10xKiYMqoeG7GJcjoh7hX07DxewBaGT44lKhUhNVY16GjHI/G0zT1Xpl60hQx5qBC0eq6JoP/Cax+oVe+1ZdThkf61kXZqmRisJBAKm1VWemYqmPDfjhROS1TnSYLq0E9KGmsBvWw1DWoR6+pbYUaMy4tbEtVVV7btqqsdEzV7XjNGtRkwRrUldesQV16zRrUpdclN7Z5rbw8MvkT+T1WbtIEOp0or++KfPZWwl0sXez9Io3jrXpLJA9+SUXSMfQKSSIHKo18msUdU005ppxSErt13aZDLlWvEfrkPULZPH3ZPH3ZPDF1O4Q7iPcJbZPySfmjhO0Q7iDeJ2QmZZGyRmnaIdxB1PfjB4KIZeBjg1gSfmAQS8IPDGJJ+FYQa7JNq/xLH/znlSrRcaPFqOdWLZlZscSTLmOO10pWpw6Vow4FFT8yeHJk0YxFMxZ7p3TiI+MjcTbZ2GRjV2yMZhjNMJpp7KRqk7/jBv5Av6IMfzDgTdbQTR1+IW/lrx75G8pxM4f2oJnr4QX2FXyKimqDjfwvXjWqZA=="
    },
    "cuda:compute_tensor_fp32": {
        "backend": "cuda",
        "entry_point": "kernel_compute_tensor_fp32",
        "entry_points": [
            "kernel_compute_tensor_fp32"
        ],
        "filename": "compute_tensor_fp32.cu",
        "format": "cuda_source",
        "sha256": "5d66927556205eb644349146e40139dc6ca640d48fe572bcc610ea101b338c3a",
        "size_bytes": 3361,
        "workload": "compute_tensor_fp32",
        "zlib_b64": "eNq1V1tT6zYQfs+v2IbhjE2CsSQuaQLtpBDmMFMKw+Wp0/H4Iud4cOxUtmkYDv+9q/UlNoEeHopmHK1Wu9+u9ElrZ2+nBzvgqzTLdufLAh6kSmQ8Bj9dLItcOrlMslQ54VJwMPwicE1tP0ty9QTLNEpyIzPHlZfzhpO2nv52MQYjjFM334G0yB2vCIdQoLPgTg5JsXBkLBcyybOWOsqlcvMoTVBJvpBJGThuZ+RROlPlf4M9mK10YPSAeeGqYAyOc3p/NnWmN6dfHQd+OYGRbYNxd45rWSxcK3tKfMuNo3kiA2vBDpPRw4jwrq5vnevZjXP39WY2PSPx4m52M4Z9nBzhY3D8YYfVaISxBTfhBLi9PwID4SnG5eU0gxUc7sP571fX2V7sJlLj7/V6W1EIPwUyjDC00UnThO/foauCY5242duSSqUK+vd/3N5fX1/d3M3OxqDt3mRLyb+LSMkMbi9hZNkDCNGX8rojKzhNldQ59hE4CaKw18uflhKTggJ3UW8KIBdQCD7p9bIcufBxRwP5GPkSk3IcBPRllMS4CByjHeSpk4eCl1zDyoTnHmDTU16UZxMaudnC6PuPuaUS19LmFj6wbQ9hm036MIb+ieob2t7Ug7BvrEyzdFUyL1RSYb18JKvHNAo02U7Fb5kfgVGSXyCwh43I1iJfi2LYrMJFY+pZ1fOqb9l4lY3Hhus44NdhwGeNxBtJ1HuFu4NJx7iuWJZ56tZ/77xaKv3H8tNY72G5mc2O9tfez+Xu4oMRt8XLEJ6391E6wOcQnyPSjFD6mSRG5tqeaQcmXiZrNE0QkhLY5rCSWCPxRhJm2wEJdcleC6wWeC0Ic9hYA6m82tpj3TnE9mlOC6wWeC0IfVTwZEisBXhS+qd9PBLzOPXcuD4N7xer1rnYQTe8PLmK/Bwd66rVkNwpWo22VbNa1Fd167XGqyn30Z7uGcwxvRPw4tR/uAhW1gqLCw3OogUOBpB/U9INaArvZNfXxaONzvUVZJYdVrdmbeM5yzRrGdnWwVtGiZy3jHYrq5ZZuQrPzSTa6WnMzbY49lgby+XCrrZKVViNTRMGLR47revqvXL1tCva2IeVFUU3Db1ZX4AdFHVu9UWz9R5ibkOUWSUTAkMAreRtZaUTbZ04OggnbUxmt6fJg3WgD0sY1oE+KnUd6NFraN6C1qShB2dtVZU1521lpRNt3UbWogNNHqIDXWUtOtBl1qIDXWZdYuOrwygPuyZ/ovvj1smfwGAQ5fXZ1m1rqdz5wsXXikrjuFE3QEs3UiUUScewX2gQPWjD6LZRyZFtoploJR6HdBPe+qXT33SbntVNeD+WLou+Lou+LotI7IdjbXj+MBa36SAQ8cT0h2NteP4wlrDpZNBJIOo/HGvDs76L/xNhWIma7rMJ68T6ZMI6sT6ZsE6stwir47z0yl/q8AOVKuxx51XXvo/tV0FWLPAGa3KxXOiqO6AyOyCWsNNU6BGjGUYzTPzHKwGb3lJtx8mHkw+v0ATNCJoRNNNZSfW6/hMX8BfmFWX4oY0VysA0TfiVstX/DvR/DcfNHFqDYa+Ozk9t3YoK6gW/JP4Ff+F9QQ=="
    },
    "cuda:compute_tensor_int8": {
        "backend": "cuda",
        "entry_point": "kernel_compute_tensor_int8",
        "entry_points": [
            "kernel_compute_tensor_int8"
        ],
        "filename": "compute_tensor_int8.cu",
        "format": "cuda_source",
        "sha256": "03e9089400b8c3c9b921cf9b30832521981eacb8a22e932bc0b415bbfc88f597",
        "size_bytes": 2988,
        "workload": "compute_tensor_int8",
        "zlib_b64": "eNqtlW1P40gMgL/nV/iKWCW0tMlMtxta9qQudLV8YBfx8ul0FyWTaTciTXp54cqx/PezJ0kzpaBKJwKltmM/M2OPzeDIgCMQWZrnx4tVCfcyS2Q8BpEuV2UhvUImeZp5UVK4YIoy9C3ynyVF9girFM1mbo3rKO+VIPKefrkYg1miyplXHEFaFl5QznvQmCApl56M5VImRa6Zo0JmfhGliW7MpQw9/6UhUPuaZuInDGC2ph1gHCxKPwvH4Hlnd+dTb3p99s3z4PfP4No2mBffb11YLv1+/piIvh9Hi0SG/aUzStx7zhTwx9WNdzW79m6/Xc+m50q8uJ1dj2GIL138mAz/OKNa4wxXx1DAFZwTBiYachcuL6c5rIF9HEG6ygexn0iiDwzjIJrDb6GcR7iyubVLC379gm0TnNK+LeNAZlmaQefu+83d1dWP69vZ+RjI79WqZfLvMspkDjeX4PbtLswxtjlkvbsOMpMwmhtG8biSuB8oMX+UDkAGlJxNNm/IEJHByAssjsDkhvIhEhI36HkIFzJKYjwQ6g9pFFKGvWY9L3dNA/BBwgcI7V4tOI3AGoH3lB8uDb734Mc9JQZKbAggKgAIp/5m9Te34El5+fkSdxHjRmNZrUxP582q97P0n75I436Ocu6qXwR22tCnQ1z0EBc8xMUO+XMPng6HKH3Ezwg/n5TFRelESY5yJ3+HAhz+PGlpY+h8zjpmaFu9WnI2EttI3NID0KIyQq/3y5tIUNZA8whe9RB2/Vo4jcAagVvWxHg2DIkNliXQOetgyRdxGvhxU+23R4HZFPQIg/BCFlkkCgxrxsGm3lvTYGPVhsHGVs+CbT1oai/QV11eWODGPkMQp+L+Ilz319iqSjmPlqh0ofiZST9Ur/BeU+zgCK58cY8NMFxD2wouBI+FzMfQdRBor22s63Elfv0K5srHTguhy9ngmA4iC/hXZqnVp2bf3pIqT82ofsrJC5fAW6X5PpdELurlq6esD1D5UDMEfi7RxUTZMk1KxV914hoBM/ahRpTWROsum7JG4V3AgwrbaXVGOmt1Tjpv9aHGcTQOtolwNM6IdI3ziXSN42ocpnFO0I9pHGozwTQQtZxgGslhGorrR6O9c501JIPOol1znTWq00zD1KyuJ5VhQt+n2l2dQLcbFc2NpOdglfmLpY8TNkvjeGPegOgOVSglncKwJAgpOoael5MVC6aqpEqj6tFrhqe6S2R66VHX+02kyqvKJY0v4ewidzz2IZkqlSoPUzXaQe547ENyW5VQlU2Vage542HVFfxfycSue+9kVsh3TWaFfNdkVsjXktngno3qb9Vrc1AT53RrqOv3uJ1lebmkMYWiRZOn24hOK7JW5NDdOoX+VC5Oy3BahtMynP0M1jJYy2Atg+1n8JbBWwZvGZy32av/Gf6BSfsTs4E5mdT5fDb+A4sQAGw="
    },
    "cuda:memory_pattern": {
        "backend": "cuda",
        "entry_point": "kernel_memory_pattern_write",
        "entry_points": [
            "kernel_memory_pattern_write",
            "kernel_memory_pattern_verify"
        ],
        "filename": "memory_pattern.cu",
        "format": "cuda_source",
        "sha256": "5629508e5c36ca7a08ef04e43ce89ce1d1e8b1837de6dd4015d7e7d2690932ae",
        "size_bytes": 2101,
        "workload": "memory_pattern",
        "zlib_b64": "eNrNVEtv2kAQvvtXTHOobKAhYEIaCJGAuFKkUFVNcqiq1jL2Qlaxd9F6lxiq9Ld3dm2wKaGqqhy6Bz9mvm923s2aBTUIBU/Td/OFgkciGIl7kJCEi5W/CKRECdihigJHQz0mxQoWnDJpp06vIPi7eP9JUEkaB5RLIuhspY0NR9dgoD0AW6FJt+3LGkzVrAH6t9vxJTCV+E9cRGku0xDYmEp4RCrQlJDI2RjOr+mh75ylEl7JfKNiiAjBhR9yxWR+qwgfoAleJglLKWcwV4GIehDEMYzvr4YQIACjDaUSJNWM0Zc779b/5H32vRtv4n2860EH7BZkpS/aN1gQkScKuCgCQyfTVF/btCy5WpCIzEDhtXNGIog5m+cP1e309/VoHJTb7ltWKgNJQ/D9iCxpSHwfP2dchISymDL9j7htQrQzNpoEGmUNoylShCKTfPhhAR46A1trYDCAE+WAIBgyg5PspDiqv4drVXEfivMCrl3FnRbnBZxbxQ2L8wKuU+JaCi4usBPdtmNjhPAW3Nb9zY2zTzotST/tQzTkGWKzBhPN6/YgIpjHhDKa6qzfLmIqJzRz282JEokSLswoC2K6xno/khVWaroC2yTbpPdYl1ub1Klfw6By6XeDwJdt3peXgBoH6hj8ueeenZ2PzovgDW+tkWsNa3URVkPY+1NvNB52RwdgbgEbt0ftoeduUl5kYcde33q2LJKZ1XE0PsKWmsd8GsTYTUtOoz8tDXsTXg1ZOCVS0FAiTw9trsJWK2d2m4ydkd0Cqz1ZrAGUztEFnbpux5nGPHy8jrLjDGMrJVc0QUk9l8gHQYLIgPq/WdLu6YbIgXP8yZl7tnIiDhZsxgdJ6EfffF6UEeWC+qAwvXFeH0zBV1R+Q+bOPJr22F1ZJu78zud/qkW+Y+xquK9YksaBIlcX6v9TNR0LijBlxr8U9E57jYLqhbIt6pu/rGrVgD71etW3/lb3bJVPfdFOBJdmK5eGAskTGg6jyK6UoLETdaWdfgEbiYGY"
    },
    "cuda:memory_sweep": {
        "backend": "cuda",
        "entry_point": "kernel_memory_sweep_write",
        "entry_points": [
            "kernel_memory_sweep_write",
            "kernel_memory_sweep_verify"
        ],
        "filename": "memory_sweep.cu",
        "format": "cuda_source",
        "sha256": "bd3b283b14f80dce33bde80f0f3c55c8b7a13a19ddd0e14e2301ca6763b566d2",
        "size_bytes": 1600,
        "workload": "memory_sweep",
        "zlib_b64": "eNrNU1FP2zAQfs+vuPGUtAyaJrS0GUhNGyQkNk0bPEzTiELsBovUzmwHUib++85JWgKMh6E9zA9W7nzfd/fdXfZ7FvQglUKp91lRwg2VnOZTWNGVkOtY3VFagJ2WJHFMYMS1XEMhGNe2cqZteNyNju8k03T3j0+3VLLl2hDNwlOoA6cAdol0Iz/WPbgql7uwMYGXq/jnnZBEdZyKUuJsKBrCKVYouNLw10SN6Q0NhEopZJyKkuuGX6bXsA9RpSlXTHDIykSSKSR5DvOLxQwSDEAFqS4lVQYRfjuPvsafoy9xdBZ9jD6dT+EQbBeqx6R1FVBQ2agHIVsNUCRKmbz7lqXXBSV0CSXmzTglkAueNVc58oOX78gOpTcMLEvpRLMU4pjQW5bSOMbPpZApZTxn3NjIAKrImV6xauTbxqwc+GUBHmPcwxEW3IdBNYm88XgSTsYn/mw8dw8uzs6COsyE2Pdwaa7jY/AGjoPqB1V4cnDoj0cLdx750UE4eQ0wHLeAib8Y+JMwdD3Pdd0o3AIkxaZy6ORwncB6sCyK08CHnfkOCstycZXkqOlWMPL6LtobbT3E4Ki0ZKlGlNmRrezOimx99aq1rWkXDL0Z5kI5+Olc5SK9OSXVXoVyHj0LttozLaw9+lrShNRBwTMmUwmhG7IMjQb5gqsB4hzrB2CkQhDWEdSfHzrFN57+Ucu9qd4clPsdH38gtDN/E37ZKG2yPLyhzc0K20/lvaXbreUNn4G7/+b/MxCsE//KFHtT16cQOCj/yazYEuztvN69NrAuwpx+v1tNsH17sB5vw/yk5mOsuUuUaLFi6YwQu9P03Sc6O5vyG83o6gM="
    },
    "hip:bandwidth_d2d": {
        "backend": "hip",
        "entry_point": "kernel_bandwidth_d2d",
        "entry_points": [
            "kernel_bandwidth_d2d"
        ],
        "filename": "bandwidth_d2d.hip",
        "format": "hip_source",
        "sha256": "cf5563f5d0cd1f5f01749dbf5cbbf890e7feb04cbbc22a7b3818b0bd07c8f4d5",
        "size_bytes": 697,
        "workload": "bandwidth_d2d",
        "zlib_b64": "eNptUclOwzAQvecrRj2lLSVqiSKUwqFLBJUAIeCCELIS26RWUzuynS5C/XcmTlkC+BDZM2+ZeQl6HvSAamXMIC8rWHEteRFDlkq2FcwuCRsx8Jei7NbARFq9h1IJaX3TjY9w0kLXuMl0EYNfISzsATP2BKiSxsKxYjQ9cfcoJBZktSYbTkPnMNF0CQEkO8ulEUpCXqWaxZAWBVwv7oPJ7fxqdgcpwoTl1FaaG/CpSgtuKGcwHJ0PMmHBWM3TtZA5Opd7Jz19fkoeyX3yQJKb5Da5e4rhbAT+MBpke8ub2QBZDPrQLm41etUagefZfckZf4MKx8slOhYKTdynisKx53EcXUvozDpASF6oLC0IgY0S7N+4fA/wHIMhBNexWlCLlDo312xl14LUQTb8KPzOEd5/0rCTo/cl/o8o7GaFoqsF253uMJHvylyssdJvKnZZp+BA419KtTPjn2I5PhrmH62G+Ka0a4BgOyThHGN3vfgatnn3L4/Kn7PXB/d/weYrEnFPd21UD97B+wDg49ng"
    },
    "hip:compute_bf16": {
        "backend": "hip",
        "entry_point": "kernel_compute_bf16",
        "entry_points": [
            "kernel_compute_bf16"
        ],
        "filename": "compute_bf16.hip",
        "format": "hip_source",
        "sha256": "163207de91688d628113a0c85be88bad5644bcfdf32ab3ed69a5b5c4337994a8",
        "size_bytes": 3943,
        "workload": "compute_bf16",
        "zlib_b64": "eNqdV11vozoQfedXzG2lFTRtAuYjvcn2XmXbVFtp7zbqtk+7lQWJSVETYA1UWa363+/Y5AMIoE3y4Hrsc8bH4xnj9s4UOIMpj5LkYh5n8Mp4yBYDmEbLOEsZ9XzDAfUliDWBG4cp/wVxFISpmmiDNZoWwQI2+nQ3ANVfRG56BlGWUi/zzyFDlkloCmG2pGzBlixMk8JwkDLupkEU4qDkQsLYjLoly5M6Rnz6Aj0Yr1IWJsiAeeby2QCub76OSM/sWRBxeEBDdEN0+saA5nsZfXlC5ixKCfVx2Y3k+8k3Ohk/0MfPD+PRjezePY4fBnCJk0g7AwuuwDYIqJcgSCsC7nSawErM3v43Ej0CCzdkee/2y/0kEWJ7inIa+PDXjPlByGYqpXN/9bfuUqrBhw/VcUtvGDfW4z8UwF91ltSz7HpvhqHrLe5w2mjikaYJs9VhoxC7YSWi600TgqGcMs7xjE+evn57mkzuHx7HNwP4fDcpJy5nP7OA44nIxOiAmoe+o23zIx8TAcHBdaZ8ukVqksVxxNMTXCmcBb6ipL9ihlIgw4ybox7AvIXMJMPtzDrD8uzAhKbUTVMeeEIOVVW2Sukbm6YRp4KhEk3ThoqCw1hFcHJ9goT5IvLcBaXwFgWzuvJSZXjXpUUp7g1XmKbI2NSZBKCucpltRwtVtnO1qbTqiKfBbzk2RbzcLcxR1xV4i2j6ejdbdVdYGdK4CZZodCB94cydyamhpGLuq4L0z1VJ0caz+HGWZjzM4e9KYcF1RF3q4Zpqbmn+0vVVvUtsvFNy3XAhREfcV3NbOwe9a/vacN+VF7f78iq+POmLNDgLS87UC7EqBiS/+jSxljjhHW+bGi7SfuO2zsXe3od1EI/GUSJgws25FN4EDNlcAkMJDCVQIrcYrgvAVqjeNXBLeFp6V3dIUbMqzwonMiw+MOxMwwAUeZf9Pyau5e5EGAeKMI4VYbSIIAeKIMeKIC0izANFmMeKMFtEWAeKsI4VYbWIsA8UYR8rwm4R4RwowjlWhNMion+giP6xIvolEfmdj99DNf88oAp9KP5+LHwrhtDpBGnxzj6NuTtfuvg55NFisR3eOordgOeuZO8jXGbCiTCKbuT1Ly4nbM7wWuzk996wDBAXBzbNAFHU2DQDRMFh0wwQxYBNM0AkKjbNAJFE3GkBiAPGpgRoDQTe6+2B2AdUArEPqARiH1AJxD6gEoh9QCUQ+4BKIEqA9/1XwPpBIoKzzmSuf9efkbszjefhPsMoMIwyw6hnkAKDlBmknmEWGGaZYdYzrALDKjOseoZdYNhlhl3PcAoMp8xw6hn9AqNfZvQlo/j0iKIFRK/ICBJ8m+M1oSa6fKzvbKNik4ptCruUFpvfDmNVOHbFdip2X6vZWLYUO8Ps6YAQhdvCU8a+mfct0bfzviP6Gyfr9/R3vDOfhQfcb2mxbKlp8K/0P8AHofg3lroJlauq+qp/e62LX4bu3pX/AVF9HxY="
    },
    "hip:compute_fp16": {
        "backend": "hip",
        "entry_point": "kernel_compute_fp16",
        "entry_points": [
            "kernel_compute_fp16"
        ],
        "filename": "compute_fp16.hip",
        "format": "hip_source",
        "sha256": "8ec2aeb72f63e7584292d702cba523fe93f96172249d9ef7010dcf3cf5b0c39f",
        "size_bytes": 4003,
        "workload": "compute_fp16",
        "zlib_b64": "eNqdV21vozgQ/s6vGLXSCpq05Z1esr1Tr021le62Ubb7aVVZQEyKSoA1EGVV9b/f2CQEWMg14QN47OcZPzMebLg8k+AMfJZk2fkiLeCVsphGI/CTZVrklASpZoP8EqYKx03inP2CNAnjXM6U0QZN6mAOu/n7YQRyECVufgZJkROvCIZQIMvQSQ5xsSQ0oksa51mtO8wpc/MwibFTcCGjdE7chuUJHTfMf4FLmKxzGmfIgEXhsvkIFsH6D1Ud4NDt3dcbfMz4Q47R7YoCued+MJ7U9V/pHFYkfSXB0iWBZgu3j9NvZDqZkacvs8nNnWg+PE1mI7jCQeSdgQnXYGk6yFfw4kaBDq7vZ7Dmg/f/3vCWDpEb07J1/8/jNOOeLyXpNAxgToMwpnOZEBR6pRqEKNIpZSxhcPL967fv0+nj7GlyN4IvD9PmCjD6swgZut1GGCBnE9YmmvtpKeIEXcbzMJCk/FdKcUooMEcLnBcw01AY+rgaqTLCJ1nzRSDEzXMWenxmIst0nZMV9fOEEc6RdUVRxpKU5Ti1j+A5XYU+IrGJinwaxhFGiHblUORXGHLVx4a7cbfW9hR4kwAvRvOCxdXAG8EKCqM8jPly4WrJ7If6PARX3D28K0PowGgCowmM9qy8j6V3ScKQsGrh5PYEKYso8dwI9a6ScN5VzrLQsyllQnAJMDt+joxtXQsAZrVZ1lVvrap3rraV3e6pEuAjXqwVLFDXNXhR4r8+zNcXayxCYdyFSzQGkL8w6s7F0FhQsc5kTvrzuqFo63mX3hKOGdlNWNWDS15wVnlrKzylsnqhW/gel9rhnAtPWCCXNq6AemEFyrjLnZf+nz+v5c8T/vReh3HLoXzOZ8fklNuOwmfklbrj7ioOiW8Y4JBH+T7ugngkTTIO426GQn4fMKYLAYwFMBbAclmralc5oCZVvdAwMFw79UK19bpqWawcDhQKfALNKhRMQ5N55XyYupG8E6IdLEQ7Xoi2R4h+sBD9eCH6HiHGwUKM44UYe4SYBwsxjxdi7hFiHSzEOl6ItUeIfbAQ+3gh9h4hzsFCnOOFOA0hpRI86uXyKEEl6pg/P9fOlTEMBmFe399PU+Yuli4e/CyJoqq7cpS6IStdidZnuCq4E27U3Yijgm9etTOcqUN+ZIsNcrM1V1CtBdX6oXoLqvdDjRbU6IeaLajZD7VaUKsfaregdj/UaUGdBvRjicUD5aOJ7YD2JbYD2pfYDmhfYjugfYntgPYltgPal9gm9P33L5rNxxVP9uZNYyp+L+KbtjO15/HvDK3G0JoMrZuh1xh6k6F3M4waw2gyjG6GWWOYTYbZzbBqDKvJsLoZdo1hNxl2N8OpMZwmwxGMGsVLkgiSV2SEGf4N4TYmZyrueZ9qttay9ZZtcLtRINtrhzFbHKtl2y3bUToCK5Y8MqyeAXBRGBauMraNsm3ytlW2bd7eOtn8G/zAPf2Ze8B4G5MVS0WBv4T/Ef5V8F9g4mZEzCqra+f+VuVXofA/lv8AkRdK7w=="
    },
    "hip:compute_fp32": {
        "backend": "hip",
        "entry_point": "kernel_compute_fp32",
        "entry_points": [
            "kernel_compute_fp32"
        ],
        "filename": "compute_fp32.hip",
        "format": "hip_source",
        "sha256": "f4e459291b34b5724f27d92f7aa1f523e116192209a221e150692f38ef50f261",
        "size_bytes": 2359,
        "workload": "compute_fp32",
        "zlib_b64": "eNqNVl1vmzAUfedXXHVSBU2b8hFIlbSbsjZdI61r1PVtmiwgJkUFjIypUk3977s2CSWBqOEhudc+5/j6xNfk/ESDEwg5K4qzZV7CC+UZTUYQsjQvBSVR7tigP8e5IXHTTPA3yFmcCb0wRms0aYIlbPJ9NgI9SpgvToCVggRldAolshybCMjKlNCEpjQTRWM4FpT7ImYZDiouFJQuiL+VBaqOCQ+f4RymK0GzAhmwLH2+GIGfJHA3m59P7m9+XP8CH2GoGoqS0wL0DNVfKdzOcUu39xOl9DD/TebTR/J09zid3Khw9jR9HMEFTloefthwBbbrgX4BfhiWaZn4gvECVghBFcj9WGVKcx38fJgXUv5c08RbThc0ghILXWZ0AbhdKB17rGkUy+cZHF0fASHLhAV+Qgi8snjRZayuAT5rUwnBHQkehwIZG4cVAKW3Da5HG/5+SG083h0JDPinxkLEq4JhiXVdQZCw8GW2WPVX6I1KbuIUkx6IZ079hZoaK2ocgS5JX6+2Ktooy4dT/G2yCv6uNRasSvFxwSj1I93s2y4eoapYOJPTjEd6lRunYPbdyBi3+AHJWdGlEexoBErD3iOS0SWKnMlF5Kal6FhrOMZNnDb7FvLRB7NverZCVi1g6MoFnCgNOAbLLdeLrMnWQWSrm2wfRLa7yc5BZKebPDiIPOgmuweR3W6ydxDZ6yYPDyIPG+SKzTjoVRdJgbH8vmy01Bh6vVg0j/aXnPvL1MfG5yxJ6uFaSN4blZSKLuGilCIyacqoLjE3Z5ibp4AXojqC633VIKsGWftBdg2y94OcGuTsBw1q0GA/yK1B7n6QV4O8/aBhDRpugT4zCnv3c6M6QG2jOkBtozpAbaM6QG2jOkBtozpAbaO2Qe/tyzZgLAH2gsS4iOIMTzX6hw1w3MitndzeyR2Zb1WyeT4wgx2Ou5N7O/mw4zYuyhTrxPqwS2VRPYxtGTtVPJCxW8WejDci67fkH2zwv1IB99tcDHUNA74p/RG+X+XfEuIXRK2qm6vh7bUpH3kjvGv/AUbrdDw="
    },
    "hip:compute_fp4": {
        "backend": "hip",
        "entry_point": "kernel_compute_fp4",
        "entry_points": [
            "kernel_compute_fp4"
        ],
        "filename": "compute_fp4.hip",
        "format": "hip_source",
        "sha256": "8e62efc4c05519fa5941ddd134e620a496ee6a804f26fb2b7b63c4af0ec45073",
        "size_bytes": 3169,
        "workload": "compute_fp4",
        "zlib_b64": "eNrFVm1P4zgQ/p5fMQfSKqGlzRttt2W14qBokY6lYtlPK2Q5qVMi8naOw2W14r/f2HHbBHUFi5Bw09jjPPPM2DN+GR4YcAAhz8vycFVUcM94xpIphHlaVIKRqPDBvIsLS8LmmeA/ocjjTJilNdVg0sJK1MnfF1MwoySn4gDySpCgivpQoZLnEgFZlRKWsJRlomx1x4JxKuI8w06lCyVjS0I7UqDcOOHhHQxhXguWlagBq4ry5RROz76eoLerqP54ZPcsuMPe/yhncL7w4fL88gRMgs7EiYgzQtPlKsxIGqWUlCFN0H30wxnV8nEnJJpEo8hX9q4W38hifk1uvlzPT85U8+Jmfj0FHz9O8G+6+HJGm5c7QfdGvgWfwHPHowmYnrtxooQaHNv14fyfq0U5TGjGpJWhYezHEfy1ZFGcsSV62oyDEMvYZ5znHPa+f/32fbG4ur6Zn03hy8WiEyXO/q1izsoXTEOclYJXoZrtPWTPlnFkGOJnwdA6VDipK3QBYQIqz51tvjSRiArPrX0MGSFUCB4H0gNimqwW5IGFIudEKpi+ZVlbXUkWo+LkecWJUjSwG/ML9k73UGGV5AFNCIGHPF7uSDzTACw66QjBeUADoUCFdQYqAA6nm4Cb3lb+banWOfi0J7Dgl+oLEa8mCVbo1icIkjy8v1jWgxrzQAlncYpCD8QdZ3SpPs1aqg1pQEuG2vbgKEKoPXCxxrRqrMOhROU80rJlQU8x7Chd1eCJaiBVEWOPNEpZt0zp+wdwjio57ZJmeAALGt5jDjA3dVTiZHEQJKycgl276GrPGdh9bDuyjX7L9kdsH2J7BvPJpQ1qVeFiGGM3ogcyx7cDX+cCJQ80QcQv5NVFcr2gvXkeZzt4A1LkZcPr6KL8fb79HG/GVpLXxIy2cNC6VH14Xc9v7MmcqrJYNJuTTI56fN78Kh2lzUIMbemQTKJ+k0oyyA7GuCV2JW98FGlzWxZnzaIgHfyoqz7uSJNdZG6LDKO/xTtdz5xBV9rpmdch6+BHXfVxR9KeNWy4g5rNSpezOZP1cWvZz6DXi8V6YcuyX3C6SimGgedJsuneEBU05g2Vah2DX0kSKbRpVExlhP708DHV2ug3qdxHjj746sHGNjWeiNasa9h5A8POawy7b2DYfY1h7w0Mey80/JZBxn3lfYLcGH6HIDeG3yHIjWHvDw0/Gs1bVXhXU+fmcec+0V737QO+rFJ02AztH/Yt7kxYO7p2de3dyrPZDB2NcDTC0QinQfzu4Mdihq7WdbWuq3XdDbunEZ5GeBrhScR2pPrO9AMHeIt+xyXeSXGnNHEYFnxWo5ni9MvLO6ElUWM05QF1astSaapH49H4H8o0hM0="
    },
    "hip:compute_fp64": {
        "backend": "hip",
        "entry_point": "kernel_compute_fp64",
        "entry_points": [
            "kernel_compute_fp64"
        ],
        "filename": "compute_fp64.hip",
        "format": "hip_source",
        "sha256": "56bc202f8398ae7eae074a6f4446dc21fa22892b6418afcd07e809417a4587a6",
        "size_bytes": 2359,
        "workload": "compute_fp64",
        "zlib_b64": "eNqNVe9v2jAQ/Z6/4tRJVVIohEACgnYTW2FFalfUdZ+mycoPh0YNceTEFdXU/31nk7BAiEok4M5+7/n88DndCw0uwOcsyy5XqYAXyhMaj8Fn61TklISpMwD9OUoNiZslOX+DlEVJrmfGuECTKljCpl8XY9ADJryYXgATOfFE2AaBtL5FckjEmtCYrmmSZ5XhKKfczSOW4OCWDBmlAXH3U0+VMuX+M3RhtslpkiEHVsLlwRjcOIbbxbI7vb/5/u0HuAhDXT8XnGagJ6j/SmG+xF3N76dK6WH5kyxnj+Tp9nE2vVHh4mn2OIYRTvYc/LLgGizbAX0Eru+LtYjdnPEMNghBFUjdSGWWzIrg7mGZSfmupuVvKQ1oCAILXSU0ANwwiL410TSK5fMEzr6dASGrmHluTAi8sig45q2uAT6lr4TglnIe+TlSSpMVArX3Pd6NViyuaJU214Y8A/6qQR8ZqmZYYWnX4MXMf1kEm84G7VHJTbTGpAX5M6duoKYmihqFoEvS5+u9mkpl+XCKf0+yhb9rlQWLWlxcMVy7utmx7HZRLFxCGDPG9W1qtMHs2MakTvZIyrK6gLcv4CkBq0EhoStUuMQV5Hal4ESrmsVNnDY7PctGB8yO6VgSWHSAoav944Qw4Bx6tigWKcm9U8i9BrJ1CtlqIPdPIfcbyINTyIMGsn0K2W4gO6eQnQby8BTysEJW7JBx0LcdJPkT+XtVaacJtFpRXj3Un1LurtYudj1ncbwb3gnJS2MrpaIrGAkpIpOqjOoPszi/3GwD3obqBBbb2mF6JabXjLFKjNWM6ZeYfjNmUGIGzRi7xNjNGKfEOM2YYYkZ7mE+MAg79kODjmBqBh3B1Aw6gqkZdARTM+gIpmbQEUzNoH3Me/1a9RiLgb0gL8rCKMFTjMbhgT+v5L2D3DrI+zLfK6R8/mMGBxz7IHcO8uGx2zcTaywUC8S2lFW1MLZk3N/GAxnb29iRcalSvBJ/Y0f/kQq44epqqGsY8EXpj/FlGrNkJT/Ezch2ad3cDOfzkVl5ft3dofq79g+6dXJE"
    },
    "hip:compute_fp8": {
        "backend": "hip",
        "entry_point": "kernel_compute_fp8",
        "entry_points": [
            "kernel_compute_fp8"
        ],
        "filename": "compute_fp8.hip",
        "format": "hip_source",
        "sha256": "090bec16288e8dba9791da4f73a36699ffc01d71c6d51d723203760792d2c234",
        "size_bytes": 4326,
        "workload": "compute_fp8",
        "zlib_b64": "eNrdV21v4kYQ/s6vmCZSZCcE/AKGQKoql4AuUl5QLlGlXqOVMWtinbGpXxKfcvz3zr4AXmIu7SXXD7Ujhhk/8+x4dmeGNPdrsA9eEqfp4XSewxeaRDTsgRfP5nlGiT/vgvYQzHUGG0RZ8hXmcRBlWqr3JJiUsAx18uG8B5ofxm62D3GekXHu1yFHJ9siGUT5jNCQzmiUpSVzkNHEzYI4QiP3hZTSCXEVbczDOEm8B2jCoMholKIHTHM3mfTg9OzqxAZt6hdHraLJRNvQ4XJ4eQLDURfiBG4Q0eII0yoKHX6/FM8Y6/XoExkNbsjtx5vByRn/en47uOlB1zyyAFeR9LYFBVhtB4YX16O0GboR1fvQMo4cBpIrmM5LEC7SrNV2Ax9+mVA/iOhEI4QHaxCiw97eC7u5xW5J+581wGvjabuazbSMrQ/YOrVdmiSYop27q093o9H1ze3grAcfz0fKUUjoX3mQ0FTNtV6V2wfckyc3oTz3QZRmSe7x/d3BpaJJ4Ndq2dc5xVggx22cYkAIyyC3rf7qSRhHU/EROK21WRwJf25bRQvPDiFuliXBmEVJNI0WGXmkXhYnhDloLV3XK327r/t2FV8WX4CO1uuOFnesoRmLBHZOd9BhGsZjNyQEHuNgUlE9Gt9PWTmEYJ5xAS9Dh2UZcQBmSK2ilbVURGuqZSFtWsY6PHObh3ied5hiWL/COIy9L+eTolFgWXDlLJihcgDZQ0LdCX/UL7kK0rGbUvQ2Gm0foUbDQrkPmlgdDhkqTnyp6zoccIaKS3Udb7iOmStiDEei+Oq6xmLfA7Ods7SzKqsqsm/foKrGXpqtSjOvLR52c1+WADvd2qB1aQ+v7v7AnmgULQOzcGA2jDoqdpcpmBOmfGDKISoN1grW+cOzDS55dEOWPvRX74uL/gZ2TOZxyrF2V72rsBGdIlZDRWchqPcd8xAnY1lNHov/me1mXewpy7aJyS6pqmZ32v6iv8FiLlk4RME7qntH0bpVZFaJDDO7xptqZGZD1SojsxUyBe+o7h1Fk5EJNux4mig5thF9Jo9L9deHg4MgW1YYu3bniTududjskjgMV+YV0dwNEkHFvx1DK2ckTCnT8N1lO0SwIQRhFkTEnU2mXkRm/swlPg5T0ynwD79gS+FthZ+tujg2dfTGsyj+9L7Ka76J19zKa72J19rKa7+J1y7zvk+CsdZ+SoIF7/snWPC+f4IFr13Fu6iJTy6wTfPGfawMtPJ5L0+YNJ+xTuYZn417rEiUppSWlPY9Gw6aZ0qEKRGmRJgCsW3y4KV5lvS1pK8lfa0Vuy0RtkTYEmEzxPpN5dD+jC94j3EHKY4S7BAavoYOv/G36WF22U9g4qaEv6NmFJ3hqcGuXFIt8PdSmNLl2Nn8ncUGkJg/vdW8WQ4fozx8jC3DR/6eWQ6g5/VUqZcmzKJf4bMcRMzHELdYV9xbfMRAetbwvdlEMsSd12HTsiiPpu4Pj6Z61Xa/w2Tqbo65fzuZvhfY/2gwPc2UfmE6y35BntDID/KLEfWPGucPEJubxG+PWPb8949YNv3/rncK2ZKyLaUjZefVzvm9ritkS8q2lI6UnZ/UN/n/mYva30HksVQ="
    },
    "hip:compute_int16": {
        "backend": "hip",
        "entry_point": "kernel_compute_int16",
        "entry_points": [
            "kernel_compute_int16"
        ],
        "filename": "compute_int16.hip",
        "format": "hip_source",
        "sha256": "6682959decad7732a1f9f4f77638317836d4167bdf8af9debc5b61176bf28e8d",
        "size_bytes": 1883,
        "workload": "compute_int16",
        "zlib_b64": "eNqVVF1v2jAUfc+vuOqkKYF1QAKBkTKJlWzloRtifZvWKB8GooY4cuyKauO/79oEEghCIy+5Pj7n2L732q2GBg0IGc3z22Um4IWwlCRDCOk6E5x4cco7NuirODMk0U05e4OMIqznxrCge0dsyRt/mQ5BFzi2TI83gAruBWLxAfYQpGLtkYSsScrzChxzwnwe07QK5oREnn8KBGpHYxauoAXuhpM0Rx0shc+iIfhJAg/TWWv8OPl2/x18pKF3yAUjOegprvFKoGPfBjEHgcplSiJ4HE+U6Y/ZT2/mzr2nh7k7nqhw+uTOhzDAScxHA0wYgdnDzAzAD0OxFonPKcthI6dFymiS7PwkZALNcmnc0jT+lpGILMo18UwgLNOpz+QrynCuYzuaRvB8LIWb+xvwvGVCAz/xPHilcXS2BroG+KFtA+l4Ys7ikKNgX4f99HEZDmilCgesKMLxODDgj0JC5KpzwBK3NIIgoeHLNNp83GCy1GASr3HQBL5ixI/UlKOk8QJ0Kfo8OtrN3ll+jGDd0h19q1UXxGxj7nFBHUND13e7hPfQ3nzFTxjwFzrCcE40fhSdaIIzmt1Zkc7aJVvttYkEPFh788m1+ih4LtJjVFycUt6pyc1r5GZNbl0jt2ry7jXybk3eu0beq8nta+R2Td6/Rt6vyQf/IVf6BWUoVbcBTdqO/N9VroYDzWbMq236LmP+cu0X9/8AH4xyTrKdlYru8K0Q0kWOqj6q5cuWw7Churwp+7Y43YFX9haGF3hlE2F4gVd2C4YXeGVbYHiBV9Yfwwu8stDMvsQrK4rhed72/EOBD3XZDfvMPmMGDXQ4pAYBqwJ0JdCrALYE+kaxXPGg/sLe+q3MLdPAdRxtq/0DI5H0Zg=="
    },
    "hip:compute_int32": {
        "backend": "hip",
        "entry_point": "kernel_compute_int32",
        "entry_points": [
            "kernel_compute_int32"
        ],
        "filename": "compute_int32.hip",
        "format": "hip_source",
        "sha256": "e7fed020e4bbc6eac776723d8f07dd9487c9234a94d7e572122321f534265828",
        "size_bytes": 1584,
        "workload": "compute_int32",
        "zlib_b64": "eNqNlEtvm0AQgO/8ilF6AbupHbDxg7iSE6PEh7SWm1vVIB5rjMJLy27kqPV/7+yCDbbiCA7D7Ow3s7MzA72OAh3waVYU12HO4ZXQlMRT8LMk54w4UcoMHdRtlGsCtFNG3yHP0KwW2rTCnRNacPO75RRULtcO60DGmePxzVc4mCDliUNikpCUFQ1zxAh1WZSlTWNBSOC45wZPZjSn/hZ6YO8YSQv0g5C7NJiCG8fwuFz15k+Lh/sf4CKGsX3GKSlATfGMNwKGfu1FDDh6hikJ4Gm+kEF/rn45K3vtPD+u7flCqstnez2FMW7emCh0mIE+NEEdg+v7POGxyzJawE5s85RmcVzGEyYdsrwQgXuKwt5zEpBNfSbeCbihW4pC8A40hav7K3CcMM48N3YceMui4MM6qwrgg64dxPFWjEY+Q4dDrQ/bp6U+WhuVPtqqQp+uPQ3+SouPrMwVQkxpBl6c+a/LYPdthwWRi0WU4KILbEuJG8gtS7pGG1CF0/fZSTaHyOKhBHuTlvheOTsQ64sHlunBP7jh1hngBsEB8CrgeAvaxy15fBc3MNf+bmIbo9HkbsI1eKnCWjV/U/N6G16veaMNb9T8oA0/qPlhG35Y82Yb3qz5URt+VPPjT3jpsMko/gjkvKFX3xLv28bwWdDtRqw5CF9y6oaJW31FR/MxUMFIXoaS2i1+cVxEEatmHDlUovUoOnKCumJMrFNA9BrFZUA0F8VlQHQTxWVAtA/FZUD0C8VlQDSImp8AoiMoPgT2zU+q+jn8xu79EV3E2rzg/TV0UvGWqBulPhD6sNRNoY80S9kr/wGVXqtf"
    },
    "hip:compute_int4": {
        "backend": "hip",
        "entry_point": "kernel_compute_int4",
        "entry_points": [
            "kernel_compute_int4"
        ],
        "filename": "compute_int4.hip",
        "format": "hip_source",
        "sha256": "35d52d08167fc293715247321decbccd0f2d44b82bf940a7f7b9e52e41e0832f",
        "size_bytes": 2546,
        "workload": "compute_int4",
        "zlib_b64": "eNqVld9zmzgQx9/5K7bJTAacpOaXjWM3nUkbzzQvrcdNn9qGwSA7moDghNSjd5f//VZACITgqfVgoe+uPlqtVvJ4pMEIQp7m+fkuk/BAOCPxHMI0yaQgPmXCBf2eZobyWzLBf0OWoqrnxrz29tvOyu3qw80cdIlDx/bFCFIp/I3cnsGTBEwmPolJQpjIWzIVhAeCpqwt5oREfvBS2JQBXfHwHsawLARhOc6DnQx4NIfdtrgwp+Oym1VdoDrLdArQfYyGxoIyP0iiXcj8PErFrAR+WX31V8u1f/tpvby6Lj9vbpfrOczQaE2rn0uwTXcG+gyCMJSJjAOR8hwKZZOMp3FMIiiZlZZmuYKPNe2YbuFNRLaUkQjDqML0fQNOTnr6bEAPav2HBti6Vtyf+eo0NFhDBns/0B2aNxkyqB1px4TzlMPRt89fv61WX9a3y+s5fLpZdUuLk78k5SSHezy3vwNO6rzlMstSLkCvT/Lj9ecr69wer7G3jSOks4huNU38zgguj2nP6Q5jAKSCdOxFY1ECVYJGsEo4g6OPR4CRxukmiH0ffqU0eq2Q9TIbiBqhN0YoOA0F+j8V85O5W8uN2irlRqsruTveGPBvqYToW8YOO4zoEjZxGj7cRMXbAquuHFzTBAenIO45CaLStCinYlXpatL7y040T2TVOBGSs8r9USu78QhWQfiASZsVUKfPPd9QvJ50s4nxUCiD79YZeD/hFw0ALUmQP1C2w3yRrD4plrLzfwhP36oCf94HphzwZvgZroCb0XFs6HqVATgBs/DqJg34D4dW3aSxeJGNhBYIqGfe1UlbVFtQq3DzmV9m4RQsiSkzi4ul43kXHy5wiTuFMV4svHhmWD2GfTDD7jGcgxlOj+EezHB7jMnBjEmPMT2YMe0xvIMZXo8x+1NGCdniE6RX9xFJ5kL171qXcwGnp1S0L8pxxoNdEtQPeSM3oFxg6Zeo8usdPvBSUdSozSkvnSrN1/9tdG6eNRfkDNRoG8Q5qTffEKw9BKtDsAYI9h6C3SHYAwRnD8HpEJwBgruH4HYI7gBhsocw6RAmA4TpHsK0Q5gOELw9BK9D8PqEx/bTW/+JfMeS/qkqXFWWgdVyB9WXZWCl16rdqA6qnYjqVju6jeOkNX3aqB5G86j9D11Ay50="
    },
    "hip:compute_int64": {
        "backend": "hip",
        "entry_point": "kernel_compute_int64",
        "entry_points": [
            "kernel_compute_int64"
        ],
        "filename": "compute_int64.hip",
        "format": "hip_source",
        "sha256": "8ec37c8b9ab8ece1783ada49881f1587fd0844742c22872d40a1549eb1b2c67c",
        "size_bytes": 1755,
        "workload": "compute_int64",
        "zlib_b64": "eNqVVFFvmzAQfudXnLoXSNalAQJtaCelTbZGareq656mFRlwKCoxyNhVqi3/fWeTpE7XTAsPx/n8fZ/vuDO9jgUdSHnVNId5LeGRckbLIaTVvJaCxgUTgQ/2Q1E7Cjhhgj9DXWHYbpzhCh5voRVudD4dgi31OhYdqKSIEzl7DyrkubEAJucxLemcMtEY4UJQTkRRsVVQ0aGhNIvJ60CiMxrx9AF6MFkIyhrkQS4Jz4ZAyhIupze90fX488UXIAhD7VRIThuwGZ7xRCHwD5NCgERmzmgG16OxFv168y2+mdzGd5e3k9FYu9O7ye0QjnGzH6Bx4QzcQQD2MZA0lXNZElHxBhZqWzJelWWrp0IuVHWjhHuWJZ5rmtHZy5llxfLWyMCP/t7HmkF6bmRZFGvkDA4uDiCO87JKSBnH8FQV2Zt9sC3AB0U7CMeqBS9SgYR1L9ptz91uxSZqdGIttG7E9jpx4JeOpIjVuUKOKZ1BUlbp4zRbfFjgB9OLcTHHRRfEA6ck01uRphYzsBXp49lWNmtl9XCKvWMtfGmZB2Ii+P3xwDY9+A3971dX0SsIybI1JNkBSUhDEWOj72A2kbUplB+puN7vaipWdLQ4mXhheHJ+En7yR+FFf4BxB+5XaUQv5L5BdvcluwbZ25fsGWR/X7JvkAf7kgcGOdiXHBjkcF9yaJCP/5Os2bOKY+/16KPEUaTep8Y9iKDbLYQ5k+9qTvI5WV34TXgj1Ahat1LaO8Wfg1QqamXq6PlWI4amo4e5q+Y12gaoMUKzG6BGBc1ugBoHNLsBquVodgNUW9HsBqjW8eAfANUeNG8ClubtXv2nfuBN/Klait/mHut3kGRjleh7re8rf9D6gfJDJ7KW1h8s9dnU"
    },
    "hip:compute_int8": {
        "backend": "hip",
        "entry_point": "kernel_compute_int8",
        "entry_points": [
            "kernel_compute_int8"
        ],
        "filename": "compute_int8.hip",
        "format": "hip_source",
        "sha256": "41b8a4ee4c59d0ea48f4a814380eb2b9b5ed643d182a52eab7e162460c6c68e5",
        "size_bytes": 2601,
        "workload": "compute_int8",
        "zlib_b64": "eNqNlW1v2jAQx9/nU9yoVAXo1jwSHtpJtDCVNy2i7attjUJiIGpIMsfeqLZ+950TQAnFEUQi9t39fz4n58tlS4EW+DTJss/LlMMroTGJ+uAn65Qz4oYx64K6CtOmiBvHjL5BmqBVzZr9bbRbDhZhw5tJH1SOU9NwWQsSztw5X1zAzgQxX7skImsSs6xkDhmhHguTuGzMCAlc79AwzxMaUn8FlzDeMBJnqIMl92jQh+Vi09M6bXTdju6HeJuJm+piGmHEwtj11sHSj10eJMzKSQ/TR3c6nrlPd7PxcJQPJ0/jWR+66NQ7+NeFa9A1wwK1C57v8zWPPJbQDDbCz2OaRBEJIEeirQtJmgn0paKchQv4FJBFGJMAkyiyc90mnJ9/sHcldm9r/6EA/g68lnZcZekSu3Hcbms1q+iafhyHDkPiMOuBpgxoSoFWPdCW6Y4/b13XajPErcl0hsxh1gJtTaKzJSsZmiZzCIVyRihNKDSe7x+fp9OH2dN41Ie7ybR6hin5xUNKstLREGeiDSs8MH88SiCv24ynaUJZA6lxEC4Uhb2lBJfF+s7CJa4NSANuGgNFIXjqaAyN2wZgQlEy9yLXhd9JGBxrDGr+NFDZwmhMhNHQZxi/aw47d7U37K2l1rC3bTtDdT5vwt/c4mNsniosMaNrmEeJ/zoJNl82eJjzyShc46QNbEWJF+SuQS7F46oK0dfrSjY7svhRwjiNi/B35WBBbA1u6vmvuKpaJAnnoG2cb8XFm/APp5peXHxwKA83qNwKX7bbGij7fVJNgPMM26Bz3I626Y1Nx+nd9JD9IgBiiR051+gljXGixihpzBM1ZkljnaixShr7RI1d0nRO1HRKGudEjVPSdOs1uWiBZ1EtKhaV2kDcr0rlO4B2O2TlUjpLqbdce9tPyN68B2WMpAUqH13h54YLipiVOXlZitI4/plTqXaxr8wLELOFF2WkOagS9BqCXiHoEoJRQzAqBENCMGsIZoVgSghWDcGqECwJwa4h2BWCLSF0agidCqEjITg1BKdCcD4S3svNadtmv2Mh/xQVjYXygm+wiVWNL0WMzWJsibFdjDti7CDyXfkPC73nMA=="
    },
    "hip:compute_tensor_bf16": {
        "backend": "hip",
        "entry_point": "kernel_compute_tensor_bf16",
        "entry_points": [
            "kernel_compute_tensor_bf16"
        ],
        "filename": "compute_tensor_bf16.hip",
        "format": "hip_source",
        "sha256": "44a35081841695bcd1a2382668d32df705b20d518fca7e1a99050447219b2313",
        "size_bytes": 8315,
        "workload": "compute_tensor_bf16",
        "zlib_b64": "eNrtWVtP20oQfs+vmAMSsklIvHYubsLREZeg8tIiSp960Mp21sEisVNfaCrKfz+z683FiZ1AY/rAwUKd7Hi+b8e7s/PFaeOwAofghEEUHQ0nCdyz0GejLjjBeJLEjMbMj4KQ2i5pg3LnTVQe3vfj8CdMAs+PlUjtShDNwfDok9PLLijuKLDiQwiSmNqJW4MEwYZOY/CTMWUjNmZ+HC25vZiFVuwFPjoFFiLGBtTKjGyRzkno3EED+lM+MSJgmFjhoAtn559OQBm7Y4u6yEnaU/HH86LkHhEm/6hCEMI1hhqNJig/xivR4jHENJ+vvtCr/jW9+XjdPzkXHy9v+tddaGof2oDT8vn0akNy9UDXmubMT0AZutMPmsmpGpXKvufCXwPmej4bKJSm9yhV4eBgzW/l+5ua9P9bAbxW75IClJ7vb21iI0TTcmF4gxTd0DcTGgW4VtFMLbKJUC/KUBcZVvZZGOJG73399OXr1dXn65v+eRc+Xl7lVnrIvideyCJZQunuVBeVUhU+viroPL1AxNiKQ2+KZIjaw8n8gedWKvHPCcNsIMG6HGJKgNUNiaH35nfSYnYnhj5tYtVTasVIZPOEqKKwaUwfmBNjXhygNFVVzcWa27FmBhvdBSEeMixxfTtUL4C+OOMFdIeE8UxuxeKR5eAK+rE3wd7ZHiKGo8C2RpTCQ+ANNjQtRdSXbFiU4o7iTE6MwFn3EgG4j9nmNfcu9a4F1ax/rXpsFR6Fz8F4UR0wxPT+BnsUOPeXg2l9is1HDM69MQ6qEN+FzBqIW70laEpqWxFDtFZvuRiq1XW0h6Cks8MRjwpCV45VFaqCIefKQu0VqM2hGKO1ZZSYXVV47gdAWglffk7TQL2YWk6cnhPbi4H5TjDw/GHUBW1qXJgaplslda0mhpoYYvp8eJoOj3BY532Tt8287vjrF+Q1x3U3ke60h+S1xjWMaEhLyzyrfIs+WCNM7jF9iBrk26deDtamkyCaYbX5k6/aAqzPhhyriDOhpqtUgxcMn+TezPuOw1f5kZdOLS0gvrUEd3ZpmB0ZnZYr01uwkBmLCMnEt7PwTmZk5pHpS2RYHYt4ks2M1LOj3MyMDFkmvp2FdzIjmVnKhu1fSc83P2A9bo+XDnsPqlUvnh1nfu1PQms4trD/h8FoNHfPiSaWF6ZU4tMxNBNOwgfLNKIG+A5R7D7eKPZ8ao0HQ8enRV9uFFGctbTOagjGikr/1F6WluxCSwpp9V1o9UJaYxdaY5m2lMXFg/gai5vSlr64KW3pi5vSGnm0T5X0X2GwdQt1OM6o5nKdL8tYlIwxHcXRvmm3eBLREml1aY1brkCKQ2QEkRFERpA0okje8FIcXWJ1idUlVp+zGzLCkBGGjDB4xOJJ5TeDb/iAt5i3F6GCYGdQ8DFU+Ec8TRcXl7/eUCui4hkVbdq5ONP4lUiqJ/zqOFrXOHNdgfRiBVpTDb1YcQpin6Ew7xryNjVEvBSXqx+5lLu1t1zK3VpbLuWOmrHCWYZe5FKWuphl6EQu5btGvJpGzH7/WH17mf38kW3y5vNfX7bZNQExn/96s80WcO/8+lPbsN9Q0luV+duKmJtdCYJorqrrSwVxU2JvSA/zfwKmP9AjDtSaNj6ri7+Ulayy7pirlJ6Sc5Xq8+faeWqb0rakbUvb2drMNwlBapvStqRtS9t5rVYesZUuJ37gLKk9b2x1UHLrF4mX1Pu3J16irsjEX1dY4P/D+66Eb1wJy9bAstWvbN17V7wSFU/83+hT5T+Ihdhk"
    },
    "hip:compute_tensor_fp16": {
        "backend": "hip",
        "entry_point": "kernel_compute_tensor_fp16",
        "entry_points": [
            "kernel_compute_tensor_fp16"
        ],
        "filename": "compute_tensor_fp16.hip",
        "format": "hip_source",
        "sha256": "5cef49c1b29370cfb03e5ddaaed79b1f437d1f73cb2359ec515464f12bc5c763",
        "size_bytes": 6457,
        "workload": "compute_tensor_fp16",
        "zlib_b64": "eNrtWFtv2kgUfudXnE2kyAYCHptbIdsVTYgaadugNNU+dKORMWNixdjUl8RVm/++Z8YDsYmBXLz7sIoV5XiOz/fNmdv5bJrVClTBCvwwPJwtYrhhgcfcPlj+fBFHjEbMC/2A2gvSAeXaWag8fORFwQ9Y+I4XKaHalyBagOHRww9nfVBs1zejKvhxRCexXYcYwYZOI/DiOWUumzMvCjNuJ2KBGTm+h06BhZCxKTVzrYlIZxhY19CEUcI7RgTMYjOY9uH45PMQlLk9N6mNnKSTiD+bdFTwA7jAx0azBcrdfC2C8hDOfD7+QsejC3r58WI0PBG3Z5ejiz60tHc4H7yDPhg6JED0Hpz+eT4Om67pscGK/M68ZYbeB5y+BPR2JxPEe2hWKvuODb9Nme14bKpQOrOTd1qPUhUODh75zWJ/S5P+vyuA1/pTsgGlF/vb29gI0bRCGD4gmx7o2wmNDbj2pp7aZBuhvilDXWRY2WdBgMu/9/Xzl6/j8fnF5eikDx/PxoV7PmDfYydgodxM6erUHvZPTfj4rKBzbkaBkyAPAvawH2/q2JVK9GPBMBGIcXPOMBvALQ6xoQ9WT9IdbS8MPWnh1qfUjJBownOhisKSiN4yK8KUOEBpqapaiO3txvZyWHrKwThKPtTnd52Hv7J3PHU78XgqOUEF/VhxYO94DxEz15+YLqVw6zvTLaVIEXtFliFKcYmwJytC4LImiQBcmHxJWnkzFemBalmV1j0TFX4Kn4XxYrlhhun9DhPXt27OpkkjwfoiGifOHBs1iK4DZk7Fo0EGmpJOzJAhWmu0bQzVGjraKihp73DIo/zAlm1VhZpgKLjy0MkadMKhGKN1ZJToXVV47gdA2jGffl6xigrWr19QVK8eudNyhe70+BZVq8cYvdAtSkR2spYb2aS3posT9pM0NKzmdSi294Mi8IQu/JCD+XTz4GK7CeyxGQcfLqM33iBBum+WJ9/SOJCvdT1dcb4WBJci08y3jG7blnk8sJAliwjJxXfy8G6u1Ssi0zNkfNJW8SSfGWnkW4WZGTmyXHwnD+/mWjKzlA1rr5IeSH4iBtweZU7nAGo1J1qeP37tLwJzNjexAge+667cK6KF6QQplbg7glbMSXgjSyMWm68QxXLhuJHjUXM+nVkeLXrHUMQWrKebqY5A3DbpnzrIU5KXUpKNlPpLKfWNlMZLKY0s5asnEw9X2ZOZUpY6mSllqZOZUhpFlPeV9L8wWJxFuT7KyVh2H2d1JYznmIpiad+0KzxpaIm0urTGFZcExSIygsgIIiNIGrFJb/BSLF1idYnVJVZfsRsywpARhowweMTDSKVUf8MBXmHeTohigCdfwWGo8IcYTR8nln9FUDOkYoyKlnRPjzV+xZLqHl/O3HUVW744rovM8r1xrdb3nq4yu+xjIek9XYV22U3kT1app9zklKz3YiWrF+2iEoSst66KzxWybYn9j3Ss8EuY3qFDnINHkvak4vtMUrJO+rpMpV6Um6lUjP+uBKe2JW1b2o603Z0FeFvxTm1L2ra0HWm7/1b5DZm4b1bTD2n5HZ0kqvzNBP769GkIkXmDH99H4ueT5Qfjexg2P4DNzwKfVzA9nOoeBoj+3sNxg/+wslb0xPdlSfW6vmWu8SpbDETqJanBjtTLlhqZeqlas2MIb4L2Jmi7K3rJUlayiJUsX2/CVaJwiR917yv/ACqnz8U="
    },
    "hip:compute_tensor_fp32": {
        "backend": "hip",
        "entry_point": "kernel_compute_tensor_fp32",
        "entry_points": [
            "kernel_compute_tensor_fp32"
        ],
        "filename": "compute_tensor_fp32.hip",
        "format": "hip_source",
        "sha256": "226f902f39c1f372814c6cbd552f700c01c8df18e2aa055a216fe70b23e73c26",
        "size_bytes": 2593,
        "workload": "compute_tensor_fp32",
        "zlib_b64": "eNqtVm1P4zgQ/p5fMQcScmhp89bCtaxOPShapN2lYtlPe8hyEqdEmyY5x+GyWvHfd+y4pWHb04nDKnFmPM/jGfuxw/DYgmOIRFFVJ8uyhm9c5DybQFSsylpyKnleFYImpe8BeUhLW4XPcym+Q1mkuSSVPTEgugOjomd/Xk+AJFnB5DEUtaRhnfShRrDvUQl5vaI84yuey2rLnUoumEyLHJ0aCxXnMWUdK9TpzET0AEOYN2piRMCyZiKewMXlpxmQZdL87pwNdcd0FzS6Gzk2UEwmzWSaU7aKl1FOV8mK0QQzcMcN/oKkreFm8Zku5rf07v3tfHapX6/v5rcTCHDwDP+Ihw93vH4EmM84sOEduI4XAMHV+3j1cVZBA/h69eFmUQ0zlnOV/9CyDtMEfot5kuY8JpS2KVNqw9HRL3622x84xv+XBdhejrp7UN5u/0ixWYdciELAwZdPn78sFje3d/PLCby/XuxUh+B/16ngVWfZe7auG64WGPGAu/IPE/wAifM4TSxLfi85Tgw17tsSZwfcfah9b7oZaTdbTdAEqApKmZQiDdXklBDeSPrII4k5KAAJbNueWha6UZFwcHGAgGVWhCyjFB6LNP4XqRK9bkamlGIlOFEkEbjWrA7A7LqS3Xi3FPtMtVbtS09oww/tizBe1wxLTO8dhFkRfbuOm0GDMtLGZbpCowfyQXAW6yGs8RnbsjL6yDKlt4GTTH8ZDWlZVDjqDEY7R3O+xNGTPcOs4gaLaTgDD3tUfFsZnKioQiTGtm3oaYYdrQsNX0BDBcUYZ2yi9Ow2UetyBO6otk3ZGzVEDqb1Q6XXb5NUcBfRW2bX8k9HydP0BYu7ZtEhnfhxF37asc52kXlbZGovNvFuNzN30LV2ZuZ3yDrx4y78tGOZzFo2PMGk1afaxKnqz7fEOoVeL5VrOap2WAq2XDE8laLIso17Q1SyVLRU+u0cglqRKGObRstI7dB/uWSJ1m+/FWofcX0wP3vaZXRfyejuZfReyejtZfRfyehvM/7fhcQz/cYL2TK+5UK2jG+5kC2jv4vxyWqfusMvrr5Zzju3+bZ+t6/Aql5hJiRyvjr3eMKwd03vmd6/V7cXiVwT4ZoI10S4bcS+qxEbiTyD9QzWM1hvw+6bCN9E+CbCVxHPlZov1lcs8B7zTiv8tuOJJ1iGDX/oaia4ruqfLcoqqmskTnN6deGoVhuqJ+vJ+gks9Nwk"
    },
    "hip:compute_tensor_int8": {
        "backend": "hip",
        "entry_point": "kernel_compute_tensor_int8",
        "entry_points": [
            "kernel_compute_tensor_int8"
        ],
        "filename": "compute_tensor_int8.hip",
        "format": "hip_source",
        "sha256": "ee478b00f7a93428082250034ee8ccce5865c554c6944b170a2f3a29705166dc",
        "size_bytes": 5814,
        "workload": "compute_tensor_int8",
        "zlib_b64": "eNrtWNtS4zgQfc9X9EIVZZPsYslOxiTsAztka3hhKIZ5YmdVTqIE1/iS9QU8xeTftyVfojgOCQxsLVVjUkFq9TlqtaTThqPDFhzCOArj+NfZPIWvPAq414dx6M/ThLOEB3EYMTdIbNBu3bku3IdBEn2DeYhWLdb7BYg1YIT36R/nfdBS7JqUJYcQpgkbpdMOlCYIUp9xj/s8SGLF7CY8chI3DFRjzPmEOXXDSMZ1Go1v4QiGmYgAcTBLnWjSh/dnF6eg+VPfYS5iSC+TH9fWIYzgCkfNIwu0e7/mwNzUlsQfLz+xy+EVu/5wNTw9k83z6+FVHyzjuAea4O+DSSEDQm0I5/GR5wR8UFHfO3fcpH0gPXSh3V7lItiPWq19dwq/TPjUDfhEY2w2zY4NmzEdDg7W7E6z3TIK+18twKc+SjagaLO9+xgbIYbRCMMBsmmAPk5obsB1N83UJY8R0k0RUhlha59HEW793ueLT58vLz9eXQ/P+vDh/LLx3Ef8n9SNeFyco3x32suz05Y2kRU0nl9c2+A7SeRmSIaoPZwsmLjTViv5NucYDaR4OGcYEiA7pCYdVCPC4DYYMnHQGXMSpB2J8Jim8Sxhd3ycYJTCW6O6rq8Dre1AqxFobwfaEthCM95/2Hu/h4CZF44cjzG4C93JI8KgyV3DxR8iCLOE04wThJXiUA6vakNlVaShshXKsNof6fAgLWP0ldmGGQb2O4y8cPz1fJL9luH9lp0z18dOG5LbiDsTOTRQoJgUGDkxR6yGbV3TBNHfxbRlA+c7ACP7E59UJEfc7KaL/f07NN3rNXN+rdGcH/OmW72OoY1meZVqC3LYnePhiozMIPnP2pLZPIy3eAR8VmYlX7pcPq5eOJancGygz4PMYBtIB4oWrVpm1bIWg1UoUaDdyq1Xtd5VLbsOpQr0uHIjxrK5jIXQOtpUY14GSKxlcxkP6S2KNU9RGrT8nIrUDcTvE+XQDqDddpPyaIpnfx45M99BbYhCz6vMFdHccaOcSrZOwEoFieioNHJnRKIZ3iLXS9yAOf5kNg5YQ/XT5O538i3uIK4DxUcfrDKSZzKSjYz0mYx0I6P5TEZTZfzRROJNeOFE5owvmcic8SUTmTOaTYyLVv6dX6spSNE8WVF19fwuVTpOfaEp2NTHxo3xBe9X2SFqh6odEzsri1Gf3ImoXETlIioX2YWLqlxU5aIqF92Fy1S5TJXLVLlM5Fpmt6iWN5jUL5gtzNmgyPc+9+qVp3wpqheG8p1oRdXlG0dZHR6W4t9RCsFi0IAp68XTMHkFeaiXkM5aUVkoVcV+YlXpNGzBk4uKXa9H24vKI/O+7ZrS+AcTu0eDPFZaEqW8A4VM5B2l1kwdL+Y7CeQPzEM2zPNy66kk/5XXUxWC+jz/O4k1biwV0VU7PbXz7r+Va6LGRdS4iBoXwbh2lNiY19TM2q6am9prymhtV9Mncu2ssrtYfirxW1Di19fg11ff19fdn4r7RhRX/iNt0foXGzkt/g=="
    },
    "hip:memory_pattern": {
        "backend": "hip",
        "entry_point": "kernel_memory_pattern_write",
        "entry_points": [
            "kernel_memory_pattern_write",
            "kernel_memory_pattern_verify"
        ],
        "filename": "memory_pattern.hip",
        "format": "hip_source",
        "sha256": "76881b7f22a503f2fc339be2f3e8ef33e61b5ea5929ed95cfea0db5f7c61d755",
        "size_bytes": 2023,
        "workload": "memory_pattern",
        "zlib_b64": "eNrNVEtv00AQvvtXDD0gOwmkidOUxk2lJDVQqa0qKAeEwHK9G3eF443Wu61TVH47s2sntkmDEOqBOfgx882389xuy4IWRIJn2at4qeA7FSlNRrCgCy5WwTKUEjVg37Klo5F+KsUKlpyl0s6cUYkPmvDgXjBJOzuMd1Sw+UqTTaZnYKAjAFshpdsPZAtu1LwD+nc4CCSkahHcc0GyQqchsKZacEJr0IxS4qyJi2NGYEc8zSQ8E32nRkSF4CKIuEplcaqIbqELfi5pmjGeQqxCQUYQJgm8P7vqTi5O380uIUQY5hxJJWim/aafr/2PwZX/IfDP/Qv/8noEA7B7kFcR6QhhSUVRLuCiTA9DzTJ9eNey5GpJCZ2DwsPjlBJIeBoXDzUceNt2JAfl9j3LymQoWQRBQOgdi2gQ4Oeci4iyNGGp/kfcpiw6GBspgZG8YyxloVBlWgA/LEBhc7C1BcZj2FcOCIopp7Cf75eivC1cr457W8oTuH4dd1DKEzi3jpuU8gRuUOF6Co6PcR7dvmNjhvAS3N6n83Nn2+mgcvpp73JDP+Oo6/QA4xrim6kWvmzzPjkBtDjQxkiPfPfw8Gh6VEZq/B408kHDekOEtRD25sCfzibD6Q6YW8Jm/Wl/4rvr+pQhN/g869GyaG6WfW+2h/2PE34TJtj6O87In/bcXqfXQi8caSlYJNFP71lhwrmo1mxTjMaWbYD1ASo3F7UxhqBLNxw4NwmPvp+R/HWOuVWaU7ZATbvQyFtBQ2JA3m9MOjzdvQIY40/hucVVOOIWwHrW0Qnj8MzncZVRoWiPS+p18FqwBF/Q+BU9G8tjFqd5y5i8izMf/6kXxYVg19N9xpZ0djS5fgf+P13TuaAKS2biy0BfQM/RUL39m6a++Muu1gm0tNv12LyN7dGqnvqgRgYn5gqtiELJFyyaEGLXWtBpZF0bp182yGgp"
    },
    "hip:memory_sweep": {
        "backend": "hip",
        "entry_point": "kernel_memory_sweep_write",
        "entry_points": [
            "kernel_memory_sweep_write",
            "kernel_memory_sweep_verify"
        ],
        "filename": "memory_sweep.hip",
        "format": "hip_source",
        "sha256": "a478f86ccd64b7312458ba540a5466940417236cff89556504eebae7f5641456",
        "size_bytes": 1605,
        "workload": "memory_sweep",
        "zlib_b64": "eNrNU8FO20AQvfsrppzshJI4NgmJC1KcmBYJEGrpoaqKZbwbs8Lxurvr4FDx7521nWCgHIp66B5Wntl5b+bNjHsdAzoQCy7l+yQv4JaKjKYTWNIlF+tQ3lGag3nDckvHBZkSa8g5y5QprUkTHbaDwzvBFN3949OKCrZYa6KpfwJV4ATALJBu6IaqA9fFYhc2JmTFMvx5xwWRLaeklFgbippwAmbMM6ngr4lq0xloCBWCizDmRaZqfhHfQA+CUtFMMp5BUkSCTCBKU/h0ctGbns0/zs4hwjDUEatCUKlx/rfL4Et4EXwOg9PgLDi/nMABmDaUj6mrWiCnou4BcNEogTySUmfvGYZa55TQBRSYPckogZRnSX0VQ9d7+Y7sUDgDzzCkihSLIQwJXbGYhiF+LriIKctSlmkbGUDmKVNLVg5dU5ulBb8MwKONezjEgrvQL8eBMxqN/fHo2J2OZvb+19NTrwrTIeY9XOnr6AicvmWh+n7pH+8fuKPh3J4FbrDvj18DDEYNYOzO++7Y923HsW078LcAQbGpGbRy2JZnPBgGxZngw85sB4UlKb+OUtS04oy8vpHmRlsHMTgqJVisEKU3ZSu7tShbX7VwTWuaNUNvgrlQDn5a1ymPb09IuVeinEfPnC33dAsrj7oRNCJVkPeMSVdC6IYsQaNGvuCqgTjH6gEYKRGEdXjV54dW8bWne9hwb6rXB+V+x8cfCG3NX4df1UrrLA9vaHO9wuZTeW/pdmM5g2fg9h/6/wwE68S/MsbeVPVJBPaLfzIrtgBzO693rw2sjdCn221X423fHozHWzM/qfkIa24TRYovWTwlxGw1ffeJztam/AYxDutE"
    },
    "levelzero:bandwidth_d2d": {
        "backend": "levelzero",
        "entry_point": "kernel_bandwidth_d2d",
        "entry_points": [
            "kernel_bandwidth_d2d"
        ],
        "filename": "bandwidth_d2d.cl",
        "format": "opencl_source",
        "sha256": "42e780572119decfce3d1647f85f5d6714a8e78d2cd286a938bbe9a7a98b3df3",
        "size_bytes": 647,
        "workload": "bandwidth_d2d",
        "zlib_b64": "eNptUFtrwjAUfu+vOI9VV5xOxqjzQaUPgttEZTDGCG1yVsNqUpLU2/C/L71Y6DRPSb7L+c7XbTvQBqqk1l6cZvCDSmDiQxQKtufMbAjrM3AT3GFyQiVbOT0QRh0hlVwYV7f8SkQampw3nsx8cAmJExmFCWSWP2gD0+YO6k8qhTYXSCt6B1kiRQwi25Id0kExcKzoBroQHAwKzaWAOAsV8+EtRTGdw9Riq8Vs6b1buzBBTZFBr//kRdyANgrDLbeWVKbH3G3ysQ5WZBEsSTAPXoLXtQ8PfXB7j150NFhmASti0IHm515xg3miruMQUq4NO8nZzQpcB+z5v75Cm4hTU/TQZDS6qHl5KQWvgpv1wO8VFts8I3CLRytGU9kTztz71vCKno9heFOh+QlrzbdUFQU4O1i+nTMsrs91nPLdGVWml3T5set+WvDLCu1GxbX0PTtn5w9kH8+l"
    },
    "levelzero:compute_bf16": {
        "backend": "levelzero",
        "entry_point": "kernel_compute_bf16",
        "entry_points": [
            "kernel_compute_bf16"
        ],
        "filename": "compute_bf16.cl",
        "format": "opencl_source",
        "sha256": "7449a3e5df7e30f2484ff2722f8d5be90e062d9bd438ea351af42f4775dd93d7",
        "size_bytes": 469,
        "workload": "compute_bf16",
        "zlib_b64": "eNp1kM9PwjAcxe/8FS94GUSCXjyM0wwzkogsAw3x0nTju7HQtUt/IPjX205i8OCtfXmv/bw3HQ8wRqmVMZO6cziQliRilKrtnCVWVPcPiAQdSXyRVqPgTqXVZ3SqkTYyo/iSYdeRYEseFzEixmqhCi5QCcXtGMpZVrjqFs7HIV3LSFBL0pqL1FjS3DZKeqHPwBDtGP9zK3qQRJd7TJGeLEnjE6gd17sYb6/rtyxb5Zt0Dq+us0U+effGl1ADH74HPhu79yiQ/qsjwZRccI2+7dMyCY+vsjXL0pxtnvM0mffHxSbNY9wholMnmrKx4oyaW9rNEMZBxQ8Eap3o+WcB9XfIgKj0z56V0tgut6HDdDC4Ia29MLyijq9RL/iClwfzDy8aaax2ZT/bcPANsi6blg=="
    },
    "levelzero:compute_fp16": {
        "backend": "levelzero",
        "entry_point": "kernel_compute_fp16",
        "entry_points": [
            "kernel_compute_fp16"
        ],
        "filename": "compute_fp16.cl",
        "format": "opencl_source",
        "sha256": "f56a0bccf81ea46c594e44ac9b6e53ffd57b01c3dc2f25c29c6036a8b3ea6e83",
        "size_bytes": 3559,
        "workload": "compute_fp16",
        "zlib_b64": "eNqdVttupEYQfecrSmtp1dhjmzvWeJ3I2cWKJcce2V4pShQhmGnGyEwzaWDlJPK/p5qL3c0lG8FTFZxz+lQVKjg91OAQ1jwviuPtvoJnyhnNlrDOd/uqpGGyNz0gGf1Gs78pz3WBDljJ/4J9nrKSFPqy5YQyRcAuf7peAgnDbZbHUQZJlkflIeRVGcZVsoAK6cCqXUgzuqOsLNpbaUl5VKY5wxs1BwpKN2GkZHFt5JKvn+AUgpeSsgIZsK0ivkHzWfj8xN+M3K0ewlVwHz7+fB9cfqnD68fgfgln+BDLOwQHLsA1LSBn8BRliQXRel3Ai3h49culiCzIIkab6OrmblUIA6eadpAmbEMT5cgDynnO4cPX24evq9Xd/WPwZQk3ooXwG/ZQ7S2nf1YpR2VJAWhX0QcUY5s00Q72PNruIqwluP18A8Gvj8Htw/XdrUJbAmVRnFFNC8NmKPAtTzdjAyIa4NUfDvooebou36ZUo9Y4i3JkXv2H0uSkR8oMx+/HOvzTF9ui7wsgIta3tGyNhumGGPp5DU4TIAL1w4Viq9MSF6dlxVkDf9WkI8SUIQqfxBEi1pNdRIwTy8UXs7EKx8JhzkmT6gswTtykPVoSiff/oRKrKnGtYo3KMEmGHIuz8L0kdZN0cQZSehx8SzuGpRMsZiEqGkhbEIf7vJCgQm4BregQzOhWBrMazN4dNDhuSKDWtnFiYnFwhEUaniVXQOpJ4YNKh49gupWOrXhnnfn/m9Zabk2Yc0yY80yYUyasOSaseSasKRP2HBP2PBP2lAlnjglnnglnyoQ7x4Q7z4Q7ZcKbY8KbZ8KbMuHPMeHPM+ErJmoXCX4ESftlQCdGdS6CT9J34hyOjtJS3tjdd65iPM+yt9vvUvso5a1YHX6Cs0rIiEQWqte/2FFiJ3MDd+OiWYRti94wZocxpzFWh7GmMXaHsacxTodxpjFuh3GnMV6H8aYxfofxFcx3GoTL/7sNGsEMGjSCGTRoBDNo0Ahm0KARzKBBI5hBg1TM6/Cvof1lqb98zdvPjRNMj+TUPB8yTIlhqgxznGFJDEtlWOMMW2LYKsMeZzgSw1EZzjjDlRiuynDHGZ7E8FSGN87wJYavMvyaIVHEQsifkZAWScpwr5DCwDX0UcrNXm71clvkyovRXu8Qp0dxe7nXy319pKxqJ+qqyxGesCicMcZ2EzsidpvYE3En0v6J/45b9g+hgOUqh1U7XYcfa/0lREVYH0eMF//qsyGuCnVetX8B6J28ag=="
    },
    "levelzero:compute_fp32": {
        "backend": "levelzero",
        "entry_point": "kernel_compute_fp32",
        "entry_points": [
            "kernel_compute_fp32"
        ],
        "filename": "compute_fp32.cl",
        "format": "opencl_source",
        "sha256": "2dfd404af458f9acc30c25a97cd43a03a8d0aace6540f303cc9bb54ea5581534",
        "size_bytes": 2305,
        "workload": "compute_fp32",
        "zlib_b64": "eNqNVe9vmzAQ/Z6/4qRJFTRNCySQKmk3ZV2qVerUKK32ZZoQECdDBYyMXbWb+r/vbKDlh6PGH6I77r3n4+Vszo4HcAwRo0Ux2uUCHgnLSDKDiKa54MTf5mMHjIQ8keQvYdSU6GXG2QvkNM64UZiziuM3KRK2+HozA8P3dwkNgwS2CQ34MVDB/VBsT0AgHTKR+iQhKcl4UT2KOWEBj2mGDxQHCkI2ftDKQtXIgkV/4AyWz5xkBTJgJwK2mcFdTrKrW7jC2v3qZj36CRkqPhG4XuHbXP9YSPLd6t5fLdf+w/f1cvFNhTcPy/UMzrFoe/jjwCU4rgfGOQRRJFKRBJyyAp4RgiKQB7HKlGQV3N6tCtnb2WDg+6Ux8ETjjc4kYwC4ugYxUnAWR/zNKYWK0A+u8axbbLjXKLV81D8PTfjXFdth35dgyNjcEV416scbwzLnChxvwZCoz5ettmotuRjhgmUl/HXQ2zzAHbZpYFinjotTUfYII1mlzChT8wSsU3db7dlkh35OC41C2FYIlYKzRyIjO5QYyS3wX1eS87LREsEsLFunNvJhiIHlOQppqLJpKAewIEw4AtsV1SYV2T6IbOvJzkFkR08eH0Qe68mTg8gTPdk9iOzqyd5BZE9Pnh5EnjbIJZuycs7xAEkFMZfBReM4zWE4jHlzsD/lLNilAYiM0SR5e/wuJe+HSkyFF3AupIxMmkLqlFjVFDPrBPCuU1NYvdobxq4x9n6MU2Oc/ZhxjRnvx0xqzGQ/xq0x7n6MV2O8/ZhpjZm2MB8YhMf2Q4M0mJ5BGkzPIA2mZ5AG0zNIg+kZpMH0DGpjXvuXq5w/+oi0uNjGGY4x+oYzf9TI7U7udPKxzFt9VOsdMulQ3E7udfKp5v4tRCo/Mvj/DkH2NMTYkfG4jCcydsvYk3EtUn0ff+GR/i0V8HWbm6GuacIXpT+DoPDVdob1PL2+suSSh/918B+V3WVV"
    },
    "levelzero:compute_fp4": {
        "backend": "levelzero",
        "entry_point": "kernel_compute_fp4",
        "entry_points": [
            "kernel_compute_fp4"
        ],
        "filename": "compute_fp4.cl",
        "format": "opencl_source",
        "sha256": "6edc40f8b7d47bc6fac77fe00ab5798c30b0360dd97d599284b3433688b20886",
        "size_bytes": 431,
        "workload": "compute_fp4",
        "zlib_b64": "eNpVkD1vgzAYhHd+xSldSNQoHTrRiSpIRaoaBEmHLpZDXgiKsZE/8tFfX5tSKd3s05393K0WERaotTJm2Q4OJ9KSRIJa9YOzxJrhGbGgM4lv0moezJm0+oZBddLGZp5MEXaXCK70NU8QM9YKtecCjVDcLqCcZXvXPML5NKTrGQnqSVozSZ0lzW2npBfGDAzRgfF/t/3Iker6iBWyqyVpfAKt4/qQYPdR7YpiU26zNbxaFXm5/PTG99ACX74GLp09ehRI/9WZEErmVRoe3RQVK7KSbd/KLF2Px3yblQmeENN1EF3dWXFDyy0dXhA2QcNPBOqdGLkD2iqKHkhrpTG7g0nuCSYqweuT+cM4evwL178805zopLHa1eMks+gHlhmPtQ=="
    },
    "levelzero:compute_fp64": {
        "backend": "levelzero",
        "entry_point": "kernel_compute_fp64",
        "entry_points": [
            "kernel_compute_fp64"
        ],
        "filename": "compute_fp64.cl",
        "format": "opencl_source",
        "sha256": "b2216a02bdc49ed2bb873b7a9978cfaef453aa9ce9c424306d70c8deb0106ebe",
        "size_bytes": 2449,
        "workload": "compute_fp64",
        "zlib_b64": "eNqNVW1vmzAQ/s6vOK1SBU3aAgkkStpN2Uq1SlkSpak0bZoQJCZFJZgZXHWb+t935iWFAGr8Ibmzn+e5851tLs8kOIM1o3F8vo04PBEWkmAEa7qLeEJsLzL7IAfkmQR/CaOKQFthwv5ARP0wkWNllHPsMkXAJp/vRiDb9jagrhPAhnI3IGdAeWK73OsCRz6EfGeTgOxImMT5lJ8Q5iQ+DXEiI0FMyMZ2qq6b5jJh60e4BOslIWGMHNhyh20w/8B+emT7XOaLe3thLe3V16U1uUnNu5W1HMEQFzUTf3S4Bt0wQR6Cs17zHQ+chLIYXhBy+20CkeOnni683JjOF7HI4lKSTnwv3BCvEveEMEYZfHiY3T8sFvPlyroZwVSUEn5gLas1ZuQ39xmJywpAim19QLFw43vSScSc7c7BDVmzL1Owvq+s2f3dfFahjYCEDhZKkmw7aw48U3/T1ChZAhy1JmEiCfPXyb5bKWyNPUka+na4WOpgaanay5YFV4F/h3JbTP0aZGErW5Lkudr+RlaVcQr2PZAF6uN1JbFCSwxGEs7CDP4q1aM7GMLbObJ6oRvdPEk4By+glMmZq3RBvTDykBWya0c0rgu4VQE3FdBbFEKyRYVzjICnMRUcZ2nmCKbisnqh6QZ08F81dQGUs1VFTvePC1yBU9AMngcpyNoxZK2FrB9D1lvIvWPIvRZy/xhyv4VsHEM2WsjmMWSzhTw4hjwokVO2h4+FnF8gIcDHwrgqXacxdDp+Uj7WxXvAQ0aDYD/9JiXerVwsNa9gyIWMcMpC6R1R8zPM1C7gc5uewnxre4xWYLR2jF5g9HZMr8D02jH9AtNvxxgFxmjHmAXGbMcMCsyggnmnQHhr3y1QA6ZWoAZMrUANmFqBGjC1AjVgagVqwNQKVMW81p9Wcf7oE9L82PNDPMZYNzzzpyVfO/D1A78n/Eoe+XiD9A8oxoFvHviDpvc35jvxjcEGd0Ak1UFbF3Yvs/vCNjLbFHahkn8gf+Kd/iUUcL/laKirKPAp1R+BE9tZPFl9GdzeDtXSeJii4qv0HyFKljs="
    },
    "levelzero:compute_fp8": {
        "backend": "levelzero",
        "entry_point": "kernel_compute_fp8",
        "entry_points": [
            "kernel_compute_fp8"
        ],
        "filename": "compute_fp8.cl",
        "format": "opencl_source",
        "sha256": "baf74d50671cc0abb5e4423c1051009f16e34df8df2e4ceff9e821771d81a906",
        "size_bytes": 431,
        "workload": "compute_fp8",
        "zlib_b64": "eNpVkD1rwzAYhHf/iiNdnNCQjsWdXGKooTTGTjp0EYrz2jGRJaOPfPTXV3JdSDfpuJOeu9UiwgK1VsYs28HhRFqSSFCrfnCWWDM8IxZ0JvFNWs2DOZNW3zCoTtrYzJMpwu4SwZW+5glixlqh9lygEYrbBZSzbO+aRzifhnQ9I0E9SWsmqbOkue2U9MKYgSE6MP7vth85Ul0fsUJ2tSSNT6B1XB8S7D6qXVFsym22hlerIi+Xn974Hlrgy9fApbNHjwLpvzoTQsm8SsOjm6JiRVay7VuZpevxmG+zMsETYroOoqs7K25ouaXDC8ImaPiJQL0TI3dAW0XRA2mtNGZ3MMk9wUQleH0yfxhHj3/h+pdnmhOdNFa7epxkFv0ApLmPxQ=="
    },
    "levelzero:compute_int16": {
        "backend": "levelzero",
        "entry_point": "kernel_compute_int16",
        "entry_points": [
            "kernel_compute_int16"
        ],
        "filename": "compute_int16.cl",
        "format": "opencl_source",
        "sha256": "d8dd8bc96d4ac8756262e9d7e0c2cdc2181c1a7b3098da15232fa83485f43d32",
        "size_bytes": 1956,
        "workload": "compute_int16",
        "zlib_b64": "eNqVlG1v2jAQx9/nU5w0aUpgiKcSGJRJrKUa0qYiWu3NtEZ5MGm0EEeOjdgD331nk7KYedHiV767//18uZzdbVnQgpDRoujEuYBvhGUknUJId7ngxEsy3nfBTsmepD8Io46ULzPOvkNOMWgXzrRM8rQcqVu8X03B9rw4pYGfgsBAC6jgXiC2b5QJmdh5JCU7kvGidCWcMJ8nNHtxFIREnl81AlXGgoXP0IXlgZOsQD3EwmfRFO5zkt18hBuMPaxXm85nyJC3J9B3O0HCQaA6zkgEnxa3knO/fvDWy433+GGzXNyq7epxuZnCBIP49S0YwBwGI+zDBPwwFDuR+pyyAg4yLDJG0/SEk64B0LyQBXYty/NOvYE9TSJjn2wLcF30iJGCsyTk52YpUYgt4Ya2XQYrDbwMla00ugMHfl4GYqx6DrbcOzHhZZVeEtk9Z6bEyRZsqXo314p6YcnFCBcsO8mPVvWI4pkyDthOdYiyHNs+FQmvoXe4wyUc+AV9UZ6nZfpR9HdmYMhUqWUS62k5qvo2yvA/9w5vl8Mxpj2VnXIqrJkG6Zsgg4aQgQkybAgZmiBXDSFXJsioIWRkgrgNIa4JMm4IGZsgk/+AKMqWstPM41VCUE/M5Oa6crFm0G4nvDrkr3Lmxzu/fA3O7j+ogpO8hKntNb4dQnKkVSWpK6PNKFotdUvacuLLTz1LtUlEq0aqzRtaNVJtqtCqkWqzg1aNVJsQtGqk2hwwt06q/W20zNLjPx8gfNO1ean0/Qn76yCq2jX0DdGn1WBa1Z5gzkjnuNI3dsoCy4f+C07q1/OLi3XNrKP1Gz+aJ3E="
    },
    "levelzero:compute_int32": {
        "backend": "levelzero",
        "entry_point": "kernel_compute_int32",
        "entry_points": [
            "kernel_compute_int32"
        ],
        "filename": "compute_int32.cl",
        "format": "opencl_source",
        "sha256": "8a4ef835f5b641efcd34b3e42723718fc4cafd34c9b045eace790588d69d496e",
        "size_bytes": 1560,
        "workload": "compute_int32",
        "zlib_b64": "eNqNVFuPmlAQfudXTNIX0BoVFC+sTdxdk5q0WeNu+tJ0CZcjJUUgh3OMvfjfOwO4INENPEzmzHzfMHwzh35HgQ54PMmyXpBK+MV4zKI5eMk+lYLZYSwMHdSIHVj0h/FEI/gqFvw3pAkm1UyblyT7gkO45f16DqptB1HiOhFITHQgkcJ25e5jfoRY7m0WsT2LRVaGQsG4I8IkPgcyxnzbqR/cvI0l935CH1ZHweIM8RBIh/tzeEpZ/PAFHjD3vFlve98gxnoHBobec0MBEtFBzHz4unykOk+bZ3uz2tovn7er5WPurl9W2zlMMTk00eiwAH1sgjoFx/PkXkaOSHgGR0rLmCdRVJSjkA5JmlGDfUWx7UIbOCShf1UnVQF8GhpxlgkeeuJNrBzkoSTiimzNZE3AZqqU8mrY1eBvMxFg1wtQydcCJsou7dBXB5qVg8MdqIT6tLho6lyLHs6E5HEBPynNV6CY+IqiMfgHQ2k1EY7vnxFuicgheZIPqEHqoIsZnNXgOFsZk8nsfiY1eC0LWzXCsCLorQh6RTBaEYyKMGpFGFWEcSvCuCKYrQhmRZi0IkwqwvQdQs7YJbzYEVw9pA2kRc5dbREt6HZDUV+KDyl3gr1T3p63cFUqEywti+XuHd41SXXoVK+UrxhtAZpOvk5dWhnrEkBTR3MbQFNGcxtAU0VzG0BTRHMbQFNDcxtAU+LmOwCaCpqrgFP9gpU/ju84wR80SdTmFb9fQ5KKX4m+Ufgj8seFb5I/wYt9Uv4DrHGv1A=="
    },
    "levelzero:compute_int4": {
        "backend": "levelzero",
        "entry_point": "kernel_compute_int4",
        "entry_points": [
            "kernel_compute_int4"
        ],
        "filename": "compute_int4.cl",
        "format": "opencl_source",
        "sha256": "5df56838d9ed5686ca45db556a9156a287d35bb6beebb9b0dc9f8b62c059f038",
        "size_bytes": 435,
        "workload": "compute_int4",
        "zlib_b64": "eNpNkM1ugzAQhO88xSq9kKgRPfRET0RBKlLVICA99GI5zoZYMTay1/np09cgInHzjGftbzZZRbACYY1z67b3cEGrUaUgTNd7QiY1vUOs8IrqD61ZDulck31Ab8Jd7JbpNMPmI0Ms2xQpxIy1yhy4Ah/8FRhP7OBPr6ME7TuGCjvU5CZLElpO0uin4RCPjM/FYaTIrDhDAvmdULuQh9Zze0xh/13vy3JXNfkWgluXRbX+CcGvoQP8hhJwk3QOIKDDR1eEsWNRZ8Oru7JmZV6x5rPKs+14LJq8SuENYrz3SgpJ6gEtJzx+wLASOPELAnZejdgDWxJFL2itsbCY0aRzhAlLcXFxT45z4L9xOwFxGyg7JCmCdmS9GLeyiP4BS7ORtA=="
    },
    "levelzero:compute_int64": {
        "backend": "levelzero",
        "entry_point": "kernel_compute_int64",
        "entry_points": [
            "kernel_compute_int64"
        ],
        "filename": "compute_int64.cl",
        "format": "opencl_source",
        "sha256": "eb1a7cbf731cedca067447b26654b93f5b8a96594b7fb3f5b1e71393ba334797",
        "size_bytes": 1701,
        "workload": "compute_int64",
        "zlib_b64": "eNqVVNtu2kAQfecrRuqLDY3C1QYcKpGEKkipgkjSl6qxfFlcq2bXWu8ieuHfO2M76ZJAVfwwmp05Z3Y8lz1vNqAJkRRFcZbkGr4zyVk2hkisc62Yn3Ll9MHK2IZlP5kUNsFnXMkfkAt0WoU9rkn+Hodw08v5GCzfTzIRBhnoTPCkCUIrP9Sr96ARCFyvfZaxNeOqqE2pYjJQqeBkIA4UjMV+sHcKy0ymMvoG5zDbKsYLZECiAxmP4S5n/OoWrtB3v5gvzz4Dx4gbBk7/LEwVaEQnnMXwaXpNce4W9/5itvQfbpaz6XWpzh9myzEM0dlxUHRhAt2BA9YQgijSa50FSsgCtuTWXIosq8KRqQsiLyjB80bD96vywEak8cFSWQ3A73WZJCuUTCP1Uq8SFWFV1IHKvXYaNTRdZjUP20Mbfr0OlmDiE7BItxOm6kT9NLbatleC0xVYhPow2UvrORZ9kikteQXfNd5cjgXFO6rU4Dd0Hm+9N5ggjp8x4TFMGBSsTJZONubkVXdVXtkmX4lpER8b296OZj3XHV2O3I/9qXvVGTze2vBUp+KZ5I5B7p5K7hrk3qnknkHun0ruG+TBqeSBQXZOJTsG2T2V7Brk4X+RS/ZKyGpUcQcwQlt7pFwYG+FBq5Uqczbf5TJI1kG9xy/mv6EKxfI6WKle4NZrikMnM1I56TRkKJrlULdobL19AA0SiuMAGhYUxwE0ECiOA6jpKI4DqLEojgOoedL5B4AahOIgYGfuef2CfcFt/EpNxdo84f/bSLLwL1HvVXqf9EGlO6S7+L7sGn8AUfDWKw=="
    },
    "levelzero:compute_int8": {
        "backend": "levelzero",
        "entry_point": "kernel_compute_int8",
        "entry_points": [
            "kernel_compute_int8"
        ],
        "filename": "compute_int8.cl",
        "format": "opencl_source",
        "sha256": "f56995cf265668c7bee61539a7a9ee295e895280a09440bf9860c64c7846c4c0",
        "size_bytes": 2327,
        "workload": "compute_int8",
        "zlib_b64": "eNqVlW1P2zAQx9/nU5w0aXJaEEka2qoFpA6KhrQJVNDeTMNKUlMiUidy7Kp74Lvv7KSQNglqI9W92L87+/w/OycdCzoQiTTPjxeZghcmOEtGEKXLTElGYy6HQBK2YskfJlJb01MuxW/IUhwjuT0qfWjVRWOTLzcjIJQukjQMElDY34FUSRqqpyPzClwtKUvYknGZl12xZCKQcco3HTljcxpUX0KziomInuEEpmvJeI48LFQg5iO4zRi//AaXOHZ/dzM7/gEc460YqOg5ED4Mj8NYwopFMhXwfXKlY93e3dO76Yw+fJ1NJ1fGvHmYzkYwxEEff24fGw/OwXU8H8hwEyyIohzWiCQBZ9pCEGNqy4M0y/VKTywrl7iECGKexPxtITGPJVXDtU9Maot4Xib5spW5DX8twCfCLZHFQBjkDNeCHtAF8oKNq2xcYG+gxoYVTCrBgRQz2cR06qfosQkhJkYXHKWzG6D7I5BiQri4AMe24TM46+trZR+1e7vN3sP9vL1mb7e/n3uv2d3zK+7G2x5br5ZFaVGnsEpx3xpqttilnXoVLJcijuRb4e5qsVXCu4OVYt4dKsu6sTts0FyLfQ6mVOwFk+UqaTwnjl2IHj8B0dTF+daiNrHe66LAcUsqUxQluVSJmaS1bMpt3mwv/NOV16RRtRgOwlH9g3iUe4ffaF7LLpjP98gu3Hf68LDswgOzCz/OzvyXiQkH83q/TsxN4qijzQUy3mLdOuu2sV6d9drYXp3ttbF+nfXb2NM6e9rG9utsv40d1NlBlTXwE34jSHmUkXfUWBtnlYM9hm43ltVD9ikTwWIZgOIiTZK37vdQuWRZGcyYZ1gWSsfRb9VI5shqbbHpmMPZ1UU83ga0oNi0A1pFbNoBLR027YDWS/gfAFokbNoBrYzofwBoObBpBF5bLyz8/uojjfvziHtg688hZop2r7B9bZ8Wdl/bg7IKyuv8J+r+CyMEOdXSEIxnvhb/AXsMerg="
    },
    "levelzero:compute_tensor_bf16": {
        "backend": "levelzero",
        "entry_point": "kernel_compute_tensor_bf16",
        "entry_points": [
            "kernel_compute_tensor_bf16"
        ],
        "filename": "compute_tensor_bf16.cl",
        "format": "opencl_source",
        "sha256": "400d2b57fa3ff5828cd35c2b743cfb99fd2a541c04e4fcf7e820265290ddb779",
        "size_bytes": 2235,
        "workload": "compute_tensor_bf16",
        "zlib_b64": "eNqtVG1P2zAQ/p5fcRrSlNDSxoW+rIAmGEVDglHxIqFNk+WkTolI48yxEWziv+/suC+pKjS2WUrOl9w995zvfO1tD7YhlqIsd6aFhgcuc54NIRazQitOFc9LIWmUkB74GX/k2U8uRWCcRrmSz1CINFd+GQydK93gaayPjs+G4FM6zUTEMkgywdQ2CK1opJMmaESBXM8oz/iM56p0n1LFJVOpyPGD9YGS8wllNS2yfI5kfA9tGD2ZyOgBU83kBFPJKCIhs1JHUyl0QWdMyfSJznSm0iJ7piyONSpMcfDPjCncXdwh1Mn46Bps6g+kZ2Ncjq/peHRFbz5fjY5O7PbsZnQ1hA7+HODjzzfoAIewF37Ac0OAGhSIooQn6HR7cHp+OS7bGcs5YDYXhwMTpu15W2mST3jyJvLeFpdSSHh3++X6djy+vLoZnQzh3BQNvmLVNhZV8h86lbx8U6R3GCqfpInnUVqVHR5FOnmlBXwPcK2XH+NiiFgt+sBaxVhttaEj1n+u9MbKr1qXbP4eBfBrHWyK9A/BN/tgypUjStOJHwb7nrVub8OYxQ98AseneHLWmSGxIYRPu6eDEP0bpBU2rRpaNWx1jXpcqTuotkx5l7HLeyHVABh9xFPB+JUe+BUiRl6aIrMBRLQQpTE0WmUWhubZZJrz6dLUiKCiYh49z2r1dCJWcvRAmgk0UHRQYk9Xhwk7xkpIpwYBNCzAhlX3jOqekfFEk7DnjGzswDcVeA+ku2Bmvw8gNkdXGWEahmKzImpACGKsqHVtt991LbVxLexqTr06Rr+mDVB1Jz1nR9bZoSF2wdKL1EmSVl37M5KkVXPq1TH6Nc2RrFjiPPDdbTGV1ftmc7Byd/ah0UjV/EaYtVVINp0x0LkUWbb4vIQqWCodmN0ewJ42MEZZBbLtZaq3GCy0mixmIlSv+YxhE4qD0bf3oFl1eRN93VkvwMg/gJF1sL9jhpfq/zGrwFaZvXjV24o0AXstDmqDcPWIa7NNz0wvxmGrDLERjCROdpzcdXLPya6TPSf7r1xqXH5MHDZx2MRhE4dNHDZx2MRhE4O9zNLN+2+Y3HdzcGWS5tiSPqYQwEebyRBYSW1iOOX6p59Cs7TDePFevN+pzYOS"
    },
    "levelzero:compute_tensor_fp16": {
        "backend": "levelzero",
        "entry_point": "kernel_compute_tensor_fp16",
        "entry_points": [
            "kernel_compute_tensor_fp16"
        ],
        "filename": "compute_tensor_fp16.cl",
        "format": "opencl_source",
        "sha256": "b0dd7ebb31af353f62cf3427408a9fe2c08e8a8691585da605a7bf1fcbed5957",
        "size_bytes": 2238,
        "workload": "compute_tensor_fp16",
        "zlib_b64": "eNqtVG1P2zAQ/p5fcRrSlNDSxkBfVkATG0FDYqMCJqFNk+WkTolI48yxEWziv+/suG1SVdM2zVJ6Pvfu8XMvvv6uB7uQSFFVe/NSwwOXBc8nkIhFqRWniheVkDQtyRD8nD/y/AeXIjBOUaHkM5QiK5RfBRPnSrd4GuvTdxcT8Cmd5yJmOaS5YGoXhFY01mkXNKJAoReU53zBC1W5o0xxyVQmCjywPlBxPqOspcWWz6lM7qEP0ZO5GT1grpmcYSg5RSRkVul4LoUu6YIpmT3Rhc5VVubPlCWJRoUpDv6FMYW7j3cIdTY9vYEUI38gQ3vF1fSGTqNrevvhOjo9s9uL2+h6Avv45xg/f7lBBziBw/ANpg0BmkggygqeYH8whPPLq2nVz1nBAWP5eDI2t/Q9bydLixlP/4q6t8OlFBJeff5083k6vbq+jc4mcGlKBl+wZltLKvl3nUle/dVNr/CqYpalnkdpXXR4FNnsNw3ge4Brs/h4L16RqFUXWKsEa6229MPmn43OaPzV6pHt53EAPzfB5kj/BHyzD+ZcOaI0m/lhcORZ6/4uTFnywGdwEUXR3mhwCHFWMPmMabRIDFlOIHw6eB+GCNYhvbBr1HGthr2BUd/V6h6qPVPrNZHqXkg1BkYfMUVIptYDv0ZEGmtTpDmGmJaiMoZGs2YIbb5tpgWfr02NCGoq5tPLEJupilnF0QNpptBBsY8S+7vOLOwZKyGdGgTQsQBbVtszbnvGxhNNwqEzsncHvinHayCDFTN7PobEpK42wjAMxW5N1IAQxGiobe1gNHD9tXWt7FpOwzbGqKWNUXWZXrIjm+zQELtg7UXaJEmvrf0ZSdJrOQ3bGKOW5kjWLHE4+O7pmMrqI7M5bjykI+h0MrV8HmbtlJLNFwx0IUWer47XUCXLpAOz22M41AbGKE0g216meqspQ+sxg0PRfstxw2YUZ6RvX0G37vEuerpMr6DIP0ORTah/YYXP6X+xqqGarF68+teKLAX7HI5b07CZ2taA0wvTg0nYq0JsACOJk/tOHjh56OTAyaGTo988Zlx+Qhw2cdjEYROHTRw2cdjEYRODvY7SDf2vGNw3k7YqzQpsRR9DCOCtjWQCrKI2MJxuo3OcgmE9rer8vHi/AOE9g44="
    },
    "levelzero:compute_tensor_fp32": {
        "backend": "levelzero",
        "entry_point": "kernel_compute_tensor_fp32",
        "entry_points": [
            "kernel_compute_tensor_fp32"
        ],
        "filename": "compute_tensor_fp32.cl",
        "format": "opencl_source",
        "sha256": "0606dec74ea28c958e970aabbafa77220a6b00a0b7eb7d9fa59ea23919d57d48",
        "size_bytes": 2216,
        "workload": "compute_tensor_fp32",
        "zlib_b64": "eNqtVW1P2zAQ/p5fcQNpcmhp49KXqICmbhQNCUZVioT2IitJ3S4iTTInRmWD/76z45ak69BAs9Scz757/Jz92G3uWbAHgUiybH+eSrjlIuZRH4Jkkcqcs5zHWSLYLD1oAYn4HY9+cpHYKmkY5+Ie0iSMc5LZfZPKtmSq6MH7sz4QxuZR4nsRzKLEy/cgkTnz5awOElEglgvGI77gcZ6ZoTDnwsvDJMYBnQMZ51PmVTxf8xmI4Ds0YbhUK2MGzKUnplhKxBAJmWXSn4tEpmzh5SJcsoWM8jCN7pkXBBIdT7GeqULPVDzcXNwg3slocKXxL0dXbDQcs8nH8XBwortnk+G4Dy2cdPFHVh3XhmNoOW0XCO1qAJicIu6tC0mawRJoy4XT88tR1oy8mAPWcXHsqkWalrUbzuDNlM/CmE/JS7jb8PAAXy3A9qp8Xbtt7XIhEgE715+urkejy/FkeNKHc3Xy8BmPfqsyBP8hQ8Gzl+/1Dq4XT8OZZTFWCAjuknD6jJiIrnBTSLg4rhPka0XpqAB1k2/R1uZkSWWlqYreto/7NvzaBJsj/WMgqm/PeW6IsnBKHPvQ2sRpg8fusA7MKHyb0IYzw8iNQBd8libZOtC1idPo/CUw5vNy4L6J/KMI38s4BqppqKFpoUUhFzXDvopKhHFtG2oaYEurZvrVTF9lYojTNUF6bZuojXoLtCNXzAz9wClzVxTrBVEFQhGj5Fa9g17HnPzWto6rJHWrGL2K5/bWW7xiRzfZYaA6snUWrZKkjar3byRpo5LUrWL0Kp4hWbDEu0uMqNXJykPVOSpJ/BBqtTBfCVe13VR484UHMhZJFK2Hn6BSLxQGTHePoC0VjHLKQFpe6vTWjwArXgF11YvP6j3wpuzWJVr79ULZdUw1W73Goq/HoptYr+KFF+m/8SqwyrwereKrDb77+kocVd6q8vZWnh+5UDoMnEbmoAiUpca2jD0wtm1sx9iusb1nLjQ2ElCDTQ02NdjUYFODTQ02NdhUYT9VaZ7kL1jcN7VvGf43oRwJlmDDO11JH7yM6cKIs+ydfnBUkwbj0Xq0fgMG+4pi"
    },
    "levelzero:compute_tensor_int8": {
        "backend": "levelzero",
        "entry_point": "kernel_compute_tensor_int8",
        "entry_points": [
            "kernel_compute_tensor_int8"
        ],
        "filename": "compute_tensor_int8.cl",
        "format": "opencl_source",
        "sha256": "750eee268150ddb02f488648cf13b1d3ed219d9393aff7a72f260bfd2f16ed40",
        "size_bytes": 2152,
        "workload": "compute_tensor_int8",
        "zlib_b64": "eNqtVG1P2zAQ/p5fcQJpSl+gcVtKaeFDJ4pWaYyqgIQ2bZabuCEiTTLHRmUT/33nvKd0CKZZ0N7Zd/c8vefsTtOAJtgijOMDN1LwwEXA/RHY4TpSklPJgzgU1AvkEEyfP3L/FxdhQydNAymeIArxzIwboyyV7sjU0ZOPsxGYlLp+uGQ+KNxvQqgkXapVO3EhUGvKfb7mgYyzLU9ywaQXBvlGzLlDWdVZJmQmwr6HDkw3GhbjwVVMOPg7fM0AacVq6YpQRXTNpPA2dK186UX+E2W2rdBhkoM506Fwd3mHpc7nk2vwhvDQ6yYIV/NrOp8u6M2nxXRynpizm+liBF08HOK/mRuYAGcwJCddMMmgWgjCKIYNHJHE6vgs4MAkXJ4NNUTHMPa9VeDw1btoG/tciFDA3u2X69v5/GpxMz0fwWetFXxFsXZqKfhP5QkevwtpD6ECx1sZBqWp2vAYes4rypsG4NpSHWERwZaF/EmQjSLLHYOwfVgZie2jbDh2bi8b8Hv7wEXuZ2Bqu+FymbGknmNajbGRRHeaMGf2A3cg9twAv5LuLZ8kj0fQIphubSzShoPUvLgAM2LYWAdavW7nACUPuITkzhxqhUsG8X0osBSjj9gYZJH6DVPXswjil6EpJo3CWAdqLw/7W2jA3TJUfzU0t3Sp/LcVGbBkMc/iMVy35UfWzNzA9n2AtIbKABMo2ypxkiotwG5kVreweoXVT+V5sbLjoyJwUFjHhTWsQZMX0CdFJLFKsyREuq+jk5In6ZdmSYoM8uat8MaZ2UBq6dVYG6eV8RxDq+XJfOz02o8Ec9cMVCBC3y+2y1J6dLJiiXkKfaXLaKdaKFFPd764ujS9u95Q/+U3mDkUHx0zGbF2OkBtzMuaWBQi/1iIbBd6PyOc0//DKC1UZfRspJ/pwKwgGevT2ttSbWn1vVDr4lmwrcPYQuFLj9S8bs3rQWv3fOlVhvVrSUc1b1Dzjt9QkNQYkhpDUmNI3saQ1BiSGkNSY4jecdnx7Dn/ho3+jg3ENo4zDZ6NP9amdTs="
    },
    "levelzero:memory_pattern": {
        "backend": "levelzero",
        "entry_point": "kernel_memory_pattern_write",
        "entry_points": [
            "kernel_memory_pattern_write",
            "kernel_memory_pattern_verify"
        ],
        "filename": "memory_pattern.cl",
        "format": "opencl_source",
        "sha256": "be5b7cdd0bffe13ff633efa43918e577fa5bd4c5b3e2a06e6a6f5430fc2bc808",
        "size_bytes": 1951,
        "workload": "memory_pattern",
        "zlib_b64": "eNq9VG1v2jAQ/p5fcfsyObyoQEq7QqkEKJMqdVvVdpOmabPSxDBrIUa2QykT++2zHUMcoFU1bbsPvPiee+65O5+Pah7UIOZMiOZ0nsMPwjOS9mBGZow/4nkkpToBlJIFSVeEM1/jw0zyR5gzmkkk/J6NwtUg/MCpJI0nnAvC6eRRkw1Hl2CgPQCE8TRl91EKueKuwX0+aUCesmwKWT7DD4wnomF8sCGasYRsMIKQxN9wFhl6DmfMMiH/nLkBW6YFSyNJU2LJCOeM45jlmSzS8/g7HEG4lCQTlGUwzSOe9ODDnGTjKxgr3+315U3zEwSd5j2VYGkjyWY0Fppi9PkuvMXX4Q0Or8J34fu7HhwDasPSpAw6oBXDnPCid8C4LVjJF0KrOPI8IZXKGGiW0oxUi9PhqKiNJktb+V4v4acHyugEkPbBYACt3AdOZK5uRWvZspb393BtF/fW2gFcx8V1rR3ABS5uaO0A7rjEtXM4PwekS/ORqhJeQ9D+eOXvB3XLoF/oqTAVZwJNq1YwcCHfTL/UFzLfFxdqsr4PdaX1LAxOT89GZ1arCVxp5ErD2icKVlOwN91wNB6ejJ6ABRY27ow6wzDYdMiKrvD1vbXnYVzsnbqqNHluQZHh2dk7ToTkNJZmTQzA7s7OwriuvdXZi3OvlHs+VQp1N/Uff0qk1YJpglp2WBUaJU2P7UCEoCuyjZmonSivuMKrPH3z87wsoTioDyzrRp42VfsX5fyqIitbY/al+kiYyoqs65d2v1jXnfa7T9S/G0KjmvSZ1+z/TctIT1mszo0AAfqt+TuT1Ku+nearF47TJdBWr7vq+lvf2is/daJKDRfmvSyJihceR0mCnDY3KoU7F+k3TcBniA=="
    },
    "levelzero:memory_sweep": {
        "backend": "levelzero",
        "entry_point": "kernel_memory_sweep_write",
        "entry_points": [
            "kernel_memory_sweep_write",
            "kernel_memory_sweep_verify"
        ],
        "filename": "memory_sweep.cl",
        "format": "opencl_source",
        "sha256": "e748e4ab6694eadb9ffc9b57502cc324512cde69a704549f9a2f281ded48ac56",
        "size_bytes": 1642,
        "workload": "memory_sweep",
        "zlib_b64": "eNq9VNtO20AQffdXTMWLnQQRNyYhSUHCwUiRaEBAq1ZVWTn22Kxwdt3dNZhU/HvXl6QOkAqpVffBsmfOnDlzWe+1DGhBILiUu3GawR0KhskIFrjg4pHIB8QUzATvMVmi4FaB9pgSj5ByypQprVEdQ5oh5EFQhZ1XXfcoaPRYEB27UyiBIwCTkDjhcz+BLOEsbsE8izrVO7BsQX48cBHKlUUihtaKoiIcNSgCzqR6M1EH1oH3PPEVTRAyXVwLUAguSMAzpqpsIriFPfByhUxSziDOfBGO4DxFNjmDifZdXUwvdz+DyVOlAYWWhNzdCqL5+g6Z+xKJr/iCBrJkdL9ee1fkwrsk3pn30Ztdj+AATBvyUkHfgVIupCiqTgEXdb2Q+rLk2DOMHRqFGG1PZeykwo8XPpxfeDMt1Pty7c2upuez7SEwAmT+PEFjB1lII8OQSrcmAMoSynDVvjShakHzvmNWhtyCnwboU30u4VCX0oZuPvR6g8HQHQ5OnePBxN7/dDYucQXCXMJN8Tg6gl7XsnRburl7un/gDPon9sRzvH13uAX/flDjh85J1xm6rt3r2bbtuSu8QJUJBo0MtjU2ngyDkGo59cxpuH2HzZLl+XIKlErQQJXLVSIaK9dcsxe+cnPrHjXtsRahKys/rBhVnZDQ0Oxa45c0On2Ir0ZIusR1TKT3pR4NDXON13nG5euHhszK0j6saVf6iqML/Kad33VoY9gF/Kaqpcrz9LaWVqv7rKcbt/XvOtvZZP7Ddf5/IyhyQ8IDbS8FSB3Szf7ReGgE5npE77bNqBlRnHa7qWe89j0Zv58F84bqI626SVT9KIgfhmajsZ2NUhvb8Qukewyf"
    },
    "metal:bandwidth_d2d": {
        "backend": "metal",
        "entry_point": "kernel_bandwidth_d2d",
        "entry_points": [
            "kernel_bandwidth_d2d"
        ],
        "filename": "bandwidth_d2d.metal",
        "format": "msl_source",
        "sha256": "a15f864d3c56794452f9b1f0acaa62e85d23e931f0cfb265314bd71f2878b325",
        "size_bytes": 728,
        "workload": "bandwidth_d2d",
        "zlib_b64": "eNp1kFFLIzEUhd/zKy4IS6a21FaRZeoKCvMg6CKuL8tQQiZJp8FpMuRmql3pfzeTTkfr6jxlTr6cc+8ZDwgMQDiLOCrrBp6UM6pKoeBGPmvpl0xOJdCV8rxKWjQz3m2gttp4iknaPWAHfMtdXd+kQKVaa6GgCfTZACT6IXSSsAb9/gKdGO4UboJYWVP+ANOs2FqJsxh75cQSxpC9eGVQWwNlw51MgVcV3LXDdb4IVFheKRRKwmT6c1ToLgXQO8VX2pQhqd5E1+u/j9kfdp89sOw2u8t+P6ZwOgU6OR8VG9+NDeGVhGM4FJ+d9qr1GBNypI2oGqngItbE0MtKF5ekwTbM8JXCmoeV4+2MkF1lsLZaflkfJRC+/6qDPC+axUI5epLM58OP0Ocy39FJj35X7zs77dnWCcowXp77Zbs/qy1qH3pn2rDSaXlIBoGh/qd6HlmtXAcm8BrRhXVAYzho+QK/doPQEJPMonLRz7T7P+6RvX+y94q7o88DNg9OYeV4nMXLLdmSNwQQ7aY="
    },
    "metal:compute_bf16": {
        "backend": "metal",
        "entry_point": "kernel_compute_bf16",
        "entry_points": [
            "kernel_compute_bf16"
        ],
        "filename": "compute_bf16.metal",
        "format": "msl_source",
        "sha256": "37f24ebfd7251e2dcef21e4559f807d66343b1dc13496ce90f1c9e26f2fd0297",
        "size_bytes": 3823,
        "workload": "compute_bf16",
        "zlib_b64": "eNqdl2tP4zgUhr/3V5wdJJS0UHJpUtQCK3amo0WCAQGzXxCK0sbtRJM6XcdhQTv89z3Hacm92jYfLDt5/Po9J74kJ90OdGEm4iQ5XqxS+MkEZ9EIZvFylUrmTeemC9qSST/SiZxwKd5gFYdcaok+WvNeESfs8o+rEWgBewlnDOZR7MsuxKn0pun8CLV5In0uIUWVQ+Dp0mMRWzIuk9rDUDLhyxBvFh4pwUNIGAs8v+X+VNm9FLMfcAKTV8l4giqwSH0RjMDzbiaPl9feX5P7h6vbb54HF+dgmwZoHEd7YXDzcA1TpWcpodu7B+9ucu89/nk/ufyiqlePk/sRnOJDTFEXBnAOjmmBdrrpCP5slsArPV6m0bEfBNSyIPI5y2pfr2/vEtI/6XQOQj6L0oDBmcq2l8ggCqcXnTQJ+QK4v2TJysdsqqdj4ufwW8DmIWeBVotHh1+/oH4bzihKvXPAhIgFfPr+7eH73d3t/ePkywhuSLn85gX7Ow0Fus2e2X2z16SaJU+HOWquE5jl4BOOxINw3ulkEwVe4jBomjRaB/BqnDDw9ITlnAnN0J+fjxS4ZQrluNmG55Mqh606XJpmOWlvJ6c5OfggaVxYYOhPT/KHYH7greIkJAteyL2FCIPnZx3+VSy+V41QTGoxrs1jugSTqeBj1X7v5F7WWQcfXZyvG9p86WtG33Jw5a1DOSbDsdCypn4ERt+Z6/q4LjRdbVOalpWmSslqk+JFqWMaEldN1qJhqFOtF66hjz6WhmEdUWwN8hZMKaUFmDSPVAAtOGeLIs4VzhWu+A0pjCKWGTb6JsYJPYzXcK1CJOrN4e1Uh0MwnVTHnOR9Tof/s9Pa8ocFc3cL5j4WzHYL1u4WrH0sWO0W7N0t2PtYsNstDHa3MNjHwqDdgrO7BWcfC067BXd3C+4+Ftx2C8PdLQz3sTAsWVAeTrrq8yCI8VzksYT4hQnsEgDtj9khmKmPy4ehtfkMgCj+h4kEAjxaZzJ6Axlvdsl4lfTpc4DGISFNnRyhxGiNdEyVs8LxNYZeL5TFk+FgJfzF0oeUiziKPm7nUis/FGsxVT2D05RkqFEUUscM7XxYdHEX7mUb7LgM0L6ERTtAuwYW7QCtaSzaAVpxWLQDtB6waAdotgp3C0BzCYsSsDUReHRsT0QdqCSiDlQSUQcqiagDlUTUgUoi6kAlESXgvf61kU3ZhJKTrRlh9F917PrReiudvWvezHmzxJvNvJXzVom3mnk75+0Sbzfzg5wflPhBM+/kvFPinWbezXm3xLvN/DDnhyV+qPjix0wcRxD/RD5M8EcANwYtwc/ww8NC26y0rUrbpnZpGmyunBlU+jiVtltpD5vCSpfoE/1hPGQKfyTwzWLdzuoDqjtZ3aX6RmT9G/CE+/EzKWC8pcHSpa7D70p/BH7iybcVO1NjXmjG6/DrZ4OuFNXeO/8BdqInFg=="
    },
    "metal:compute_fp16": {
        "backend": "metal",
        "entry_point": "kernel_compute_fp16",
        "entry_points": [
            "kernel_compute_fp16"
        ],
        "filename": "compute_fp16.metal",
        "format": "msl_source",
        "sha256": "4fd128336f09815c01be60437e9564bf0a803361d01e619dcc72ea76f74434ed",
        "size_bytes": 3489,
        "workload": "compute_fp16",
        "zlib_b64": "eNqVl29vmzoUxt/zKY40qYI2bcH8q5KuU3dvqk2606revYsiZBKTooFBNq5aTf3u13YghADbhRfVMf75ec45WKft9bkB57BhBeeXu1LAT8IoyeawKfJSVCRKSicAMycVzixFLmnF3qAsUlqZ3JrXfHSMK+z+89c5mFvykm4IJFmBq3MoRBXFIplJbcorTCsQUuUMqMgjkpGc0Ir3NtOKMFyl8uXRlhY8A07INsIj72Od7j3bPMM1LF8rQrlUgZ3AbDsHnGXwTRUF+xw5mFTavBB4xlmC4OHbvT7//fHf6HH5FP348rS8/1uHX38sn+ZwIzdlZ87Bg4/gOwjMm/oo3mw4vKpNKaIiBBmmZB89/PP9kSvla8P4kNJNJrYEbnV7I15tszS+MwRP6Q4ozgkvsWyf3l0Yxr7V8FKk26G2mwbIZ7DlsFrJnwlhpm2t1zMN/uYjtLgzhrefpYVRH+58qJZ0f0/GLekdSOULO1n6alU9M4K3UVnwVKUQpTTasXS7XlvwS7NpAqZC7z526mq21cNIJRhd6PW70eaiPyLg6Fl+VhWaSY5N+wr58t7WZVyqZAtm7pfWDOwrP7GsxalIXI6rxF2VWKugYRnaylwqK3npdLdMZaAOnJyQV7DmkSkLmalqerIIYtXAA6i0ZjrlQZSSXYtSjVKNanZPMfuA6GTtK0dWBBeyMjtAR3nrbyNfCwvOwPGFJatvTtyE//NInWZt7UyzdqZbO2PWaJo1mm6NxqzdadbudGt3zNqbZu1Nt/bGrP1p1v50a3/MOphmHUy3Dsasw2nW4XTrsGOtvZOCgaknb1pJf1ssVHB7NP4XcHGRVseT9UPJ8C7HICgrsuzwupUqccpqMR3ewo1QMmpxLKTHtJopangyW86x2X5o1Y05ME7DOOMMahg0zrgN444zXsN444zfMP44EzRMMM6EDRN2mD80SI7qPzZogOk1aIDpNWiA6TVogOk1aIDpNWiA6TWoy7z3f7vrCw9cN03ffWZfvVry8h9Wb53ffzXvtLzT4Z1hHrU86vBomHdb3u3w7jDvtbzX4b1h3m95v8P7w3zQ8kGHD4b5sOXDDh9q/uhAXBQZFD8ln/IkpXKQmNyWc+fsaO2crNHJ2lXrzk1onpbxTs74J+vgZB0OlSVymafMT9ajkrqQMVKxu489Ffv7OFBxI1L/2b2Sc3WtFGS9HTORWxZ80vryXxEeVW8ludWed6b9Gj78ZatHSLV34z8ETLmf"
    },
    "metal:compute_fp32": {
        "backend": "metal",
        "entry_point": "kernel_compute_fp32",
        "entry_points": [
            "kernel_compute_fp32"
        ],
        "filename": "compute_fp32.metal",
        "format": "msl_source",
        "sha256": "9622433bacbe7fda9fc97c9307a117a6fb0f0c1919a4252843de9149e7b8fe6d",
        "size_bytes": 2451,
        "workload": "compute_fp32",
        "zlib_b64": "eNqNlmtvmzAUhr/zK440qYJegQRSJVmnbku1Sasadf0WRchJnMwaGORL1Wnqf9+xgeYCdOFDdA5+3tfHB9vt1akDp7AUuZQXm0LDbyo4TYewzLNCK5qsi14IbkYVST1DTrgSf6DIGVeu9IYVn+ziBrv9/H0I7oo+syWFdZoTdQq5VslCr8/Rm0tFuAKNLifAdZbQlGaUK9kYZIoKohi+3BmyhicgKV0lpOP9wpZ7K5a/4AomL4pyiS6w0USshkDSFO7NoqCsUYLLcZpnCndTXPDd/a2VP0x/JtPJY/L07XFy+9WG358mj0O4xsEgxp8QPkIYxeBeA1kudaZTonIh4QURdIGCMJtZzyr48TCVxv7KcT4wvkz1isLYtjiRapWyxY2jJeMb4CSjsiDYQjs6cpyy3fCcs1Vb610H8GltO8xm+LumwvW9+fzcgu98iC0edOHbT7OFwya897G2ZO99crEl+2+kmRc2uPTZTP0SlKySIpfMlJAwnmwEW83nHvy1LFuDa9Cbj3vrqofNI6jSgo9s/upsaykLAYKfdp0R178MI9y11QIuzGgu3DL1zsG/jNbeqKFemNpaHBb7DgvrEHZYcLpBiwszBe41azkqCy0J4eOwfxmgHs4w8OPQknbUtevH19qDEwgiXU1RSYMjpEG7NDxCGrZLe0dIe+3S/hHSfrs0OkIatUvjI6Rxu3RwhHSwIy21uQDXbnSmjF6PTDDeOW0jODtjancjfygE2WQENBd5mr693lqZW6gys+EYrrWxMcmukT0VfrVrhX8OeLvaXVct7I0JaiboZsKaCbuZXs30upl+zfS7mahmom4mrpm4mxnUzGCP+U+D8Jj+t0EtTKNBLUyjQS1Mo0EtTKNBLUyjQS1Mo0H7zGvzMl3keQr5b9QxuWYc9zE2Djf9yU4eHOThQd4z+V4h9bNl+gea6CCPD/JBy40rdYZ1Yn14Mk1RZxiHJu6Vcd/EURnHJq5Nqj+yMzzUc+OA692dDH09Dz5Zf/znQybqT0HHds4b138Z3H3xzWMugVfnH3MUnJQ="
    },
    "metal:compute_fp4": {
        "backend": "metal",
        "entry_point": "kernel_compute_fp4",
        "entry_points": [
            "kernel_compute_fp4"
        ],
        "filename": "compute_fp4.metal",
        "format": "msl_source",
        "sha256": "f0865f0c729e73afe1699f8e50da962c71a7ebe51ec60a88338663504d85398e",
        "size_bytes": 450,
        "workload": "compute_fp4",
        "zlib_b64": "eNp1UcFqwkAQvecrHhYkSos99JSeLAYqtDVEPYd1M4mLm92wO2u1X99N6EEo3oY3b968ebOYJ5hDOuv9U9sHnMgZ0hmk7frAVDX9C9KOWOjZQMwNuyt6qwynfpb90asb9sBavq0zpDWdlSQ02gqewwauDqF5jMrGszCMEEWmMKGrSFNHhv2/pmJyglUEb1qj4BSeqK7EHfwwul06ecQC+YXJ+KiCNghXZ9h/bfdFsSl3+QoR/RzOw/YoamVafAgTaS0NAptiWxV5We3ey3y5Gsv1Li8zPCOlS6+VVKyvaAVT/YofchaNOBGoC3r0PdhYJMkDOWcdJjeLsztroYU8eZg4fiYco99v4WKK8Q9/KUPFg12QYy6T5Bcu+5iY"
    },
    "metal:compute_fp64": {
        "backend": "metal",
        "entry_point": "kernel_compute_fp64",
        "entry_points": [
            "kernel_compute_fp64"
        ],
        "filename": "compute_fp64.metal",
        "format": "msl_source",
        "sha256": "0da0f879b649df1c27a0fb5c543aa40f70e0e35a410538d737e6b3fceb08d833",
        "size_bytes": 454,
        "workload": "compute_fp64",
        "zlib_b64": "eNp1kUFrwkAQhe/5FQ8LEqXFHkoP6clioEJbQ9TzsiZjDG52l91Z0f76brQFQbwN8x7fvMdMxgnGqJzx/qmxAXtymlSGynQ2MImtfX1B2hFLNeqduWZ3gjWt5tSPsj+/uLb3tun7PENa06GtCLUJG0VjmMBiE7aPEa49S80IETOEDp0gRR1p9jdiy+Qkt3F5JV2IQ3iiWsh7wuaceOqqHSbIj0zaRw6aIF2dYf29XBfFolzlM8TtV18Ry52sW93gU+poa6gHLIqlKPJSrD7KfDo7j/NVXmZ4RkpHq9qqZXVCI5nqN/yQM9jKPYG6oM7J+xiTJHkg54zD4Opwduds7EEe2jB8sNY4ho6kA+HyjkvL0f+TBskvV3GY3Q=="
    },
    "metal:compute_fp8": {
        "backend": "metal",
        "entry_point": "kernel_compute_fp8",
        "entry_points": [
            "kernel_compute_fp8"
        ],
        "filename": "compute_fp8.metal",
        "format": "msl_source",
        "sha256": "744a7b0e8c3c3b5993d8ef9dd79d15645587a8f0cd888aac11b0726327d19a33",
        "size_bytes": 450,
        "workload": "compute_fp8",
        "zlib_b64": "eNp1UcFqwkAQvecrHhYkSos9lvRkMVChrSHqOaybSVzc7IbdWav9+m5CD0LxNrx58+bNm8U8wRzSWe+f2j7gRM6QziBt1wemqulfkHbEQs8GYm7YXdFbZTj1s+yPXt2wB9bybZ0hremsJKHRVvAcNnB1CM1jVDaehWGEKDKFCV1Fmjoy7P81FZMTrCJ40xoFp/BEdSXu4IfR7dLJIxbIL0zGRxW0Qbg6w/5ruy+KTbnLV4jo53AetkdRK9PiQ5hIa2kQ2BTbqsjLavde5svVWK53eZnhGSldeq2kYn1FK5jqV/yQs2jEiUBd0KPvwcYiSR7IOeswuVmc3VkLLeTJw8TxM+EY/X4LF1OMf/hLGSoe7IIcc5kkvzujmKQ="
    },
    "metal:compute_int16": {
        "backend": "metal",
        "entry_point": "kernel_compute_int16",
        "entry_points": [
            "kernel_compute_int16"
        ],
        "filename": "compute_int16.metal",
        "format": "msl_source",
        "sha256": "7be9012a774311bbb9ccab9335ef1157097c16db4e26728522fc6eb9374a6f1c",
        "size_bytes": 2060,
        "workload": "compute_int16",
        "zlib_b64": "eNqVlW1v2jAQx9/nU5xUqUqgFc/QAa3EVKb1RbWq6ztEI5MYai1xIj9UVBvffWeTNmah0ciL6HL3v1/O54vTanjQgEhkUl5ucg2/qOA0GUOUpblWNGRcdYbgp1SRJDDSOVfiDfIMA74MxkVCeKA3utnXuzH4MX1lEQWN7gZkWoUrvb5AOJeKcGX958B1GtKEppQrWQkyRQVRDJ2VkKQ0Dslx98oWOxPRC7RgvlWUS2TARhMRj4EkCdybJcG+QAk+x5e8UugML1cMSfIlEwruZ7eW8+PhZ/gwfwyfvj/OZ7fWvHuaP47hCoPYnwZ04Rq6A+zUFZAo0qlOiMqEhK0Jay6yJKGx4RlXF7JcGnDL884YjxIdU5jaHodSxQlb3XhaMr4BTlIqc4IttNGJ5+37Da8Zi4/23vcAr2N9h8UC72sq/HawXF5YXc1OlPLOZ/Jyb0px9zPxfrdKYa9WuCqF/Q+hicMGF75YqBdBSRzmmWSmAFx7uBEsXi4D+G21bA2+kd5cH6zqPWwuQZUWfGKfd15Zyvvu4ybiru4ffL+o/xza22946QD+QEcHk2oeieN/81ZH8vZL2qeItpNh626iCOeqvf0y740w6bloYOCQJgeIThXRPRHRrSJ6JyJ6VUT/RES/ihiciBhUEcMTEcMqYnQiYlRFXP0HwjLWmQDfDjweR9fQ1hNjTJ1vbgLNJlPuQJ/lgmxSUpw3H+4SJRXNC5g1p3g6acMxTy7Jfh7OTKLdsN9D00x3scwPoTN5aNcInflCu0boTBHaNUJnVtCuEToTgXaN0Nl3MawTOruL9nHh7tPDBf8SZX7Z5WfsZoCYskvo6aHn4N3Vq+wB6gcuYWg8o6AoqvgXLHAal+b95jeOpWB05/0F4gxWPA=="
    },
    "metal:compute_int32": {
        "backend": "metal",
        "entry_point": "kernel_compute_int32",
        "entry_points": [
            "kernel_compute_int32"
        ],
        "filename": "compute_int32.metal",
        "format": "msl_source",
        "sha256": "765e0c058c11c2cc1a7c3b7cd4f4922f21c4c8a0542dd9361f38157ec62073d7",
        "size_bytes": 1710,
        "workload": "compute_int32",
        "zlib_b64": "eNqNVNtO20AQffdXjIRU2QmIxM6NGJCCiFQeUBHlLQrWxt6EVe21tRdE1fLvnV0bNilxFD+MxjPnjGf2zPq840EHUlFKebapNPyigtN8CmlZVFrRhHEVheAXVJE8MNA5V+I3VCUmfBlMG0Kygze42c3dFPyMvrKUgsZwB0qtkpVen2JxLhXhysa/AddFQnNaUK7klyRTVBDFMPglJSnNErI/vLLNzkT6Aucwf1OUS6wBG01ENgWS53BvRoK6QQk+x4+8UojCsxWrK8H97NZW+fHwM3mYPyZP3x/ns1vr3j3NH6cwwWR/hCaEKwiHI/AnQNJUFzonqhQS3kxac1HmOc1MPRMKoaykKXzueSeMp7nOKFzaE06kynK2uva0ZHwDnBRUVgQP0GZjz6tPG15Llu09ed8DfPadOiwWaNdU+L1guTy1uAM6OHi/De6UceCwDVxr5YDRQeDKAQefQKvJBgdfLNSLoCRLqlIy0wDOnmwEy5bLAP5YLFuDb6DXVztTfaTNI6jSgsf2/d1zrdTfQQlR06brv9DX8f8IkmUfiFWDcG2KHuZsB13M4Ib03i7m0Xh8cXOhA3huCsdbhL4jhEcRQkeIjiJEjjA4ijBwhOFRhKEjjI4ijBxhfBRh7AiTAwTLWJcCfEvDC30FPR0b53Jrb2PodpnaXoqTSpBNQZo7+xl2paSiVVPMupd4w7WpY962K9kVM1uApmPXqWtWJt4FGNXRtAOMymjaAUZVNO0AoyKadoBRDU07wKgkRgcARhU0ewHv2xes+RctUMGlURLP5hnnD5Dk45ToR7U/MP6w9kfGHwex9+79A+cp5UE="
    },
    "metal:compute_int4": {
        "backend": "metal",
        "entry_point": "kernel_compute_int4",
        "entry_points": [
            "kernel_compute_int4"
        ],
        "filename": "compute_int4.metal",
        "format": "msl_source",
        "sha256": "cba7ce7018834cdc953d8f27d031ce52eb27f9017e19e91c414bd9359f1ef1ad",
        "size_bytes": 453,
        "workload": "compute_int4",
        "zlib_b64": "eNp1kcFuwjAQRO/5ihGVqoBa0UNP6YmKSEVqSxTgbBlnSSwcJ7LXFPr1daIekGhv9tvx7Kx3Pkswg3Kd9491H3AkZ8lkUF3bByahLT8jbYmlmQ7K3LK7oO8iT/00+9WLa/kgW7yuMqQVnbQihEhn6AKLfTg8RGvrWVoe+T1saAUZasmyvylqJidZR3hT8kSVkH/j/Zh14VSDOfIzk/XRA3WQrsqw+9zsimJdbvMlIv0YhsOmkZW2Nd6ljbKaBoN1sRFFXortW5kvluNxtc3LDE9I6dwbrTSbC2rJVL3gm1yHgzwSqA1mTD3EmCfJHTnXOUyuGmf/tIWR6uhh4/MToYl5v6QjjGuQTnMTV6FVvHt2QY0/M0l+AFz8miI="
    },
    "metal:compute_int64": {
        "backend": "metal",
        "entry_point": "kernel_compute_int64",
        "entry_points": [
            "kernel_compute_int64"
        ],
        "filename": "compute_int64.metal",
        "format": "msl_source",
        "sha256": "76a65dd49d2c1276ccd2c62c27c0b286cd2c9e3bb6fea1491c5acc17be0e0bcc",
        "size_bytes": 1843,
        "workload": "compute_int64",
        "zlib_b64": "eNqVlNtO20AQhu/9FCMhITuBkjiODTEghZKqSKAiCldRsHzYBKs+aQ8RVcu7d2ZjsBExanwxmt35/vHuzuwe9QzoQcxLIQ5XlYJfjBcsm0Bc5pWSLEgL6Tpg5kyGmUXorJD8N1QlBkxhTWpB8I4nbnpxNQEzYes0ZqCyslj1oFQyiNTyALMXQoaFBIX8PhQqD1jGclZI8SGYSsZDmeJkO0QJ90EwlgRhx3yk1zvl8RMcwexZskJgFlipkCcTuKEdgf1l1IcC068ZuM5hlNYp4GZ6Seoftz+D29ldcP/9bja91O7V/exuAscYHLpobDgDe+yCeQxhHKtcZaEsuYBnCquCl1nGEkpHUzaUlaBlHRnGXlrEmUoYnOrDDYRMsjQ6N5RI8f9FmDNRhXh2OuobxuagYV2mydZDNw3Ab+uBw3yOdsm4ObAWiwMNflKCBh924U1RGtjeArfL1JCjz8moIZ03kv4LK9z7fC6fOAuToCpFSkvA7QcrniaLhQV/NJsuwST0/Ozdvl7D9HEmFS98PX4xmrXU5ccyYl3rdf+F4cO1/4EJk+SVibqYKBQMIT2gJVn+5l+bKB9gzNRMn/TYToPnk9nI804uTrxvztT7Ohw/XFvwWC/Fb4uHLbG9q9huiUe7ikctsbOr2GmJx7uKxy2xu6vYbYm9XcVeS3z8X2KtXpYcTN27+LScwUD55Jy2LpAP/X4q2725V/FwlYf16/E23aQSklV1Mu2e4lujKA+N2pl0p1OToenppu5T2/rvAWokNN0ANQuaboAaAk03QEVH0w1QYdF0A1Q87n4CUIHQbAVe2ve8fhTneB0XVFQ8m0fcv4UiE3eJ/mjjO+SPN75LvofX98X4BxusCD0="
    },
    "metal:compute_int8": {
        "backend": "metal",
        "entry_point": "kernel_compute_int8",
        "entry_points": [
            "kernel_compute_int8"
        ],
        "filename": "compute_int8.metal",
        "format": "msl_source",
        "sha256": "f91c70296da7a4bf835afee1f607249ea532863c34a52891d546fa862a5466c5",
        "size_bytes": 2457,
        "workload": "compute_int8",
        "zlib_b64": "eNqVldtu4jAQhu/zFCOtVCVAVQ4pIKBIrEq1vai26vauopGTGGo1OJEPqNVu333HDod0k7AFCTPY30w8M7/ji4YDDYhEKuX5KtPwSgWnyQiidJ1pRQPG1RDcNVUk8Qw550q8Q5bivCu90ZYPirjBZt9vR+DGdMMiChpnG5BqFYR62cLQXCrClZ0/A67XAU3omnIlS4tMUUEUw8nSkqQ0Dkj1dGj3OhPRC1zA/E1RLjEGrDQR8QhIksCdyQjyDUpwOT5kgzuNXojwYXgeMgUbGqlUwN3s2kb7ef8ruJ8/BI8/Huaza2vePs4fRjDERR+/nT4OXbiCTrvrgzvcRSNRJOENkYRwaiwEMaixupBm0kS/cJxvjEeJjilMbLUDqeKEhVNHS8ZXwMmayoxgNe3q2HEwacUiYDxhfL9zxpkK9PDNd001YMXilq0LvG5/86p58NsB/Nja5QshkRT3jh7QBPcVh472MKHeQI8tK6jSgm8f5Nop87H/Xde1/k1oa1OJAbo+g5s/DKZTaHsenEH77eZGe60630617/Arvt1q307/K869aueuX3C2vt7Y+XCcXPOwSbFWFfrPa1OlfXh6wnFJhdv2FovWoQXVp+GAd+rww/k4wN06eJvZHuwdBcMD6O/BnapwTb0ISuIgSyUzG8DUg5Vg8WKxExdbgmvQ6dWnrHbLB03l+sLCFiSZy3mtE9RkjeS26ewaBH+MYss9LgrpBBiVcwKNUvmH3umllBOJ4//kFH7tweEpOYUn5RQez8nZO/gg2pjN4bVj3zht3dq9aMaf2E6Z7dSx3TLbrWN7ZbZXx/pl1q9jL8vsZR3bL7P9OnZQZgdF1sJLvHzy1zheRldY1LExJoUzP4Zmk6nigfqWCbJaE9BcpEmynz6Ekopm22DWnKAotIlj/hUj2eNpeotDwx7EppHu+DNgGopDPWC6iEM9YFqHQz1g+iX8I4BpEg71gOmM6B8BTDtwqAQ+al9OeK+jn4v1ecYaeObaxEzR7uW2b+zL3O4be7BVwfY2eMK+LzACkYF6z+jE9GfqYlR7y/wFk0OvPw=="
    },
    "metal:compute_tensor_bf16": {
        "backend": "metal",
        "entry_point": "kernel_compute_tensor_bf16",
        "entry_points": [
            "kernel_compute_tensor_bf16"
        ],
        "filename": "compute_tensor_bf16.metal",
        "format": "msl_source",
        "sha256": "4af93d058b394db393a8f5c0883ba0889066d55d11c5489f569a0e6c81bc3ff2",
        "size_bytes": 3217,
        "workload": "compute_tensor_bf16",
        "zlib_b64": "eNrFVm1v2zYQ/q5fcWuAQIqTWJay1nCcDN7qYgGaJkjSfTEMgpIoh6jexpfCwZr/viMlW3bseo0/rIJtUdTzPHc83x3ZPXLgCGJRSnkyqzR8YaJg2QDiMq+0YkSxQpaCRGnvLbg5UzTzDGFcKPEEVckL5Upv0NDIFpZBj36/GoCbsK88ZpBmJVVHUGpFIp0eo6VCKloo0Ch2CIXOCctYzgolN15yxQRVHCdXXlnBQ5CMJYR+Zz6yXo9E/AhdGM+Nf6gCM01FMgBCrscPo4/kr/Hd/dXNJ0Lg8gLCng/u0C6ZSJ4nM1HqiuRUCT6/hHYmsnb68741cXN7T27Hd+Thz7vx6L0dXj2M7wZwhi/7+HWDZlB/uxAGHlxAzw/OwA0DQKF5H66vRxLm+A4+fLy5ld2MFszodx3ngBdxphMGC99UkvHocnP+pc+OlryYQUFzJiuK/4TFnRvFFH5JWMoLlrgbsfDg2zfYnIahiZDnHDAhSgFvPn+6/3x7e3P3MH4/gGujvDWHBPtbc8FkAwlPe51t4nX8vTcoXyQ8dZw6weBryZMdyeY6gNfWRIPJBH9TJlzfm06PLXBH6rXw3vfgbTK24GATvJaeLTLcjYxa5NkSaezCDCMwmahHwWhCqlJy4wLhBZkJnqwhZQ1tE4EXCZsbaM22ky8YWau9RC8FplMP/ml9rh2GiEqGGeyf/ppCB28B3jHPmwWfGFQpmkfPg44V2HKtM6N1ZuR5mKmt6S31BxS9yOkXRlKeZch5WQHDGnoMffxcuvWT2zv1UyO+Wzsyod5L38TlR/QLNttL/2RhwFpoOUvp2P8v3XVZ/Dsbd7eJ9V4ttkiKHaLBfqI7NcP9NN9ZUauaYl9zbV1wZRJcn5vBcKXwz6HT4WpRFOY6qASd5RR0IcosW063UhXlohGzwyGcaSNjHlaF1peV60zxKnsiNI41PlDF3Ng/BtzvbGbituc3sfhBcm+N3HsdOVgjB68jh2vkcBHtVy4ay2X/RdfkPRddk/dcdE0OV8jPTv1rbyuNuemuMsZcix8noTk3vD2bnm/HHYHimWnDDR5z2bX9/xA3Uu1Zrl7ktRU2RwoidW5bt5++LCSpSlGH2whjjWyUWkSF4Lg/5SzHsqMzORiY4YprDWVpqXNhxSa4zUzRwcUYh2GgpxsdrHGh9/NdCH6+C+H/7gKeCl2TQcO1g9Fqj2qOVhNETTGNuMQjJDZHd2HHg99amwOgkqinitWN99L15+8+/OGbSzf+PTvPzr8ZBQK/"
    },
    "metal:compute_tensor_fp16": {
        "backend": "metal",
        "entry_point": "kernel_compute_tensor_fp16",
        "entry_points": [
            "kernel_compute_tensor_fp16"
        ],
        "filename": "compute_tensor_fp16.metal",
        "format": "msl_source",
        "sha256": "ea7d77d69496c3d34f67fcd5b17df8015719b1df98286cce44c0ab8b32d3dd41",
        "size_bytes": 3095,
        "workload": "compute_tensor_fp16",
        "zlib_b64": "eNrNVm1v2joY/Z5f8UiTqqS0JSTcFVHWiW1Md9KtVnX7hpDlJE6w5rzILxPV1P9+bScQAhQNvmwRBMc55/g8jx/b9C8duISYl0JcZ5WCH4QXhI0hLvNKSYIkKUTJUVoN3oKbE4mZZwizQvJnqEpaSFd444aGDrAMevrhyxjchPykMYGUlVheQqkkilR6pUcqhMSFBKXFLqBQOSKM5KSQYu8llYRjSXXn1isreAGCkAThV/oj63rK4yX0YbYy/rQKZArzZAwPJi6YVhUjtz1wJzZOJGieZLxUFcqx5HR1D23PErN0tBpZ1a+P39Dj7Al9//dpNv1km1++z57GMNQvR/rrBk2j/vYhDDx4BwM/GIIbBqCFViN4eJgKWOl38Pm/r4+iz3BBjH7fcd7QImYqIbB2JhNGo/v9/l3HjhK0yKDAOREV1sm3uDvHqecLfpY0OTJ3rgP6OjhvMJ/re0q463uLxZUFHpnJFj54Dd7ObQsO9sGd2W6R4XFk1CKHG6QZFzKdgflcLjnBCapKQY0FRAuUcZp0kKKGtkmmRUJWBlqzbecOg7XaG/RGYLHw4FfrGUxRQYQF0cVh2q5/808KPfBvAv2r66gJ+tqEVvLm0fOgZ0VeubrsqMuOPM/T5dB62CtxwNpNjn8QlFLGNGW3yCYGeAUj/bl3re3Bjb80ssdUI5Prk5V1Qn5DuSDZycrXa2mrva8a+6dImklsbB6QGpwqpWugCT9o4z8gHJwvfFQ3PF/3diuracnBtQuDSq3oqzvTmGyt/Dvo9ahcrwpzvak4znIMquAlY5vuVqrClDditjmBoTIy5mFbqBtarpikFXtGOI6VfsCSuLF/Bfr8sJWpjxG/ycdvkgcd8uA0ctAhB6eRww45XGf7xKD1ojk/6Jp8ZtA1+cyga3K4RX5x6rv92dqZ6+1VxLrU4uU8NGfy2+Hi7iDsEiRlZhtu4LqeXbv9X0A4UJ6lqnVVm2MGzGmNhMpNKd746e5SErLkdbKNsF4ne4stwpxTfTzlJEcpw5kYj01zy1lD2YzUe1eP7RrJuT5r9HHS2+nSHWGgFvv7WmNo8LcZCv42Q+EfNkRTcE3lTTr/p7Z3tuYf2VyjFrr8qEhpobdUdz2qB+9bB2PAAsnnikzsqPeuv7r9/NE3l2rcvjgvzv9ozdwi"
    },
    "metal:compute_tensor_fp32": {
        "backend": "metal",
        "entry_point": "kernel_compute_tensor_fp32",
        "entry_points": [
            "kernel_compute_tensor_fp32"
        ],
        "filename": "compute_tensor_fp32.metal",
        "format": "msl_source",
        "sha256": "e183baef7cc129cd672c5f3578e19ddb97d672b0b022befaf2dbc3247086536c",
        "size_bytes": 3093,
        "workload": "compute_tensor_fp32",
        "zlib_b64": "eNrFVttu2zgQfddXDFAgkOwkliW3MRw3C++uixbYoEHaN8MQKIlSieoGXgIHRf59h5RkWbHrrv2wFWyJIs85nCGHMxoNLBhAxEshrtJKwXfKC5rNICrzSkkaSFqIkgdJ5Xtg51SSzNGEZSH5M1QlK6QtnFlDCw6wNHrx56cZ2DF9YhGFJCuJHECpZBCq5BJnKoQkhQSFYhdQqDygGc1pIcXeIJOUE8mwc2fICF6AoDQOyE/6Q2P1gkffYATLjbYPVSBVhMczuNd+waKqMnozBHtu/AwEy+OUl6oKciI529xB12O0p5upkf388CV4WD4GXz8+Lhd/m+anr8vHGUxwcIp/22sa9X8EvufAexi73gRsXFgU2kzh/n4hYINj8OGfzw9ilJGCav2RZb1hRZSpmEJrmowzFt7t97822VKCFSkUJKeiIrj6BndrWfWGwVPJ4iObZ1uA18GNg9UK7wnltuus15cGeGQrO/j4Z/Buczuwtw/ubXeH9I8jww452SL1vJDiCqxW8hunJA6qUjBtQsCKIOUs7iFFDe0WmRUx3WhozTadrxhZp71FbwXWawd+dDbXBkNIBMXocK/fJjDEh4dPjKHG4SuNKnnz6jgwNAIHrj4z7DNDx8Eo6KbeD20gaEROvtMgYVmGlNfBNTfIS5ji784eX7sJSh5VDPX6nqSqF+HXqgVNT1K9amSN7gHFyD1JTm9YY+QhsfHJYu22HxH1zhM9qumfp3nTrWVScrBN5DOpQ1jd6sZ852jfwnDIZBv2+npTcZLmBFTByyzbdndSFWG8ETPNOUyUltEvu0J9t3KVSVZlzwGJIoUvRFI7ci8BK4QJQywUbrMW/5E87pHHp5G9Htk7jez3yH672ic6jafkfKdr8plO1+Qzna7J/g75xarv5jEawAPlV1sxEBGGGtZ5G9uy1OV0OybaoltnZDHaydvOta60WnGns8nIjeTK13X83WR9exg3AMkynbpbE/BbwtSMC/DHyjFc1Z4UI6xLfCBUbtK9m7w+mkKWvN5ALYynbu/whoRzhjUtpzkeZJKK2Uw3d/2qKduZhu+N2ApL0xoNbNvY9D213suJjQnj32+C9/tN8P93E1gCto6gee9jajfrNZ9jK0StMYyYSFiB6dZu53Hgj27OGRARyOeK1qkc6+vm5sNfrr5UY9+L9WL9C7CI3dg="
    },
    "metal:compute_tensor_int8": {
        "backend": "metal",
        "entry_point": "kernel_compute_tensor_int8",
        "entry_points": [
            "kernel_compute_tensor_int8"
        ],
        "filename": "compute_tensor_int8.metal",
        "format": "msl_source",
        "sha256": "ff70c5e909deae5abd1dc3cca7436bb73ded5e223b91fe222262863e70e265b8",
        "size_bytes": 472,
        "workload": "compute_tensor_int8",
        "zlib_b64": "eNp1kU1rAjEQhu/7K4YWyioVeyzbk8WFemhdVj2HmB23wXwxmYj21ze7eChob+GZeT+GzKcFTEGRj3HWhwRHJIemAuVtSIyC0UVPQjt+hdIiSzMZBLVjukDwmZdxUl1l4o5q2F68ryooOzxphZAynYJPLPbp8JyDXGTpeORP4JIVaNCi43gz1IwkWWd4M4qInZD38X6svCD1DXOoz0O57AF9ktRVsPva7Jpm3W7rJWT6OdwIUduuJ5+CsJJJnweDdbMRTd2K7UdbL5bjc7Wt2wpeoMRzMFppNhfoJWP3Bj9IHg7yiIA2mbH1UGNeFI9I5Ake/gRX/8RC5zGC8wwxheCJwWWnE8L4H9edbM86mMtMKpXGLHwofgHS9aOe"
    },
    "metal:memory_pattern": {
        "backend": "metal",
        "entry_point": "kernel_memory_pattern_write",
        "entry_points": [
            "kernel_memory_pattern_write",
            "kernel_memory_pattern_verify"
        ],
        "filename": "memory_pattern.metal",
        "format": "msl_source",
        "sha256": "7e99f4b66ea59fc12c8c90c419f10b2120deeb03bd18287c2d0ba2ec30cdf6c9",
        "size_bytes": 2178,
        "workload": "memory_pattern",
        "zlib_b64": "eNrNlltv2jAUgN/zK840qUqAqUB6WYEiQZVJk9pp2rqHqeqsNDHUWmIj26GUqfvtO3ZCLgW2vlTaeQBkf+fuc8Rhy4EWRFIo9W6+yOAnlZwmA0hpKuQjWYRa4wm4KdVh4hk24Fo+wkIwrl3lDQoN0lQgD5Jp2tlzuaSSzR6Nscn0I1h0AODGdMkiChlabsFdNutAJLjSIdeQJYLPD4BnKXkQMlb1K8QPYGM6FTHd1lOUxt7GX+59UPqz8Gt57UDhJdQiZRHJ3VAphSSRyLjOw5LRPRxCsNKUKyY4zLNQxgMIkwSuTOULK8rA0+/XwVfyOfhCgsvgKvh0PYAjcHuwskH5fTCxwoLKvLIgZJEyhquU8XfoOG8Zj5IspjCynSVKxwm7GzuZYnwOPEypWoQYtr0dOg4mplkEjCeM5y0qkzfuXJsysHjVyS/ziuSntvrwywEUNsO3hHdwfg7dzANJdYbvq7vqFpINt7henftQyA6uX+eOC9nB+XVuUsgO7qjiehmMRjYzF3OEA/B73y69bZXjSuW3u1sJtayaLdMazivgh60Ufrn2ezwGv+950MYozwL/9PRselZEadTctSHXBuudINZC7P1xML2YnEz3YH6BXfSn/Ungb2pTBNywN3SeHCefXlgKFv9tzF1r5fn0ws0Nfs6odLve7W3HMnsnq4J7O+CtWavw/l7btoYl55ecrfucmTt9L2kYk4VQTOPUEcbJXLK4SeIBUWxNS14RnK0C3DzrGQ5ZNQOmqea3i268oT0ZVbnmB+2S2TjwNsaMYNg3iN2iqcac2Qlrrh07Xnknn17WtXwdNNr2bAn+r837xzKtVI5et98WTUSEm9O6V2DW2fB13oJZMOV7ePPCB1E3YKTdroc7LO+enOrTOGokNbY7ujJU1H1GdXRPwjgmdLVIWMS0W2tCp1GYzuavBAaLZZQ0CVeN9/oH/CW9hA=="
    },
    "metal:memory_sweep": {
        "backend": "metal",
        "entry_point": "kernel_memory_sweep_write",
        "entry_points": [
            "kernel_memory_sweep_write",
            "kernel_memory_sweep_verify"
        ],
        "filename": "memory_sweep.metal",
        "format": "msl_source",
        "sha256": "c5b22168a9ac0cb978e3b44431c79651e093bf38b094c77eff996e96e92c6fba",
        "size_bytes": 1718,
        "workload": "memory_sweep",
        "zlib_b64": "eNq9VWFv2jAQ/Z5fcdOkKQE6CKRQoENq2lSa1E7T1n2YqtZKEwPWEjuzHQhM/PfZToCwDm3S1PmDBefnd3fvnpV2w4IGRJwJcTLLcviGOcXJCFKcMr5CYolxBnaKZZg4GhlQyVeQMUKlLZxRhUd1OFpyInHrt0cLzMl0pYku/PdggCMAO8YLEmHIE0ZnDXjKpy2IGBUypLIMvgGap+j7kvFYPD8TGMfOlrRMMdqRGvA/UbegogolS0mEctV8AzDnjKOI5VSWuXk0hzYEhcRUEEZhloc8HsGt1g66b3tNsGkoyQJD3zt5IlUKc9f/ehd8Rh+DTyi4CW6DD3cjOAPbhQJ0rr4HpjzIMC81A8arPiELhdAcbct6TWiU5DGGczMvJGSckKeJlQtCZ0DDFIssVG2Y07FlqUYliYDQhNBKfBBZQmRKir5nl4HCgR8WqFX+XcM7VVUTOsUw6A0GQ384uPYuBpfu6ZebscFphL2GR71NJtDrOI7qsFP416dn3qB/5V4GXnDqD4/gu4MKP/SuOt7Q991ez3XdwN/iOZY5p1DL4Dpja2NZpd9gwUh83Ja24XjmN7i/V/sUc7vjPDy0DOi4TfZo9yhaG2eP6+5weqAwI/pMzjkOY5QxQaQyDCIUzTiJD5EqgARZ4x1eIGWDCridzVT5oZoXiQslqPltqzTO2ETOa9WXkeYOtM3gbNn0UnXfK9iD4qpZQl98LJ9bOYzN3whfOvVA+V8f5f/T/w9PeX+l97IjM9CEReqdmvRCKd3Jxy80TjIFezfSV8dmWr+hV7NZL3C8O9tY+10zH7QxUW3UiSqlp1hGcxTGMcKFSh4Raddkbx1I0dp+flR/SjiOk7A4sNxPSZkrMw=="
    },
    "opencl:bandwidth_d2d": {
        "backend": "opencl",
        "entry_point": "kernel_bandwidth_d2d",
        "entry_points": [
            "kernel_bandwidth_d2d"
        ],
        "filename": "bandwidth_d2d.cl",
        "format": "opencl_source",
        "sha256": "6772c23d685e2aaa510de7cb8163be96f6cc605a32a768fc12ee9c689cf47663",
        "size_bytes": 645,
        "workload": "bandwidth_d2d",
        "zlib_b64": "eNptUMtuwjAQvOcr9ujwCgRUVaEcAOVQiT7UcqmqykpsN1gNdmQ7PFrx7zgPIqWwJ693ZnZ2vI4DHSBKat1Pshx+mBIsDSCOBN1zajaY+hSQzJggqVtgQ2HUETLJhUHaDWoGbhEK3HzxGADCOEllHKWQW/ykA1SbHjSfRAptLiOtSA/yVIoERL7FO0Ym5cK5IhvwIDwYJjSXApI8UjSAF+tpuYIljAa+Nx4MAREZpUwTRmHk3/djbkAbxaItt5JEZsdSbvGxDt/xa/iGw1X4FD6vAxj7gEZ3/fhoWGUGLItCF9qfe8UNKzQ8x8G4uht2ktObGSAHbP2/XzFriRNTBtFGtMJocEUqJa4et/OBv6tZYv3MAJWNmzBTy2NO0dCdXsGLNZTdZGj+yxrOt1Q1BDg9WLzdMy2fD42dqu/OatGLu6LsuZ92+GWJ9qLyWemenJNzBppTzeA="
    },
    "opencl:compute_bf16": {
        "backend": "opencl",
        "entry_point": "kernel_compute_bf16",
        "entry_points": [
            "kernel_compute_bf16"
        ],
        "filename": "compute_bf16.cl",
        "format": "opencl_source",
        "sha256": "651cb284414429fc2682fe977019f390447f75bc4282fc5494191ceae1e029af",
        "size_bytes": 422,
        "workload": "compute_bf16",
        "zlib_b64": "eNp1kD9PwzAUxPd8ilNZ0gpUWBjCFNogKgGN0na2HOcljeLYkf9Ay6fHiSpRBrbn893z77xcRFhAGG3tXTN4dGQUyQRC94N3xMr64RGxHkgJOR+tmXLmjEG3ysV2nlwC7No/2tLnTYKYsUbqkkvUUnO3gPaOlb6+hQ9xKN8zktSTcvYitY4Md61WQZgysEQV439O5QSSGnHEEtnJkbIhgcZzUyU4fOwOeb4t9tkaQbWOqypcYBs6rN6wwlfrjgEEKjz0SbCCS24wFX15T8fV23zH8qxg+9ciS9fTuNlnRYJ7xHQaZCtaJ89ouKPqCd9kNGreEaj3cqIfAZdRdEPGaIPZFVLyyyG56Ow/FGiVdcaL6Stm0Q8iz4t2"
    },
    "opencl:compute_fp16": {
        "backend": "opencl",
        "entry_point": "kernel_compute_fp16",
        "entry_points": [
            "kernel_compute_fp16"
        ],
        "filename": "compute_fp16.cl",
        "format": "opencl_source",
        "sha256": "1429011c58aa76d427c80d0f5bb9bd1f367555179e5b07fcb88b9af3768c572a",
        "size_bytes": 3614,
        "workload": "compute_fp16",
        "zlib_b64": "eNqdVm1vm0gQ/s6vmDZStSROwjuR09zJlzptpF5s2YlUqTqtsL04KBjoAlWqNv/9ZnmJFwPXE/40wz7Ps8/MrBfOjxU4hjWP0/R0m+TwxHjEwjGs412SZ4z6ie4AiRMWrUNVQKdRxn9AEgdRRlJ1XBGojBewyV+3YyCUbsN45YXgh7GXHUOcZ3SV+yPIkQ5RvqMsZDsWZWn1KMgY97IgjvBBwYGUsQ31GtmqMDLh60c4h+lzxqIUGbDNPb5B5yF9euSvRmbzJZ1PF/T+02I6+VCEt/fTxRgucBFrOwYLrsDWDSAX8OiFvgHeep3Cs1i8+XsiIgNCL2JldPN5Nk+FgXNFOQp8eLNhfhCxDZH2VeHXLyD1wvVitlzSj/MH+mmypDdzsf7uHbzpeK4cMc5jDm8f7pYP8/lscT/9MIYZdv/6c3MknH3LA46epG2B1b14i0LRJvCVo4R7252HXZjeocT0y/30bnk7u2vQxsAibxUyRaG0HCd8j4NN12iJAvg7HCv6yHiwzl7nW6DWOMWsY9KHi9LMpaXG9Lufr1T4eSi2Rd9XQESsbllWGaXBhmjqZQHGmRGB+uOqYavWEj/OspxHJfxFkbYQ5wM8+ii2ELHq7zyinRk2HunSKpwKhzEnZaqOQDuz/WprSWSV/IfKqqmyKlSMTplIkiGnYi880aRokir2QMoBB893zTBUgsWMREUtaQNWNIlTCSrkRlCJtsER28rgqABHewcljmsSqLKtnelYHJxgkZpjyBWQYlK4kONfBnQ7V7EVe9aF+79pleXKhD7EhD7MhN5nwhhiwhhmwugzYQ4xYQ4zYfaZsIaYsIaZsPpM2ENM2MNM2H0mnCEmnGEmnD4T7hAT7jATbsNE4cLHFyCp3gzoRMsvRfBeek9cwslJkMk3dv2eyyMeh+Hr471U4gW8EivC93CRCxmRyELF9S/uKHEncw3vxlF5EVYtesXoNUbvxxg1xujHmDXG7MdYNcbqx9g1xu7HODXG6ce4NcZtYH7TILz8f9ugDkyrQR2YVoM6MK0GdWBaDerAtBrUgWk1qIl5aX81VJ8sxZuvPP1cO8P0RE71yzZDlxh6k6F3MwyJYTQZRjfDlBhmk2F2MyyJYTUZVjfDlhh2k2F3MxyJ4TQZTjfDlRhuk+EWDIkiLoT4CQlBip/oeK+QVCu+yfe5fpAbB7kp8sbBqH57iHVAsQ9y5yB31Y6y8p2oqyhHeMKicMYYm2VsidguY0fEtUj1Jf4Vb9l/hAKW29gs36kq/Fnoj8FLabEd0Z7dm2tN/HLUeVH+BYLmy04="
    },
    "opencl:compute_fp32": {
        "backend": "opencl",
        "entry_point": "kernel_compute_fp32",
        "entry_points": [
            "kernel_compute_fp32"
        ],
        "filename": "compute_fp32.cl",
        "format": "opencl_source",
        "sha256": "904303964f77dd8fcc06e9046831513c5689362c31ab4a11388c2be3de2f8bf6",
        "size_bytes": 2303,
        "workload": "compute_fp32",
        "zlib_b64": "eNqNlV1vmzAUhu/zK440qYKmTfgIpEraTVmXapU6Nep6N00IiMlQCUbGrjpN/e87NtDy4ajhIjoHP+9r88aG6ekITiFmtCzPd4WAJ8Jyki0gpvtCcBIkheuAQQuSx5kp0XXO2V8oaJpzozQXtSBo8xJbfb1dgBEEu4xGYQZJRkN+ClTwIBLJGQiUQy72AcnInuS8rG+lnLCQpzTHG0oDJSHbIOx0kVrIisV/YArrF07yEhWwEyHbLuAe13p9B9dgT5ypO7HAyNHxmcDNBh/l5sdKqe83P4PN+iF4/P6wXn1T5e3j+mEBFzho+/jjwBU4ng/GBYRxLPYiCzllJbwggi5QhKnqlGdd3N1vSmk/HY2CoEoGnmm61aVkjACvfkKMlJylMX+LSlExBsI1ofUHW/G1hjpB6u9HJvzrm+1w3VdgyNrcEV4vNEi3hmUuFZwmYEjq81VnWY2XvBjhguUV/joaTB7iDMk+NKyJ4+G2qNYI53KUMqNqzTOwJl5Sz9lWR0FBS41D1HWIlINzwCInO7Q4l1Pgv64sl9VCK4JZOGxNbNTDGAvLdxRpqGHTUAnggDDhBGxP1JPUYvsosa0XO0eJHb3YPUrs6sWzo8Qzvdg7Suzpxf5RYl8vnh8lnrfElZqyap/jAZIOYimLy9ZxWsJ4nPL2xv5UsHC3D0HkjGbZ2+13K/l+qM1UeQkXQtrIpm2kTolV72JmnQG+7NQurB/tjbEbxj7MOA3jHGbchnEPM7OGmR1mvIbxDjN+w/iHmXnDzDvMBwHhsf0wIA0zCEjDDALSMIOANMwgIA0zCEjDDALqMq/Dl6vcf/QJZWmZpDluY8wN9/xJq7d7vdPrXdl31lFf78isJ/F6vd/r55r3byn28iOD/+8Y5JrGWDuydqt6Jmuvqn1ZNyb19/EXHunf0gEftz0Z+pomfFH+CwjLQE1nWC/zm2tLXvLwv47+A2RqY5A="
    },
    "opencl:compute_fp4": {
        "backend": "opencl",
        "entry_point": "kernel_compute_fp4",
        "entry_points": [
            "kernel_compute_fp4"
        ],
        "filename": "compute_fp4.cl",
        "format": "opencl_source",
        "sha256": "93f49dfec168e397ba509d95444e911cfa7f052bf6ac02dfc398c5ca5acbcb39",
        "size_bytes": 417,
        "workload": "compute_fp4",
        "zlib_b64": "eNpVkEtvwjAQhO/5FSN6CagVPfSUnlKIVKSqRAHOluNsgoVjR37w6K+vEyGV3nbHM+tvd7lIsICwxrmXbgg4kdWkMgjTD8ETa4c3pGYgLdR8dBba2xsGI7VP3Ty7+9mDfXTlH5sMKWOdMjVXaJXhfgETPKtD+4wQ09ChZ6SoJ+3dXZKeLPfS6ChMGTiihvF/XT1x5FYcsURx9aRdTKAL3DYZDt+7Q1luq32xRlSd57qJD9jGFVZfWOEi/TGC4Ey6MRbjfptdPo7cljtWFhXbf1ZFvp7Kzb6oMrwipeugpJBe3dBxT807fsgatPxEoD6oiXoEWybJE1kbJ88eULK//xUXJwcdA2fCMaJduKUJ435DSO28DWK6wyz5BfKBi10="
    },
    "opencl:compute_fp64": {
        "backend": "opencl",
        "entry_point": "kernel_compute_fp64",
        "entry_points": [
            "kernel_compute_fp64"
        ],
        "filename": "compute_fp64.cl",
        "format": "opencl_source",
        "sha256": "55be8e43601efa7730cc65501fab0531605ea49542b02c67f55cf90538260a77",
        "size_bytes": 2544,
        "workload": "compute_fp64",
        "zlib_b64": "eNqNlW1vmzAQx9/zKa6tVEGTtoQEEqXtpqwla6UuifIgTZomC4iToRLIDJ46rf3uO/OQQAA1vEju8O9/Pt/Z5vpCggtwWBCGl+sthxfKfOr1wQk2Wx5RstoaHZCDLfUdTxGo6UfsL2wD14/kUOmnApLnBTb48tQHmZC1F9iWB8uA2x69gIBHxOarJnDUg883hHp0Q/0oTF+5EWVW5AY+vkhEEFK6JFbRteNcBsz5BddgvkbUD1EDa26xJSbvkZdfbJfLeDIjE3NK5o9Tc/AQm09zc9qHHg62DPzR4A403QC5B5bj8A33rChgIbwiMvw2gK3lxp4mvNR4Hk9CkcW1JJ25KzhZ0pXr06Wcm1yBtzeQs4H76Xg2I18nC/I4mJHhRIyfn8NJxXvpjDIWMDhdjGaLyWQ8nZsPfRhjF+6fi61h9Dd3GQ3zawaaFeQUA/lLdyWdbZm13lhYCnOEIczvc3M0exqPCrI+UN/CEksSIUlb4U/gLqtaLEuAT6m9mEjEXCfa9TnGHOxmVNHxw8Fc73NDxV1QM2Ar8O8w3BpTvwNZ2MqaRmmuxF3KqnITw9g2WVCf7gqJZbHEw2jEmZ/g71J5dgunWG0sWU58Rb3S9GaaLFzCygsCJieu0oQ9pacpFILZZBuE9QHtYkC7EFCriejTtahCyl3izLjf44lukuWkJFMRU69amg4N/FcNTYCZTo7rhAMc9yy0dJ5Ololbx4hbNWLtGLFWI24fI27XiDvHiDs1Yv0YsV4jNo4RGzXi7jHibk4cq1d4ocjpQRMB+I0wbnPH7gYaDTfKb//s3uA+Czxv93ofStyMabDYvIUeF2GEkw8UnyU13dtMbQJe6PEuTJe2Y1oZ06pntIzR6pl2xrTrmU7GdOoZPWP0esbIGKOe6WZMt8B8UCA8vR8WqIIpFaiCKRWogikVqIIpFaiCKRWogikVqMi8l69gsf+CF5S5IX5hcRtj3eJP6t5vHfjagd8WfiGP9NkjnQOJfuAbB3636h4O+UbcwtjgBoikGmhrwm4ndkfYemIbws6ipB/SH3imf4oIuN78bBhXUeBzHL8PVkiS+WT1tTsc9tTcs3jGiO/Sf+r5syE="
    },
    "opencl:compute_fp8": {
        "backend": "opencl",
        "entry_point": "kernel_compute_fp8",
        "entry_points": [
            "kernel_compute_fp8"
        ],
        "filename": "compute_fp8.cl",
        "format": "opencl_source",
        "sha256": "3feaea7ebfb5e11e12ff424f5ff235e7cfd31f2b8479a49532715215136864bc",
        "size_bytes": 417,
        "workload": "compute_fp8",
        "zlib_b64": "eNpVkEtvwjAQhO/5FSN6CagVPVbpKYVIRapKFOBsOc4mWDh25AeP/vo6EVLpbXc8s/52l4sECwhrnHvphoATWU0qgzD9EDyxdnhDagbSQs1HZ6G9vWEwUvvUzbO7nz3YR1f+scmQMtYpU3OFVhnuFzDBszq0zwgxDR16Rop60t7dJenJci+NjsKUgSNqGP/X1RNHbsURSxRXT9rFBLrAbZPh8L07lOW22hdrRNV5rpv4gG1cYfWFFS7SHyMIzqQbYzHut9nl48htuWNlUbH9Z1Xk66nc7IsqwytSug5KCunVDR331Lzjh6xBy08E6oOaqEewZZI8kbVx8uwBJfv7X3FxctAxcCYcI9qFW5ow7jeE1M7bIKY7zJJfAKiLbQ=="
    },
    "opencl:compute_int16": {
        "backend": "opencl",
        "entry_point": "kernel_compute_int16",
        "entry_points": [
            "kernel_compute_int16"
        ],
        "filename": "compute_int16.cl",
        "format": "opencl_source",
        "sha256": "99c9521c20e2d4e535b2a6a67a487dafd87bbadfb32f61ddb860df7a9e3ef781",
        "size_bytes": 1952,
        "workload": "compute_int16",
        "zlib_b64": "eNqVlG9vmzAQxt/zKU6aNEGyNoQ0JEvaSVnLtEqbWmV9N62IPw5FIxgZu8q05bvv7NAMZx4afuU7P/fjeDg8GlgwgITRuj7LKgHfCStJsYCEbivBSZiXfOyDTStSJoUjtUHJ2Q+oKJ7YtbNoKkKtQOpW728XYIdhVtA4KkDgwQCo4GEsNm9UCKXYhqQgW1LyuknlnLCI57R8SdSEpGHUDmLVxoolTzCCYMdJWaMeMhGxdAF32On1J7iG8bk3mpy7YJfIeyYw9s/inIOonyjj8Hl1ozB391/C+2AdPnxcB6sbtb19CNYLmOMhvvkAPLgCb4oezCFKErEVRcQpq2Enj0XJaFGQVPJkygNa1RI8sqwwPFgDzzRPjTbZFuA6sYiRmrM84UevlChBR7jBtdPDln+nR42TxnTswM/Tgwy7vgJb7p2M8KbLME9t11kqcb4BW6reXWlNvbDkYoQLVh7ke6v9iMN3QDvVQ1Tk2PahSXgN7u4DLuHALxiL5nlaZZSmf1fGhkpV2hQxV6tR3Q9Rht/Z3b0NJjMse2ycclqspQYZmyBeT4hngkx6QiYmyEVPyIUJMu0JmZogfk+Ib4LMekJmJsj8PyCKsqHsMPP4KyHIFUu5uWz9WEsYDnPeHvJXFYuybdTcBsf0H1TNSdXA1PYS7w4hOTJqk9Qvo80oRgP1lwzlxDevepRqk4hRh1SbN4w6pNpUYdQh1WYHow6pNiEYdUi1OWB+l1T72hiZpft/XkB4p2vz0vL9Ef11ENV2DXMTzGk9mFbbE6yZ6hxf5mZO02Bz0X/FSf12vHGxr6W1t34DClIk9A=="
    },
    "opencl:compute_int32": {
        "backend": "opencl",
        "entry_point": "kernel_compute_int32",
        "entry_points": [
            "kernel_compute_int32"
        ],
        "filename": "compute_int32.cl",
        "format": "opencl_source",
        "sha256": "43d7ebc4ce8aadd5ba266281896ee7aba93802c3699b73d0b7f880b2f2dc6bfb",
        "size_bytes": 1554,
        "workload": "compute_int32",
        "zlib_b64": "eNqNVNtum0AQfecrRuoL2E1sg40vxJWcxFIitUrk5q1qEJc1RcWAlt3IVet/78yCA0Z2BA+j2ZlzDsPMLIOeBj0IeFYUV1Eu4TfjKUsWEGS7XArmxqmwTNCznKVBYhB2nQr+B/IMM3phLCqGe0Ig3Or2cQG660ZJ5nsJSEz0IJPC9eX2szpCKncuS9iOpaKoQrFg3BNxlh4DBWOh6zUPvipjxYNfMID1XrC0QDxE0uPhAp6w0ruvcAeja3NgXQ9BT1HvjYFlXvmxKHW+re6VyNPzd/d5vXFfHjbr1b1yH1/WmwXMMDmy0ZiwBHNigz4DLwjkTiaeyHgBe0rLlGdJwkLSo5AJWV6Q8EDTXLdsDLxlcXi2SboG+LQaxFkheByI904pUID9EGd61k42utdOVX08G/YN+NtORFj1EnTyjYiJqko3DvWh4ShwvAWdUF+WJ0UdtejhTEielvCD1n4FNhNfURYG/2AknTbCC8Mjwq8QCqKSfEgFUgV9zOCshvv52ppO57dzacBrJew0CKOaYHYimDXB6kSwasK4E2FcEyadCJOaYHci2DVh2okwrQmzDwiKsc14uSO4ekgbSoecm8YiOtDvx6K5FJ9y7kU7r7o97+FaqhAsr8SUe4N3TZIOnZpKasVoC9D01Dr1aWWcUwBNHc1lAE0ZzWUATRXNZQBNEc1lAE0NzWUATYnbHwBoKmjOAg7NC1b9OH7gBH/SJLE3r/j9BpJ0/Er0rdIfkz8pfZv8KV7sg/YfGTSscg=="
    },
    "opencl:compute_int4": {
        "backend": "opencl",
        "entry_point": "kernel_compute_int4",
        "entry_points": [
            "kernel_compute_int4"
        ],
        "filename": "compute_int4.cl",
        "format": "opencl_source",
        "sha256": "6639c3e35378cdee76cc3e8c3d9d8fba967fd465f94fbced6069a2d668238bce",
        "size_bytes": 421,
        "workload": "compute_int4",
        "zlib_b64": "eNpNkM1uwjAQhO95ilF7CagVPfSUnlKIVKSqRAHOlnGWEOHYkb3mp09fJ6Uqt53Z2dW3O5smmEI56/1z0wccyRnSGZTt+sAkWsOvSG1PRunJEC0Muyt6Gxupn2S3AXGfH2L5+zJDKkSj7U5qhOhPYQOLXdg/jRImdII0dWTY36yWyUlurfkzPFEt5L3YjRS5UwfMUFyYjI95NEG6OsP2a70ty1W1KRaIrmdp6tjAKh4w/8Qc55YPEQMnMrV1GM9brvNh56pci7KoxOajKvLFWC43RZXhBSldet2qlvUVjWSq3/BNzmIvjwTqgh6hB7JZkjySc3H1wx1L9g+gpTp6mDhwIhwi21k6+uWQLsJ1xK2K2rMLanzFQ/IDk7KNXA=="
    },
    "opencl:compute_int64": {
        "backend": "opencl",
        "entry_point": "kernel_compute_int64",
        "entry_points": [
            "kernel_compute_int64"
        ],
        "filename": "compute_int64.cl",
        "format": "opencl_source",
        "sha256": "84f78c8daacdbc261b68c0f40c92e6cae5f2464719b768656debb9cc42c4ff49",
        "size_bytes": 1709,
        "workload": "compute_int64",
        "zlib_b64": "eNqVVNtum0AQffdXjNQXsJv4Do6JKzkOViK5SuQmT1Wz4rKmqJhFy26Uqs2/dwZIuk7squZhNLtzzmGYC912C9oQSVGWJ0mh4QeXOc+mEIltoRVnaa6cEVii4HmU2YT1cyV/QiEwYpX2tGGwHQLh5hfXU7AYSzIRBhnoTORJG4RWLNSbj6ARCLneMp7xLc9V2VylistApSKnC+JAyXnMgp1TWGUyl9F36IL/pHheIgMSHch4CjeY7GIFC+ifDrrD0x4s71crdru+WV6vfLBylH/k4IxOwlQ1qp/nl5Xkze0Xduuv2d3V2p9fVu71nb+ewgSDfQfNAGYwGDtgTSCIIr3VWaCELOGJwjqXIst4THp0NQBRlCTcbbUYqysFjyKN91bNagE+bysmealkGqnX0lWoCAuk9hTxbdAopxkyC7v/PrTh11uxBBOfgUW+nXDVJMrS2OrZXgVON2AR6tNsJ60XLXokV1rmNfy59e7lWFB8R50a/Ib+/cp7hwni+AUTHsKEQcmrZOlkY05e/a46KnsUqzAd4mNje09n/tB1zy7O3OVo7i764/uVDQ9NKp5J7hvkwbHkgUEeHkseGuTRseSRQR4fSx4bZOdYsmOQ3WPJrkGe/Be5Ym+ErEcVdwAVetoj59zYCA86nVSZs/mhkEGyDZo9fr3+K1UqXjRilXuOW69Jh06mUjXpNGRo2tVQd2hsvV0ADRKawwAaFjSHATQQaA4DqOloDgOosWgOA6h50vkHgBqEZi/g2dzz5g/2FbfxGzUVa/OA328jycKvRH9Y+yPyx7XvkO/i/+W59Qf8hNbx"
    },
    "opencl:compute_int8": {
        "backend": "opencl",
        "entry_point": "kernel_compute_int8",
        "entry_points": [
            "kernel_compute_int8"
        ],
        "filename": "compute_int8.cl",
        "format": "opencl_source",
        "sha256": "8832767d10cb181154bbb860f5b3b82c96514c460cfa506a453e88fff081076a",
        "size_bytes": 2325,
        "workload": "compute_int8",
        "zlib_b64": "eNqVlWFP2zAQhr/nV5w0aXJaoEka2qoFpA6KhrQJxPg2DStJTYlIncixq6KN/76zk0LaJKiNVPdiP3f2+T07vY4FHYhEmufHi0zBCxOcJWOI0mWmJKMxlyMgacZ4lNganXEpXiFLcYDk9rh0oFVeY9NvN2MglC6SNAwSUNjfgVRJGqqnI/MKXC0pS9iScZmXXbFkIpBxyjcdOWNzGlRfQrOKqYieoQeztWQ8Rx4WKhDzMdziQi9/wCW4J16vf+IA4RhvxUBFz4HwYXQcxhJWLJKpgJ/TKxPs9u4XvZvd04fv97PplTFvHmb3YxjhoI8/d4CNB+fgOp4PZLSJFkRRDmtEkoAzbSGIQbXlQZrlOnrPsnKJa4gg5knM31cS81hSNVr7xOS2iOdlli9bqdvw1wJ8ItwTWQyEQc5wLegBXSAv2LjKxgX2h2piWMGkEhxIMZNNTKd+ih6bEGJidMFROrshuj8CKSaEiwtwbBu+grO+vlb2Ubu32+w92s/ba/Z2B/u595vdPb/ibrztifVmWZQWhQqrFPetoWiLXdopWMFyKeJIvlfurhZbNbw7WKnm3aGyrhu7wwbNtdjnYErFXjBZrpLGc+LYhejxExBNXZxvLWoT66MuChy3pDJFUZJLlZhJWsum3ObN9sI/XXlNGlWL4SAc1T+IR7l3+I3mteyC+XyP7MJ9pw8Pyy48MLvw8+zMf5mYcDCvj+vE3CSOOtpcIJMt1q2zbhvr1Vmvje3X2X4b69dZv409rbOnbeygzg7a2GGdHVZZAz/hR4KURxl5R020cVY52BPodmNZPWRfMhEslgEoLtIkee/+CJVLlpXBjHmGZaF0HP1WjWSOrNYWm445nF1dxJNtQAuKTTugVcSmHdDSYdMOaL2E/wmgRcKmHdDKiMEngJYDm0bgrfXCwu+vPtK4P4+4B7b+HGKmaPcL29f2aWEPtD0sq6C8zn+j7n8wQpBTLQ3BeOZr8R8jcXjz"
    },
    "opencl:compute_tensor_bf16": {
        "backend": "opencl",
        "entry_point": "kernel_compute_tensor_bf16",
        "entry_points": [
            "kernel_compute_tensor_bf16"
        ],
        "filename": "compute_tensor_bf16.cl",
        "format": "opencl_source",
        "sha256": "3d748aabf6eaad8e175c52e14c585fc748850bec9709cdf18a9b1075ca5eae7a",
        "size_bytes": 2122,
        "workload": "compute_tensor_bf16",
        "zlib_b64": "eNqtVG1P2zAQ/p5fcRrSlNDSxoW+rICmAkVDYmtV4NM0WUnqhIg0zvyCiib++86J+5KqQmObJed8zt3j53znax86cAiR4FIeJYWGJyZylg0h4otCK0YVyyUXNIxJD1xesDzKPOMxzpV4gYKnuXKlN7R+dI+bsR5d3AzBpTTJeBhkEGc8UIfAtaKhjpugEQVyvaAsYwuWK2m3UsVEoFKe40bpA5KxOQ1qWljyGYnoEdowXpqT0QMSHYg5xpFRREJmUoeJ4Lqgi0CJdEkXOlNpkb3QIIo0KoFiBmcyvaPT8Yzef5mNR1fl8uZ+PBtCB38OcLqrBel5cA4n/ie8GLycq+noDspresLJCwlL6HR7cH07mcp2FuQMkPHX84Gh23acgzTO5yx+H8EDJgQX8OHh293DdDqZ3Y+vhjDBrFze7s2YYD91Kph81ykf8Jh8nsaOQ2mVVnjm6fyNFLsO4NhNL56LR0RqnefSKsJsqj0Z3/25lfutX7Uq2L8fevBrFyxB+ufgmrWXMGWJ0nTu+t6ps2UtH7lQAwjoM8aBHpXuuf7y+HpgbDemiDWAkBZcGkOjVWa+b+Y+05wlG1MjPH95gbZm6jqPKp4wkAw9/FY3hgaKDkqswCp8ODJWXFjV86BRAuwZdc+w7hkaTzTxe9aoPNtzzZ19BNJdMyv3BxD5JohK8VxDsVkRNSAEMbbUunbc79oi2DvWdjWnXh2jX9MGqNqbXrEju+zQkLT8jRepkyStuvZnJEmr5tSrY/RrmiVZscTX69r6NpnVp2ZxtlXtp9BopGpVw2YcFCJIFgHoXPAsW29voIogFRasXJ7BiTYwRtkGKsvLZG/dCmjVC8wbrj6rrhDMKbYxt3wHzarKm+hr73oNRv4BjOyC/R0zfFT/j1kFts3s1am+pUhjKJ/FWa11bV9xrRvphanFyG9JHwvBSGJlx8pjK0+s7FrZs7L/xqPG4UbEYhOLTSw2sdjEYhOLTSw2MdibKG2H/o7B/TAXJ+M0x5J0MQQPPpeRDCGQtAwMu1z/+tI3Q1uMV+fV+Q2SH2cJ"
    },
    "opencl:compute_tensor_fp16": {
        "backend": "opencl",
        "entry_point": "kernel_compute_tensor_fp16",
        "entry_points": [
            "kernel_compute_tensor_fp16"
        ],
        "filename": "compute_tensor_fp16.cl",
        "format": "opencl_source",
        "sha256": "4ef7e9e03014c2247dee50d05c90df6b5aa2658b589e872f860d4d4c40cc4d73",
        "size_bytes": 2113,
        "workload": "compute_tensor_fp16",
        "zlib_b64": "eNqtVG1P2zAQ/p5fcRrSlNDSxoW+rICmAkVDYmtV4NM0WUnqhIg0zvyCiib++86J+5KqQgPNknM+5+7xc77ztQ8dOIRIcCmPkkLDExM5y4YQ8UWhFaOK5ZILGhekBy4vWB5lnvEY50q8QMHTXLnSG1o/usfNWI8ubobgUppkPAwyiDMeqEPgWtFQx03QiAK5XlCWsQXLlbRbqWIiUCnPcaP0AcnYnAY1LSz5jET0CG0YL83J6AGJDsQc48goIiEzqcNEcF3QRaBEuqQLnam0yF5oEEUalUAxgzOZ3tHpeEbvv83Go6tyeXM/ng2hgz8HON3VgvQ8OIcT/wteDF7O1XR0BzEunnDyQsISOt0eXN9OprKdBTkDJPz9fGDYth3nII3zOYvfx++ACcEFfHr4cfcwnU5m9+OrIUwwKZe3exMm2G+dCibfdconPCafp7HjUFplFZ55On8jw64DOHazi+fiEZFap7m0ijCZak/Cd39upX7rV60I9u+HHvzZBUuQ/jm4Zu0lTFmiNJ27vnfqbFnLRy7UAAL6jHGgR6V7rr88vvSN7cYUsQYQ0oJLY2i00mzg+2buM81ZsjE1wvOXF2hrpq7zqOIJA8nQw291Y2ig6KDEAqzChyNjxYVVPQ8aJcCeUfcM656h8UQTv2eNyrM919zZZyDdNbNyfwCRb4KoFM81FJsVUQNCEGNLrWvH/a4tgr1jbVdz6tUx+jVtgKq96RU7sssODUnL33iROknSqmv/RpK0ak69Oka/plmSFUt8va6tb5NZfWoWZ1vVfgqNRqpWNWzGQSGCZBGAzgXPsvX2BqoIUmHByuUZnGgDY5RtoLK8TPbWrYBWvQC7VjlXPSGYU2xibvkKmlWNN9HT3vQainwYiuxCfYQVPqf/xaqC2mb16lTfUqQxlM/hrNaytq+21oX0wtRg5LekjwVgJLGyY+WxlSdWdq3sWdl/4zHjcCNisYnFJhabWGxisYnFJhabGOxNlLYz/8Tgfplrk3GaYym6GIIHX8tIhhBIWgaG3a1/jV3Qr7pVdT+vzl/cwmOM"
    },
    "opencl:compute_tensor_fp32": {
        "backend": "opencl",
        "entry_point": "kernel_compute_tensor_fp32",
        "entry_points": [
            "kernel_compute_tensor_fp32"
        ],
        "filename": "compute_tensor_fp32.cl",
        "format": "opencl_source",
        "sha256": "3bf6e3b94c994d4e549c4396c8706209c9106d3ed56eb94eeec33254d668d05c",
        "size_bytes": 2190,
        "workload": "compute_tensor_fp32",
        "zlib_b64": "eNqtVW1P2zAQ/p5fcQNpcmhp49KXqIAmBkVDYmtVyqdtsvLidBFpkjkxKhr8950TNyRdhwaapeZ89t3j5+zHbvfAgAPwRJJlh8tUwh0XMY/G4CWrVOac5TzOEsGC9KgHJEl57EWmypjEuXiANAnjnGTmWOexHWkq+uzj1RgIY8socZ0Igihx8gNIZM5cGbRBIgrEcsV4xFc8zjM9FOZcOHmYxDhQ5EDGuc+chucWfM6E9wO6MFmrlTEDltIRPtYRMURCZpl0lyKRKVs5uQjXbCWjPEyjB+Z4nkTHUayDku50dsNmkzlbfJpPzi6K7tViMh9DDydt/JFNxzbhFHpW3wZCh3AxO7uBxSVu1Z0NSZrBGmjPhsvr6SzrRk7MAUl/PrUV465h7IcBvPN5EMbcJ68hasLjI3wzANub8otCTWOfC5EI2Lv9cnM7m03ni8nFGKZ4xufXO89f8J8yFDx7/abu4VqxHwaGwVipFLhPQv8F1ZCium3F4OK4jpdX0imiPBRIvkNE25M1OdWmGsLaPe6a8GsbbIn0T4GovrnkuSbKQp9Y5rGxjdMHh91jHZhR+iahHSvAyK1AG1yWJlkVaJvE6gz+EhjzZT3wUEf+UYTrZBwD1TS00PTQoojLmuFQRSVCu6YJrQJgR2tmus1MV2ViiDXUQcXaJlEb9R7oQG6YafqeVeeuKLZLogqEIkbNbXpHo4E++Z2timskDZsYo4Znj6ot3rCj2+wwUB1ZlUWbJGmn6f0bSdppJA2bGKOGp0mWLPHeEi1qdbLyWHVOahI/hlYrzDfCVW0/Fc5y5YCMRRJF1fAzVOqEQoMV3RPoSwWjnDpQIS91etUjwMpXQF318rN5Dxyf3dmk0H67VHYbU/VWV1j07Vh0G+tNvPAi/TdeJVad15NRfguDb35xJU4ab1V9exvPj1wpHXpWJ7NQBMpSbXvaHmnb13ag7VDb0QsXGhvxqMamGptqbKqxqcamGptqbKqwn6vUT/JXLO672rcM/5dQjgRLMOFDUckYnIwVhRFrPbo8t1STGuPJeDJ+A35TglM="
    },
    "opencl:compute_tensor_int8": {
        "backend": "opencl",
        "entry_point": "kernel_compute_tensor_int8",
        "entry_points": [
            "kernel_compute_tensor_int8"
        ],
        "filename": "compute_tensor_int8.cl",
        "format": "opencl_source",
        "sha256": "c4b3e4dfa8a1da8acd5ee95e3d059dc6c8e6ec3b5a41f2efd9b1167e006a2a34",
        "size_bytes": 2035,
        "workload": "compute_tensor_int8",
        "zlib_b64": "eNqtVG1v2jAQ/p5fcVqlKSlsxEAphfYDW5lWaVsRbT9Nm2WCSaOGOHPsimrqf985MXmhrGqnWbw8Zz++e3x3dufQgUMIpMiyd2Gq4Y7LhMcjCMQ61YpTxZNMSBolagiuSHkSxJ7ZMU2UfIBU4IKbeSO7j+7ZZtiTDxcjcCkNY7FgMWicPwShFV3oVTs3IdFrymO+5onK7FSkuGQqEsl2IuN8SVndWORiJjK4hQ5MNyYs8iHUTC7xELFRgLIyvQil0CldMyWjDV3rWEVp/EBZEGg0mOLGz+Xsis6mc3r9eT6dnOfw4no6H0EXF4f4dbeg1/XgDIbkpAsuGcD5bHIF0RDuel0QaQYbOCI56sQs4cAUfD0bGqkdxzmIVsmSr14n7oBLKSS8ufl2dTObXc6vp+cjuMRyfPyyt1SS/9KR5NmrorzBMMkyWjkOpUU94V5Ey2dq6zqAY6euGBYjBKoscE4KsIxqT6l3F2tF312y5d87vfDg9+5CiNrPwDXYC7myKmm0dH1v7NTY2a2QmDRG7/EQuKOwPdff+MQnyK2oeXYXNBWZIRprS/sbNeFhRTV/nr/5ZIdu6jCiFyzjlo90c4Sf9uBbgEd9C4UPbQPmoQK/ipN7aQFpg0XdEvVK1C9S+WTY5aOSOCjRcYmGjdDkSeiTkkn8ClaCSPf56KTSSfoVrESRwTZ5K7wZrm0e1OHrsQGntVYaQ6sVqW2LmHGQShauGehEijgupytXKYukdZbDU+hr48YYdUd59Uzmy2tGi3sWDc1ne9vYkuLj4OYt1i4aqI37bBJLR+QfHZFdR69XhH36fxQVjuqKHp3it2iYFeRtfdp4B+oprd9tvS6vcOC/z3wsfGWRhtVtWD1o7e8vMypav7HpqGENGtbxCxyShkLSUEgaCsnLFJKGQtJQSBoK0TquMm6f3u+Y6B+YQEzj2Nbg0fkD1XNU0A=="
    },
    "opencl:memory_pattern": {
        "backend": "opencl",
        "entry_point": "kernel_memory_pattern_write",
        "entry_points": [
            "kernel_memory_pattern_write",
            "kernel_memory_pattern_verify"
        ],
        "filename": "memory_pattern.cl",
        "format": "opencl_source",
        "sha256": "28144361ef73390c5662603c4beb41e01d70e346b98e53080b4caf8b3bbaf613",
        "size_bytes": 1925,
        "workload": "memory_pattern",
        "zlib_b64": "eNq9VNuO0zAQfc9XDC/IacP2ku0uva3UVkFCKhfB8oAQWN3ELRZpXNnObltUvh3bcVunl9UKAfPQi+fMmTMzHtcqHlQg5kyIF7NFDj8Iz0jagTmZM77Ci4mU6gQQW5AsTn0NjjLJV7BgNJNI+B0bgssR+IFTSYIzznvC6XSlyQbD12CgHQCE8Sxld5MUcsVdgbt8GkCesmwGWT7HD4wnIjA+2BLNWUK2GEFI4m85iwwdhzNmmZB/zhzAjumepRNJU2LJCOeM45jlmSzS8/g71CBaSpIJyjKY5ROedOCdauFoDCNoXDRr4UVdY4efb6OP+H30AUfj6E309rYDl4AasDTcYRO0NFgQXjQJGLeVKZ1C6HQ1zxNSyYmBZinNSLkKHY6KImiytCUeNQ1+eqCMTgFpH/T7UM994ETmavb1Zd1a3j3CNVzcK2sncE0X17J2Ahe6uIG1E7jLPa6RQ68HSJfmI1UlPIew8WnsHwe19kG/0LkwFWcCTavW0Hch30y/1Bcy3zc3EDZ9H6pKazsKr6/bw7bVagLXGrnWsMaVglUU7GUrGo4GV8MzsNDCRs1hcxCF2w5Z0SW+rrfxPIyLBVN3kiaPbSIyPAcLxomQnMbS7IMB2CU52AzXdbQjR3HulXLPZ0qh7qb+48+ItFowTVDdDqtEo6TpsZ2IEHRNdjFTtRP7K67wKk/X/OztSygOqn3LupWnTdX+RTm/qsjS1ph9Kb8GprIi6+ap3S/W9aD97lv074YQlJM+8mz9v2kZ6SmL1bkRIEC/NX9nknrVd9N89sRxugTaqlVXXXfn23j7T52oVMONeS/3RBPJ5jTGkyRBTpuDUuHORfoNp2td4A=="
    },
    "opencl:memory_sweep": {
        "backend": "opencl",
        "entry_point": "kernel_memory_sweep_write",
        "entry_points": [
            "kernel_memory_sweep_write",
            "kernel_memory_sweep_verify"
        ],
        "filename": "memory_sweep.cl",
        "format": "opencl_source",
        "sha256": "893a5e02f37c3a9ec29efa9528ab573c520ad9ef180cb84cef71fb20323e3c8c",
        "size_bytes": 1500,
        "workload": "memory_sweep",
        "zlib_b64": "eNq9U11vmzAUfedX3L1BkjXQkFDIGqmkVJqUfWjrHqZptSg4zBqxmW0akqn/feYjmdM2U6VN8wOC63PPPfeey7BnQA8SzoR4mRUlfMec4jyAFV4xvkFijXEBJiswTXKrhkZU8g0UjFBpCivoEpCOR2tOJB48eXWHOVluaqKL8DU0wADARCjL2W2cQ5kzmvXgtlwO2neg5Qr9WDOeil1EYJxaO4qWMNAoEkaFfDbRAPaJdyyPJckxlKq5HmDOGUcJK6lsq/HkGwwhqiSmgjAKWRnzNIB3ajjzBczBOTkdjk7sGht+vo4+ovfRBxQtojfR2+sAzsB0oGq4Jy40QqDAvJ0BMN51AkUsRF1vaBhCKj0JEJoTineai5zIFakmrtkGKgt+GqBO+7mFc1WlD3blRyPP80Pfu3IvvLkz/rSYNrgaYW7hpn7MZjCyLUsptqvwanzmepNLZx650Tj0j+BPvQ7vu5e264ehMxo5jhOFOzzHsuQUtAqONTXuDQOhdiPUoEl6fHHMhuXhRnAsJCeJbBxtEJrPureP7pp16WakxzMlQnXWfFgZll1BRFLTtqaPaVT5FD+ZIcgW73OWysrOGpJWCq/qTJvXV5rMNtI/72h3+uqjGvyiLr+qVM3sGn7T9tLWuX/eSNutejDTg1/k7yY7OGT+wz/0/yyoa0POEhVvBAiVYpf/yB6yBHNv0YtjHukZ9en3dT3T/d298ftZMx+oninVOlEs2YokKE5TUxvs4KBVbTt+AXBv2ic="
    }
}
# END EMBEDDED_KERNELS


@dataclass
class CompiledKernelBinary:
    """Compiled or prepared kernel payload ready for BackendPlugin.load_module."""

    backend: str
    workload: str
    entry_point: str
    format: str
    binary_bytes: bytes
    compile_flags: tuple[str, ...] = ()
    compiler_log: str = ""
    cache_key: str = ""


_MULTI_ENTRY_POINTS: dict[str, tuple[str, ...]] = {
    "memory_pattern": ("kernel_memory_pattern_write", "kernel_memory_pattern_verify"),
    "memory_sweep": ("kernel_memory_sweep_write", "kernel_memory_sweep_verify"),
}


def default_entry_points(workload: str) -> tuple[str, ...]:
    """Return all canonical kernel entry-point symbol names for a workload."""
    normalized = workload.replace("-", "_")
    if normalized in _MULTI_ENTRY_POINTS:
        return _MULTI_ENTRY_POINTS[normalized]
    if workload in WORKLOAD_SPECS_BY_NAME:
        return (WORKLOAD_SPECS_BY_NAME[workload].default_entry_point,)
    return (f"kernel_{normalized}",)


def default_entry_point(workload: str) -> str:
    """Return primary C/kernel entry point name for a workload slug."""
    return default_entry_points(workload)[0]


def kernel_key(backend: str, workload: str) -> str:
    """Build deterministic registry lookup key."""
    return f"{backend}:{workload}"


def compute_cache_key(
    backend: str,
    target_arch: str,
    driver_version: str,
    workload: str,
    source_hash: str,
    compile_flags: Sequence[str],
) -> str:
    """Compute SHA-256 cache key per Section 6 specification."""
    payload = "|".join(
        [
            backend,
            target_arch,
            driver_version,
            workload,
            source_hash,
            " ".join(compile_flags),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_home_dir(home: Path | None = None) -> Path | None:
    if home is not None:
        return home
    try:
        return Path.home()
    except (RuntimeError, OSError):
        return None


def default_cache_dir(
    *,
    platform_name: str | None = None,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path | None:
    """Resolve the OS-standard runtime binary cache directory, or None if unavailable.

    - Explicit override: $CROSS_GPU_CACHE_DIR
    - macOS: ~/Library/Caches/cross-gpu
    - Linux/POSIX: $XDG_CACHE_HOME/cross-gpu (if absolute), fallback ~/.cache/cross-gpu
    """
    active_env = os.environ if env is None else env
    explicit = active_env.get("CROSS_GPU_CACHE_DIR", "").strip()
    if explicit:
        return Path(explicit)

    active_platform = sys.platform if platform_name is None else platform_name
    if active_platform == "darwin":
        resolved_home = _safe_home_dir(home)
        if resolved_home is None:
            return None
        return resolved_home / "Library" / "Caches" / "cross-gpu"

    xdg_cache = active_env.get("XDG_CACHE_HOME", "").strip()
    if xdg_cache:
        xdg_path = Path(xdg_cache)
        if xdg_path.is_absolute():
            return xdg_path / "cross-gpu"

    resolved_home = _safe_home_dir(home)
    if resolved_home is None:
        return None
    return resolved_home / ".cache" / "cross-gpu"


class RuntimeBinaryCache:
    """In-memory + best-effort disk cache for compiled kernel binaries."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._memory: dict[str, CompiledKernelBinary] = {}
        self.cache_dir: Path | None = (
            cache_dir if cache_dir is not None else default_cache_dir()
        )

    def get(self, cache_key: str) -> CompiledKernelBinary | None:
        if cache_key in self._memory:
            return self._memory[cache_key]
        if self.cache_dir is None:
            return None
        meta_path = self.cache_dir / f"{cache_key}.json"
        bin_path = self.cache_dir / f"{cache_key}.bin"
        try:
            if not (meta_path.is_file() and bin_path.is_file()):
                return None
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            raw_bytes = bin_path.read_bytes()
            if hashlib.sha256(raw_bytes).hexdigest() != meta.get("binary_sha256"):
                return None
            entry = CompiledKernelBinary(
                backend=meta["backend"],
                workload=meta["workload"],
                entry_point=meta["entry_point"],
                format=meta["format"],
                binary_bytes=raw_bytes,
                compile_flags=tuple(meta.get("compile_flags", ())),
                compiler_log=meta.get("compiler_log", ""),
                cache_key=cache_key,
            )
            self._memory[cache_key] = entry
            return entry
        except (OSError, KeyError, ValueError, TypeError):
            return None

    def put(self, cache_key: str, compiled: CompiledKernelBinary) -> None:
        self._memory[cache_key] = compiled
        if self.cache_dir is None:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "backend": compiled.backend,
                "workload": compiled.workload,
                "entry_point": compiled.entry_point,
                "format": compiled.format,
                "compile_flags": list(compiled.compile_flags),
                "compiler_log": compiled.compiler_log,
                "binary_sha256": hashlib.sha256(compiled.binary_bytes).hexdigest(),
            }
            (self.cache_dir / f"{cache_key}.bin").write_bytes(compiled.binary_bytes)
            (self.cache_dir / f"{cache_key}.json").write_text(
                json.dumps(meta, sort_keys=True),
                encoding="utf-8",
            )
        except OSError:
            # Degrade gracefully on read-only filesystems.
            return


_DEFAULT_CACHE = RuntimeBinaryCache()


def encode_payload(raw_bytes: bytes) -> tuple[str, str]:
    """Compress raw bytes with zlib and encode as base64 + SHA-256 hex."""
    digest = hashlib.sha256(raw_bytes).hexdigest()
    compressed = zlib.compress(raw_bytes, level=9)
    b64 = base64.b64encode(compressed).decode("ascii")
    return b64, digest


def decode_payload(entry: dict[str, Any]) -> bytes:
    """Decode and verify integrity of an embedded kernel payload entry."""
    required = ("backend", "workload", "format", "sha256", "size_bytes", "zlib_b64")
    for field_name in required:
        if field_name not in entry:
            raise ValueError(f"Embedded kernel entry missing field {field_name!r}")
    compressed = base64.b64decode(entry["zlib_b64"].encode("ascii"), validate=True)
    raw = zlib.decompress(compressed)
    if len(raw) != int(entry["size_bytes"]):
        raise ValueError(
            f"Size mismatch for {entry['backend']}:{entry['workload']}: "
            f"expected {entry['size_bytes']}, got {len(raw)}"
        )
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != entry["sha256"]:
        raise ValueError(
            f"SHA-256 mismatch for {entry['backend']}:{entry['workload']}: "
            f"expected {entry['sha256']}, got {actual_sha}"
        )
    return raw


def get_embedded_kernel(
    backend: str,
    workload: str,
    registry: dict[str, dict[str, Any]] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Retrieve and verify embedded kernel bytes for (backend, workload)."""
    store = EMBEDDED_KERNELS if registry is None else registry
    candidates = (
        kernel_key(backend, workload),
        kernel_key(backend, workload.replace("-", "_")),
    )
    for key in candidates:
        if key in store:
            entry = store[key]
            return decode_payload(entry), entry
    raise KernelNotFoundError(
        f"No kernel registered for backend={backend!r}, workload={workload!r}"
    )


def scan_kernels_directory(kernels_dir: Path = KERNELS_DIR) -> dict[str, dict[str, Any]]:
    """Scan kernels/<backend>/ deterministically and pack all kernel files."""
    packed: dict[str, dict[str, Any]] = {}
    if not kernels_dir.exists():
        return packed

    for backend in sorted(BACKEND_EXTENSION_FORMATS):
        backend_dir = kernels_dir / backend
        if not backend_dir.is_dir():
            continue
        ext_map = BACKEND_EXTENSION_FORMATS[backend]
        for file_path in sorted(backend_dir.iterdir(), key=lambda p: p.name):
            if not file_path.is_file():
                continue
            if file_path.name.startswith(".") or file_path.name.lower() == "readme.md":
                continue
            fmt = ext_map.get(file_path.suffix.lower())
            if fmt is None:
                continue
            workload = file_path.stem
            raw_bytes = file_path.read_bytes()
            b64, sha256_hex = encode_payload(raw_bytes)
            key = kernel_key(backend, workload)
            entry_points = list(default_entry_points(workload))
            packed[key] = {
                "backend": backend,
                "workload": workload,
                "filename": file_path.name,
                "format": fmt,
                "entry_point": entry_points[0],
                "entry_points": entry_points,
                "sha256": sha256_hex,
                "size_bytes": len(raw_bytes),
                "zlib_b64": b64,
            }
    return packed


def pack_kernels(
    kernels_dir: Path = KERNELS_DIR,
    target_module_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Pack kernel payloads from kernels_dir and rewrite EMBEDDED_KERNELS in module."""
    packed = scan_kernels_directory(kernels_dir)
    target = Path(__file__).resolve() if target_module_path is None else target_module_path
    content = target.read_text(encoding="utf-8")
    begin_marker = "# BEGIN EMBEDDED_KERNELS"
    end_marker = "# END EMBEDDED_KERNELS"
    if begin_marker not in content or end_marker not in content:
        raise RuntimeError(f"Missing EMBEDDED_KERNELS markers in {target}")

    serialized = json.dumps(packed, indent=4, sort_keys=True)
    replacement = (
        f"{begin_marker}\n"
        f"EMBEDDED_KERNELS: dict[str, dict[str, Any]] = {serialized}\n"
        f"{end_marker}"
    )
    prefix = content.split(begin_marker, 1)[0]
    suffix = content.split(end_marker, 1)[1]
    target.write_text(prefix + replacement + suffix, encoding="utf-8")
    return packed


def verify_embedded_kernels(
    registry: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Validate integrity and decompressibility of every packed entry."""
    store = EMBEDDED_KERNELS if registry is None else registry
    errors: list[str] = []
    for key, entry in sorted(store.items()):
        try:
            decode_payload(entry)
        except Exception as exc:
            errors.append(f"{key}: {exc}")
    return errors


def _capability_macros(
    capabilities: DeviceCapabilities | dict[str, bool] | None,
) -> list[str]:
    if capabilities is None:
        return []
    cap_dict = (
        capabilities.to_dict()
        if isinstance(capabilities, DeviceCapabilities)
        else dict(capabilities)
    )
    return [
        f"-DCROSS_GPU_HAS_{name.upper()}={1 if enabled else 0}"
        for name, enabled in sorted(cap_dict.items())
    ]


class _NvrtcApi:
    """Explicitly typed ctypes bindings for NVIDIA NVRTC."""

    def __init__(self, lib: ctypes.CDLL) -> None:
        c_void_pp = ctypes.POINTER(ctypes.c_void_p)
        c_size_p = ctypes.POINTER(ctypes.c_size_t)
        c_char_pp = ctypes.POINTER(ctypes.c_char_p)

        self.nvrtcCreateProgram = bind_c_symbol(
            lib,
            "nvrtcCreateProgram",
            ctypes.c_int,
            [c_void_pp, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, c_char_pp, c_char_pp],
        )
        self.nvrtcDestroyProgram = bind_c_symbol(
            lib, "nvrtcDestroyProgram", ctypes.c_int, [c_void_pp]
        )
        self.nvrtcCompileProgram = bind_c_symbol(
            lib, "nvrtcCompileProgram", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int, c_char_pp]
        )
        self.nvrtcGetProgramLogSize = bind_c_symbol(
            lib, "nvrtcGetProgramLogSize", ctypes.c_int, [ctypes.c_void_p, c_size_p]
        )
        self.nvrtcGetProgramLog = bind_c_symbol(
            lib, "nvrtcGetProgramLog", ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p]
        )
        self.nvrtcGetPTXSize = bind_c_symbol(
            lib, "nvrtcGetPTXSize", ctypes.c_int, [ctypes.c_void_p, c_size_p]
        )
        self.nvrtcGetPTX = bind_c_symbol(
            lib, "nvrtcGetPTX", ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p]
        )
        self.nvrtcGetCUBINSize = bind_optional_c_symbol(
            lib, "nvrtcGetCUBINSize", ctypes.c_int, [ctypes.c_void_p, c_size_p]
        )
        self.nvrtcGetCUBIN = bind_optional_c_symbol(
            lib, "nvrtcGetCUBIN", ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p]
        )
        self.nvrtcGetErrorString = bind_optional_c_symbol(
            lib, "nvrtcGetErrorString", ctypes.c_char_p, [ctypes.c_int]
        )


class _HiprtcApi:
    """Explicitly typed ctypes bindings for AMD hipRTC."""

    def __init__(self, lib: ctypes.CDLL) -> None:
        c_void_pp = ctypes.POINTER(ctypes.c_void_p)
        c_size_p = ctypes.POINTER(ctypes.c_size_t)
        c_char_pp = ctypes.POINTER(ctypes.c_char_p)

        self.hiprtcCreateProgram = bind_c_symbol(
            lib,
            "hiprtcCreateProgram",
            ctypes.c_int,
            [c_void_pp, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, c_char_pp, c_char_pp],
        )
        self.hiprtcDestroyProgram = bind_c_symbol(
            lib, "hiprtcDestroyProgram", ctypes.c_int, [c_void_pp]
        )
        self.hiprtcCompileProgram = bind_c_symbol(
            lib, "hiprtcCompileProgram", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int, c_char_pp]
        )
        self.hiprtcGetProgramLogSize = bind_c_symbol(
            lib, "hiprtcGetProgramLogSize", ctypes.c_int, [ctypes.c_void_p, c_size_p]
        )
        self.hiprtcGetProgramLog = bind_c_symbol(
            lib, "hiprtcGetProgramLog", ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p]
        )
        self.hiprtcGetCodeSize = bind_c_symbol(
            lib, "hiprtcGetCodeSize", ctypes.c_int, [ctypes.c_void_p, c_size_p]
        )
        self.hiprtcGetCode = bind_c_symbol(
            lib, "hiprtcGetCode", ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p]
        )
        self.hiprtcGetErrorString = bind_optional_c_symbol(
            lib, "hiprtcGetErrorString", ctypes.c_char_p, [ctypes.c_int]
        )


def _compile_cuda_nvrtc(
    source_bytes: bytes,
    workload: str,
    target_arch: str,
    entry_point: str,
    flags: tuple[str, ...],
) -> CompiledKernelBinary:
    try:
        lib = load_shared_library(NVRTC_LIB_CANDIDATES)
    except OSError as exc:
        raise KernelCompilationError(
            f"NVRTC library unavailable for CUDA runtime compilation: {exc}",
            compiler_log=str(exc),
        ) from exc

    api = _NvrtcApi(lib)
    prog = ctypes.c_void_p()
    rc = api.nvrtcCreateProgram(
        ctypes.byref(prog),
        source_bytes,
        f"{workload}.cu".encode("utf-8"),
        0,
        None,
        None,
    )
    if rc != 0:
        raise KernelCompilationError(f"nvrtcCreateProgram failed with code {rc}")

    effective_flags = list(flags)
    if target_arch and not any(f.startswith("--gpu-architecture") for f in effective_flags):
        arch_arg = target_arch if target_arch.startswith(("sm_", "compute_")) else f"compute_{target_arch}"
        effective_flags.append(f"--gpu-architecture={arch_arg}")

    try:
        c_flags = (ctypes.c_char_p * max(1, len(effective_flags)))(
            *[f.encode("utf-8") for f in effective_flags]
        )
        compile_rc = api.nvrtcCompileProgram(
            prog.value, len(effective_flags), c_flags if effective_flags else None
        )
        log_size = ctypes.c_size_t(0)
        log_text = ""
        if api.nvrtcGetProgramLogSize(prog.value, ctypes.byref(log_size)) == 0 and log_size.value > 1:
            log_buf = ctypes.create_string_buffer(log_size.value)
            if api.nvrtcGetProgramLog(prog.value, log_buf) == 0:
                log_text = log_buf.value.decode("utf-8", errors="replace")

        if compile_rc != 0:
            err_name = ""
            if api.nvrtcGetErrorString is not None:
                raw_err = api.nvrtcGetErrorString(compile_rc)
                if raw_err:
                    err_name = raw_err.decode("utf-8", errors="replace")
            raise KernelCompilationError(
                f"nvrtcCompileProgram failed ({err_name or compile_rc})",
                compiler_log=log_text,
            )

        # Prefer native CUBIN when compiling for a concrete sm_XX target and supported by NVRTC.
        if (
            target_arch.startswith("sm_")
            and api.nvrtcGetCUBINSize is not None
            and api.nvrtcGetCUBIN is not None
        ):
            cubin_size = ctypes.c_size_t(0)
            if api.nvrtcGetCUBINSize(prog.value, ctypes.byref(cubin_size)) == 0 and cubin_size.value > 0:
                cubin_buf = ctypes.create_string_buffer(cubin_size.value)
                if api.nvrtcGetCUBIN(prog.value, cubin_buf) == 0:
                    return CompiledKernelBinary(
                        backend="cuda",
                        workload=workload,
                        entry_point=entry_point,
                        format="cubin",
                        binary_bytes=bytes(cubin_buf.raw[: cubin_size.value]),
                        compile_flags=tuple(effective_flags),
                        compiler_log=log_text,
                    )

        ptx_size = ctypes.c_size_t(0)
        if api.nvrtcGetPTXSize(prog.value, ctypes.byref(ptx_size)) != 0 or ptx_size.value == 0:
            raise KernelCompilationError("nvrtcGetPTXSize failed", compiler_log=log_text)
        ptx_buf = ctypes.create_string_buffer(ptx_size.value)
        if api.nvrtcGetPTX(prog.value, ptx_buf) != 0:
            raise KernelCompilationError("nvrtcGetPTX failed", compiler_log=log_text)
        return CompiledKernelBinary(
            backend="cuda",
            workload=workload,
            entry_point=entry_point,
            format="ptx",
            binary_bytes=bytes(ptx_buf.raw[: ptx_size.value]),
            compile_flags=tuple(effective_flags),
            compiler_log=log_text,
        )
    finally:
        api.nvrtcDestroyProgram(ctypes.byref(prog))


def _compile_hip_hiprtc(
    source_bytes: bytes,
    workload: str,
    target_arch: str,
    entry_point: str,
    flags: tuple[str, ...],
) -> CompiledKernelBinary:
    try:
        lib = load_shared_library(HIPRTC_LIB_CANDIDATES)
    except OSError as exc:
        raise KernelCompilationError(
            f"hipRTC library unavailable for HIP runtime compilation: {exc}",
            compiler_log=str(exc),
        ) from exc

    api = _HiprtcApi(lib)
    prog = ctypes.c_void_p()
    rc = api.hiprtcCreateProgram(
        ctypes.byref(prog),
        source_bytes,
        f"{workload}.hip".encode("utf-8"),
        0,
        None,
        None,
    )
    if rc != 0:
        raise KernelCompilationError(f"hiprtcCreateProgram failed with code {rc}")

    effective_flags = list(flags)
    if target_arch and not any(f.startswith("--offload-arch") for f in effective_flags):
        effective_flags.append(f"--offload-arch={target_arch}")

    try:
        c_flags = (ctypes.c_char_p * max(1, len(effective_flags)))(
            *[f.encode("utf-8") for f in effective_flags]
        )
        compile_rc = api.hiprtcCompileProgram(
            prog.value, len(effective_flags), c_flags if effective_flags else None
        )
        log_size = ctypes.c_size_t(0)
        log_text = ""
        if api.hiprtcGetProgramLogSize(prog.value, ctypes.byref(log_size)) == 0 and log_size.value > 1:
            log_buf = ctypes.create_string_buffer(log_size.value)
            if api.hiprtcGetProgramLog(prog.value, log_buf) == 0:
                log_text = log_buf.value.decode("utf-8", errors="replace")

        if compile_rc != 0:
            raise KernelCompilationError(
                f"hiprtcCompileProgram failed with code {compile_rc}",
                compiler_log=log_text,
            )

        code_size = ctypes.c_size_t(0)
        if api.hiprtcGetCodeSize(prog.value, ctypes.byref(code_size)) != 0 or code_size.value == 0:
            raise KernelCompilationError("hiprtcGetCodeSize failed", compiler_log=log_text)
        code_buf = ctypes.create_string_buffer(code_size.value)
        if api.hiprtcGetCode(prog.value, code_buf) != 0:
            raise KernelCompilationError("hiprtcGetCode failed", compiler_log=log_text)

        return CompiledKernelBinary(
            backend="hip",
            workload=workload,
            entry_point=entry_point,
            format="hsaco",
            binary_bytes=bytes(code_buf.raw[: code_size.value]),
            compile_flags=tuple(effective_flags),
            compiler_log=log_text,
        )
    finally:
        api.hiprtcDestroyProgram(ctypes.byref(prog))


def _compile_levelzero_spirv(
    source_bytes: bytes,
    workload: str,
    target_arch: str,
    entry_point: str,
    flags: tuple[str, ...],
) -> CompiledKernelBinary:
    if source_bytes[:4] in (SPIRV_MAGIC_LE, SPIRV_MAGIC_BE):
        return CompiledKernelBinary(
            backend="levelzero",
            workload=workload,
            entry_point=entry_point,
            format="spirv",
            binary_bytes=source_bytes,
            compile_flags=flags,
        )

    ocloc_bin = shutil.which("ocloc")
    clang_bin = shutil.which("clang")
    llvm_spirv_bin = shutil.which("llvm-spirv")

    with tempfile.TemporaryDirectory(prefix="cross_gpu_l0_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        src_path = tmp_path / f"{workload}.cl"
        src_path.write_bytes(source_bytes)

        if clang_bin and llvm_spirv_bin:
            bc_path = tmp_path / f"{workload}.bc"
            spv_path = tmp_path / f"{workload}.spv"
            clang_cmd = [
                clang_bin,
                "-cc1",
                "-triple",
                "spir64-unknown-unknown",
                "-emit-llvm-bc",
                "-cl-std=CL3.0",
                *[f for f in flags if f.startswith(("-D", "-I"))],
                "-o",
                str(bc_path),
                str(src_path),
            ]
            res1 = subprocess.run(clang_cmd, capture_output=True, text=True, check=False)
            if res1.returncode != 0 or not bc_path.is_file():
                raise KernelCompilationError(
                    f"clang SPIR-V bitcode compilation failed (rc={res1.returncode})",
                    compiler_log=f"{res1.stdout}\n{res1.stderr}".strip(),
                )
            res2 = subprocess.run(
                [llvm_spirv_bin, str(bc_path), "-o", str(spv_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if res2.returncode != 0 or not spv_path.is_file():
                raise KernelCompilationError(
                    f"llvm-spirv translation failed (rc={res2.returncode})",
                    compiler_log=f"{res2.stdout}\n{res2.stderr}".strip(),
                )
            return CompiledKernelBinary(
                backend="levelzero",
                workload=workload,
                entry_point=entry_point,
                format="spirv",
                binary_bytes=spv_path.read_bytes(),
                compile_flags=flags,
                compiler_log=f"{res1.stderr}\n{res2.stderr}".strip(),
            )

        if ocloc_bin:
            out_prefix = tmp_path / workload
            cmd = [
                ocloc_bin,
                "compile",
                "-file",
                str(src_path),
                "-spirv_only",
                "-out_dir",
                str(tmp_path),
                "-output",
                out_prefix.name,
            ]
            if target_arch:
                cmd.extend(["-device", target_arch])
            if flags:
                cmd.extend(["-options", " ".join(flags)])
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            spv_candidates = sorted(tmp_path.glob("*.spv"))
            if res.returncode != 0 or not spv_candidates:
                raise KernelCompilationError(
                    f"ocloc SPIR-V compilation failed (rc={res.returncode})",
                    compiler_log=f"{res.stdout}\n{res.stderr}".strip(),
                )
            return CompiledKernelBinary(
                backend="levelzero",
                workload=workload,
                entry_point=entry_point,
                format="spirv",
                binary_bytes=spv_candidates[0].read_bytes(),
                compile_flags=flags,
                compiler_log=f"{res.stdout}\n{res.stderr}".strip(),
            )

    raise KernelCompilationError(
        "No Level Zero SPIR-V compiler toolchain found (need clang + llvm-spirv or ocloc)",
        compiler_log="Missing clang/llvm-spirv and ocloc in PATH",
    )


def _prepare_source_macros(source_bytes: bytes, flags: tuple[str, ...]) -> bytes:
    """Prepend preprocessor #define lines for backends without flag-only macro passing."""
    defines: list[str] = []
    for flag in flags:
        if not flag.startswith("-D"):
            continue
        body = flag[2:]
        if "=" in body:
            name, val = body.split("=", 1)
            defines.append(f"#define {name} {val}")
        elif body:
            defines.append(f"#define {body} 1")
    if not defines:
        return source_bytes
    preamble = ("\n".join(defines) + "\n").encode("utf-8")
    return preamble + source_bytes


def _compile_metal(
    source_bytes: bytes,
    workload: str,
    entry_point: str,
    flags: tuple[str, ...],
) -> CompiledKernelBinary:
    if source_bytes[:4] == METALLIB_MAGIC:
        return CompiledKernelBinary(
            backend="metal",
            workload=workload,
            entry_point=entry_point,
            format="metallib",
            binary_bytes=source_bytes,
            compile_flags=flags,
        )

    prepared_source = _prepare_source_macros(source_bytes, flags)
    xcrun_bin = shutil.which("xcrun") if sys.platform == "darwin" else None
    if not xcrun_bin:
        return CompiledKernelBinary(
            backend="metal",
            workload=workload,
            entry_point=entry_point,
            format="msl_source",
            binary_bytes=prepared_source,
            compile_flags=flags,
        )

    with tempfile.TemporaryDirectory(prefix="cross_gpu_mtl_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        src_path = tmp_path / f"{workload}.metal"
        air_path = tmp_path / f"{workload}.air"
        lib_path = tmp_path / f"{workload}.metallib"
        src_path.write_bytes(prepared_source)

        res1 = subprocess.run(
            [xcrun_bin, "-sdk", "macosx", "metal", "-c", str(src_path), "-o", str(air_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        combined_err = f"{res1.stdout}\n{res1.stderr}".strip()
        if res1.returncode != 0:
            if "unable to find utility" in combined_err.lower():
                return CompiledKernelBinary(
                    backend="metal",
                    workload=workload,
                    entry_point=entry_point,
                    format="msl_source",
                    binary_bytes=prepared_source,
                    compile_flags=flags,
                    compiler_log=combined_err,
                )
            raise KernelCompilationError(
                f"Metal xcrun compilation failed (rc={res1.returncode})",
                compiler_log=combined_err,
            )

        res2 = subprocess.run(
            [xcrun_bin, "-sdk", "macosx", "metallib", str(air_path), "-o", str(lib_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if res2.returncode != 0 or not lib_path.is_file():
            raise KernelCompilationError(
                f"Metal metallib linking failed (rc={res2.returncode})",
                compiler_log=f"{res2.stdout}\n{res2.stderr}".strip(),
            )
        return CompiledKernelBinary(
            backend="metal",
            workload=workload,
            entry_point=entry_point,
            format="metallib",
            binary_bytes=lib_path.read_bytes(),
            compile_flags=flags,
            compiler_log=f"{res1.stderr}\n{res2.stderr}".strip(),
        )


def _compile_opencl(
    source_bytes: bytes,
    workload: str,
    entry_point: str,
    flags: tuple[str, ...],
    source_format: str,
) -> CompiledKernelBinary:
    if source_format == "opencl_binary":
        return CompiledKernelBinary(
            backend="opencl",
            workload=workload,
            entry_point=entry_point,
            format="opencl_binary",
            binary_bytes=source_bytes,
            compile_flags=flags,
        )
    return CompiledKernelBinary(
        backend="opencl",
        workload=workload,
        entry_point=entry_point,
        format="opencl_source",
        binary_bytes=source_bytes,
        compile_flags=flags,
    )


def extract_opencl_program_binary(backend: Any, module: Any) -> bytes:
    """Delegate extraction of CL_PROGRAM_BINARIES bytes from a built OpenCL module."""
    return bytes(backend.extract_program_binary(module))


def reload_opencl_program_binary(
    backend: Any,
    ctx: Any,
    workload: str,
    binary_bytes: bytes,
    *,
    entry_point: str | None = None,
) -> Any:
    """Reload cached CL_PROGRAM_BINARIES bytes via clCreateProgramWithBinary."""
    compiled = CompiledKernelBinary(
        backend="opencl",
        workload=workload,
        entry_point=entry_point or default_entry_point(workload),
        format="opencl_binary",
        binary_bytes=binary_bytes,
    )
    return backend.load_module(ctx, compiled)


def compile_kernel_runtime(
    backend: str,
    workload: str,
    target_arch: str = "",
    capabilities: DeviceCapabilities | dict[str, bool] | None = None,
    source_override: bytes | str | None = None,
    extra_flags: Sequence[str] | None = None,
    *,
    driver_version: str = "",
    entry_point: str | None = None,
    cache: RuntimeBinaryCache | None = None,
    registry: dict[str, dict[str, Any]] | None = None,
) -> CompiledKernelBinary:
    """Resolve, compile, and cache a kernel payload for any of the 5 backends."""
    if capabilities is not None and workload in WORKLOAD_SPECS_BY_NAME:
        if not device_supports_workload(capabilities, workload):
            spec = WORKLOAD_SPECS_BY_NAME[workload]
            raise UnsupportedKernelError(
                f"Device lacks capability {spec.required_capability!r} required by {workload!r}"
            )

    resolved_entry = entry_point or default_entry_point(workload)
    source_format = ""
    if source_override is not None:
        raw_bytes = (
            source_override.encode("utf-8")
            if isinstance(source_override, str)
            else bytes(source_override)
        )
    else:
        raw_bytes, meta = get_embedded_kernel(backend, workload, registry=registry)
        source_format = str(meta.get("format", ""))
        resolved_entry = entry_point or str(meta.get("entry_point") or resolved_entry)

    macro_flags = _capability_macros(capabilities)
    combined_flags = tuple([*macro_flags, *(extra_flags or ())])
    source_hash = hashlib.sha256(raw_bytes).hexdigest()
    active_cache = _DEFAULT_CACHE if cache is None else cache
    cache_key = compute_cache_key(
        backend=backend,
        target_arch=target_arch,
        driver_version=driver_version,
        workload=workload,
        source_hash=source_hash,
        compile_flags=combined_flags,
    )

    cached = active_cache.get(cache_key)
    if cached is not None:
        return cached

    if source_format in PRECOMPILED_FORMATS:
        result = CompiledKernelBinary(
            backend=backend,
            workload=workload,
            entry_point=resolved_entry,
            format=source_format,
            binary_bytes=raw_bytes,
            compile_flags=combined_flags,
            cache_key=cache_key,
        )
    elif backend == "cuda":
        result = _compile_cuda_nvrtc(
            raw_bytes, workload, target_arch, resolved_entry, combined_flags
        )
    elif backend == "hip":
        result = _compile_hip_hiprtc(
            raw_bytes, workload, target_arch, resolved_entry, combined_flags
        )
    elif backend == "levelzero":
        result = _compile_levelzero_spirv(
            raw_bytes, workload, target_arch, resolved_entry, combined_flags
        )
    elif backend == "metal":
        result = _compile_metal(raw_bytes, workload, resolved_entry, combined_flags)
    elif backend == "opencl":
        result = _compile_opencl(
            raw_bytes, workload, resolved_entry, combined_flags, source_format
        )
    else:
        raise KernelCompilationError(f"Unsupported backend {backend!r} for runtime compilation")

    result.cache_key = cache_key
    active_cache.put(cache_key, result)
    return result


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m cross_gpu.embed_binary",
        description="Pack, verify, and list embedded kernel payloads.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    pack_p = sub.add_parser("pack", help="Pack kernel files from kernels/<backend>/")
    pack_p.add_argument(
        "--kernels-dir",
        type=Path,
        default=KERNELS_DIR,
        help="Root directory containing per-backend kernel subdirectories",
    )

    sub.add_parser("verify", help="Verify integrity of all embedded kernel payloads")
    sub.add_parser("list", help="List all registered embedded kernels")

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.action == "pack":
        packed = pack_kernels(kernels_dir=args.kernels_dir)
        print(f"Packed {len(packed)} kernel payload(s) from {args.kernels_dir}")
        return ExitCode.SUCCESS

    if args.action == "verify":
        errors = verify_embedded_kernels()
        if errors:
            for err in errors:
                print(f"ERROR: {err}", file=sys.stderr)
            return ExitCode.INIT_ERROR
        print(f"Verified {len(EMBEDDED_KERNELS)} embedded kernel payload(s): OK")
        return ExitCode.SUCCESS

    if args.action == "list":
        if not EMBEDDED_KERNELS:
            print("No embedded kernels registered.")
            return ExitCode.SUCCESS
        for key, entry in sorted(EMBEDDED_KERNELS.items()):
            print(
                f"{key}\tbackend={entry['backend']}\tformat={entry['format']}"
                f"\tsize={entry['size_bytes']}\tsha256={entry['sha256']}"
            )
        return ExitCode.SUCCESS

    return ExitCode.USAGE_ERROR


if __name__ == "__main__":
    sys.exit(_main())
