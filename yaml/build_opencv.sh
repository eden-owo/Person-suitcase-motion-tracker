#!/bin/bash
set -e

pip3 uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true
rm -rf /usr/local/lib/python3.10/dist-packages/cv2 || true

# 取得 NumPy include 路徑
NUMPY_INCLUDE=$(python3 -c "import numpy; print(numpy.get_include())")

# 下載 OpenCV 4.9.0 主 repo
rm -rf /opt/opencv /opt/opencv_contrib
git clone --branch 4.9.0 --depth 1 https://github.com/opencv/opencv.git /opt/opencv
git clone --branch 4.9.0 --depth 1 https://github.com/opencv/opencv_contrib.git /opt/opencv_contrib


# 2) 乾淨重建
cd /opt/opencv
rm -rf build && mkdir build && cd build

PYTHON3_EXEC=$(which python3)
PYTHON3_INCLUDE_DIR=$(python3 -c "from sysconfig import get_paths as gp; print(gp()['include'])")
PYTHON3_PACKAGES_PATH=$(python3 -c "from sysconfig import get_paths as gp; print(gp()['purelib'])")

cmake .. \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
      -D CMAKE_CUDA_ARCHITECTURES=89 \
      -D WITH_CUDA=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=ON \
      -D WITH_CUDNN=ON \
      -D WITH_TBB=ON \
      -D WITH_OPENGL=ON \
      -D WITH_GSTREAMER=ON \
      -D WITH_LIBV4L=ON \
      -D WITH_OPENJPEG=ON \
      -D PYTHON3_EXECUTABLE=$PYTHON3_EXEC \
      -D PYTHON3_INCLUDE_DIR=$PYTHON3_INCLUDE_DIR \
      -D PYTHON3_PACKAGES_PATH=$PYTHON3_PACKAGES_PATH \
      -D PYTHON3_NUMPY_INCLUDE_DIRS=$NUMPY_INCLUDE \
      -D BUILD_EXAMPLES=OFF \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D BUILD_opencv_python3=ON \
      -D BUILD_opencv_world=OFF \
      -D BUILD_LIST=cudev,core,imgproc,imgcodecs,highgui,videoio,calib3d,dnn,cudaarithm,cudaimgproc,cudawarping,cudafilters,python3 \
      -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_DOCS=OFF \
      -D CUDA_NVCC_FLAGS="--expt-relaxed-constexpr" 



# -D WITH_NPP=OFF \
# -D CMAKE_CXX_FLAGS="-DCV_CUDA_HAVE_NPP=0 -DHAVE_NPP=0" \
# -D CMAKE_CUDA_FLAGS="--expt-relaxed-constexpr -DCV_CUDA_HAVE_NPP=0 -DHAVE_NPP=0" \

make -j"$(nproc)"
make install
ldconfig
