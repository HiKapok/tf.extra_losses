CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(extra_losses)

find_package(CUDA REQUIRED)

EXECUTE_PROCESS(COMMAND python3.5 -c "import os; print(os.getcwd(), end='', flush=True)" OUTPUT_VARIABLE CWD)
MESSAGE(STATUS "Found CWD: " ${CWD})

EXECUTE_PROCESS(COMMAND python3.5 -c "import subprocess; process = subprocess.Popen('nvidia-smi -i 0 --query-gpu=name --format=csv'.split(), stdout=subprocess.PIPE); output, _ = process.communicate(); output = str(output); device_capability_map = {
    'Tesla K80'   : '37',
    'Tesla K40'   : '35',
    'Tesla K20'   : '35',
    'Tesla C2075' : '20',
    'Tesla C2050' : '20',
    'Tesla C2070' : '20',
    'Tesla V100'  : '70',
    'Tesla P100'  : '60',
    'Tesla P40'   : '61',
    'Tesla P4'    : '61',
    'Tesla M60'   : '52',
    'Tesla M40'   : '52',
    'Tesla K80'   : '37',
    'Tesla K40'   : '35',
    'Tesla K20'   : '35',
    'Tesla K10'   : '30',
    'GeForce GTX 1080 Ti' : '61'
}; cap = '61';
for k, v in device_capability_map.items():
    if k in output:
        cap = v
        break
print('gencode arch=compute_' + cap + ',code=sm_' + cap)" OUTPUT_VARIABLE GPU_CAPABILITY)
MESSAGE(STATUS "Found GPU_CAPABILITY: " ${GPU_CAPABILITY})

# Pass options to NVCC
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --keep --keep-dir ${CWD} -${GPU_CAPABILITY} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr")

#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --keep --keep-dir ${CWD} -gencode arch=compute_61,code=sm_61 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr")

# compiler flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 ${OpenMP_CXX_FLAGS} -Wall -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -DGOOGLE_CUDA=1")

# TensorFlow dependencies
EXECUTE_PROCESS(COMMAND python3.5 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow as tf; print(tf.sysconfig.get_include(), end='', flush=True)"  OUTPUT_VARIABLE TF_INC)

EXECUTE_PROCESS(COMMAND python3.5 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow as tf; print(tf.sysconfig.get_lib(), end='', flush=True)"  OUTPUT_VARIABLE TF_LIB)


MESSAGE(STATUS "Found TF_INC: " ${TF_INC})
MESSAGE(STATUS "Found TF_INC_EXTERNAL: " ${TF_INC}/external/nsync/public)
MESSAGE(STATUS "Found TF_LIB: " ${TF_LIB})


INCLUDE_DIRECTORIES(${TF_INC})
INCLUDE_DIRECTORIES(${TF_INC}/external/nsync/public)
LINK_DIRECTORIES(${TF_LIB})

# approach 1
# CUDA_ADD_LIBRARY(l_softmax_gpu SHARED l_softmax_op.cu OPTIONS -I$TF_INC/tensorflow/stream_executor/cuda -I/usr/local)

# ADD_LIBRARY(l_softmax SHARED
#   l_softmax_op.h
#   l_softmax_op.cc
#   )

# TARGET_LINK_LIBRARIES(l_softmax tensorflow_framework ${CUDA_LIBRARIES} l_softmax_gpu)


# approach 2
CUDA_COMPILE(PSROI_ALIGN_CU_O l_softmax_op.cu MODULE OPTIONS -I$TF_INC -I/usr/local)
CUDA_COMPILE(PSROI_ALIGN_GRAD_CU_O l_softmax_grad_op.cu MODULE OPTIONS -I$TF_INC -I/usr/local)

ADD_LIBRARY(extra_losses SHARED
  ${PSROI_ALIGN_CU_O}
  ${PSROI_ALIGN_GRAD_CU_O}
  l_softmax_op.h
  l_softmax_op.cc
  l_softmax_grad_op.cc
  common.h
  common.cc
  )

TARGET_LINK_LIBRARIES(extra_losses tensorflow_framework ${CUDA_LIBRARIES})
