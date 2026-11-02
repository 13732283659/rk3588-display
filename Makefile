# ================= 配置区域 =================
# 交叉编译器前缀 (根据实际安装的工具链修改，如 aarch64-linux-gnu- 或 aarch64-none-linux-gnu-)
CROSS_COMPILE ?= aarch64-linux-gnu-
CC            := $(CROSS_COMPILE)gcc
CXX           := $(CROSS_COMPILE)g++

# Sysroot 路径 (指向包含 ARM 架构头文件和库的根目录)
# 请确保该路径下存在 usr/include/libdrm, usr/include/rga, usr/lib/librockchip_mpp.so 等
# CROSS_COMPILE=/home/gujun/workspace/rk3588_doc/atk_dlrk3588_linux6.12/buildroot/output/alientek_rk3588/host/bin/aarch64-buildroot-linux-gnu-
ROOT_SYSROOT  ?= /home/gujun/workspace/rk3588_doc/atk_dlrk3588_linux6.12/ubuntu/binary

RGA_HOME_DIR ?= /home/gujun/workspace/rk3588_doc/atk_dlrk3588_linux6.12/external/linux-rga
RKNN_ROOT     ?= /home/gujun/workspace/rknn-toolkit2/rknpu2
RKNN_DEMO_DIR := $(RKNN_ROOT)/examples/rknn_yolov5_demo
RKNN_API_DIR  := $(RKNN_ROOT)/runtime/Linux/librknn_api

# 目标可执行文件名
TARGET        := rk3588_camera_rga_file

# 源文件
OBJS          := main.o rknn_infer.o rknn_postprocess.o

# ================= 编译参数 =================
# C 标准及优化等级
CFLAGS        := -std=c11 -O2 -Wall -Wextra -g
CXXFLAGS      := -std=c++11 -O2 -Wall -Wextra -g

# 头文件路径 (-I)
# 注意：RK 的 rga.h 可能在 include/rga 或 include/linux/rga 下，需根据实际 sysroot 调整
INCLUDES      := -I$(ROOT_SYSROOT)/usr/include \
                 -I$(ROOT_SYSROOT)/usr/include/libdrm \
				 -I$(RGA_HOME_DIR)/build/build_linux/install/include/ \
				 -I$(RKNN_API_DIR)/include \
				 -I$(RKNN_DEMO_DIR)/include

# 库文件路径 (-L) 及链接库 (-l)
# --sysroot 告诉编译器去哪里找基础系统库 (libc, libpthread 等)
# -lrga: RGA 用户态库 (注意：旧版可能是 librockchip_rga，新版通常是 librga)
# -ldrm: DRM 库，MPP/RGA 零拷贝常需此库
# -lpthread: MPP 多线程通常需要
LDFLAGS       := --sysroot=$(ROOT_SYSROOT) \
				 -L$(ROOT_SYSROOT)/usr/lib \
				 -L$(ROOT_SYSROOT)/usr/lib/aarch64-linux-gnu \
								 -L$(RKNN_API_DIR)/aarch64 \
				 $(RGA_HOME_DIR)/build/build_linux/librga.a \
								 -ldrm -lrknnrt -lpthread -lm

# ================= 构建规则 =================
all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Linking $@"
	        $(CXX) -g $(OBJS) -o $@ $(LDFLAGS)
	@echo "Build success: $(TARGET)"

%.o: %.c
	@echo "Compiling $<"
	$(CC) $(CFLAGS) $(INCLUDES) --sysroot=$(ROOT_SYSROOT) -c $< -o $@

%.o: %.cpp
	        @echo "Compiling $<"
	        $(CXX) $(CXXFLAGS) $(INCLUDES) --sysroot=$(ROOT_SYSROOT) -c $< -o $@

rknn_postprocess.o: $(RKNN_DEMO_DIR)/src/postprocess.cc
	        @echo "Compiling $<"
	        $(CXX) $(CXXFLAGS) $(INCLUDES) --sysroot=$(ROOT_SYSROOT) -c $< -o $@

main.o: rknn_infer.h
rknn_infer.o: rknn_infer.h $(RKNN_DEMO_DIR)/include/postprocess.h

clean:
	rm -f $(OBJS) $(TARGET) pure_dma_1080p.yuv

.PHONY: all clean

