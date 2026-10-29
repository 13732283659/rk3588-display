# ================= 配置区域 =================
# 交叉编译器前缀 (根据实际安装的工具链修改，如 aarch64-linux-gnu- 或 aarch64-none-linux-gnu-)
CROSS_COMPILE ?= aarch64-linux-gnu-
CC            := $(CROSS_COMPILE)gcc

# Sysroot 路径 (指向包含 ARM 架构头文件和库的根目录)
# 请确保该路径下存在 usr/include/libdrm, usr/include/rga, usr/lib/librockchip_mpp.so 等
# CROSS_COMPILE=/home/gujun/workspace/rk3588_doc/atk_dlrk3588_linux6.12/buildroot/output/alientek_rk3588/host/bin/aarch64-buildroot-linux-gnu-
ROOT_SYSROOT  ?= /home/gujun/workspace/rk3588_doc/atk_dlrk3588_linux6.12/ubuntu/binary

RGA_HOME_DIR ?= /home/gujun/workspace/rk3588_doc/atk_dlrk3588_linux6.12/external/linux-rga

# 目标可执行文件名
TARGET        := rk3588_camera_rga_file

# 源文件
SRCS          := main.c
OBJS          := $(SRCS:.c=.o)

# ================= 编译参数 =================
# C 标准及优化等级
CFLAGS        := -std=c11 -O0 -Wall -Wextra -g

# 头文件路径 (-I)
# 注意：RK 的 rga.h 可能在 include/rga 或 include/linux/rga 下，需根据实际 sysroot 调整
INCLUDES      := -I$(ROOT_SYSROOT)/usr/include \
                 -I$(ROOT_SYSROOT)/usr/include/libdrm \
                 -I$(RGA_HOME_DIR)/build/build_linux/install/include/ 

# 库文件路径 (-L) 及链接库 (-l)
# --sysroot 告诉编译器去哪里找基础系统库 (libc, libpthread 等)
# -lrga: RGA 用户态库 (注意：旧版可能是 librockchip_rga，新版通常是 librga)
# -ldrm: DRM 库，MPP/RGA 零拷贝常需此库
# -lpthread: MPP 多线程通常需要
LDFLAGS       := --sysroot=$(ROOT_SYSROOT) \
				 -L$(ROOT_SYSROOT)/usr/lib \
				 -L$(ROOT_SYSROOT)/usr/lib/aarch64-linux-gnu \
				 $(RGA_HOME_DIR)/build/build_linux/librga.a \
				 -ldrm -lpthread -lm -lstdc++

# ================= 构建规则 =================
all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Linking $@"
	$(CC) $(OBJS) -o $@ $(LDFLAGS)
	@echo "Build success: $(TARGET)"

%.o: %.c
	@echo "Compiling $<"
	$(CC) $(CFLAGS) $(INCLUDES) --sysroot=$(ROOT_SYSROOT) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) pure_dma_1080p.yuv

.PHONY: all clean

