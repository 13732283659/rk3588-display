#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <linux/dma-buf.h>   // 用于内核 Cache 一致性同步锁
#include <linux/dma-heap.h>  // 用于统一的物理连续内存大管家分配

// 引入瑞芯微 RGA 核心头文件
#include "im2d.h"
#include "RgaUtils.h"

#define CAMERA_DEV      "/dev/video-camera0"             // 根据板子实际挂载的节点修改
#define DMA_HEAP_DEV    "/dev/dma_heap/cma" // RK3588 统一连续内存堆节点
#define OUTPUT_FILE     "pure_dma_1080p.yuv"
#define TOTAL_FRAMES    1000                       // 测试抓取 100 帧

// 输入 4K 规格
#define SRC_WIDTH       3840
#define SRC_HEIGHT      2160

// 输出 1080P 规格
#define DST_WIDTH       1920
#define DST_HEIGHT      1080

// 全局中央统一控制块
struct GlobalDMABuffer {
    int fd;         // 内核标准的 DMA-BUF 文件描述符
    size_t size;    // 物理分配尺寸
    void *vaddr;    // 供 CPU 访问、落盘映射的用户态虚拟地址
};

// 分别为源端（Camera输入）和目的端（RGA缩放输出）在最开头通过 DMA-HEAP 圈地
struct GlobalDMABuffer g_src_pool; 
struct GlobalDMABuffer g_dst_pool; 

int g_v4l2_fd = -1;
FILE *g_out_fp = NULL;

// 🛠 辅助函数：跨越所有硬件屏障的统一 DMA-HEAP 物理连续内存开辟逻辑
int allocate_from_central_heap(size_t size, struct GlobalDMABuffer *out_buf) {
    int heap_fd = open(DMA_HEAP_DEV, O_RDWR | O_CLOEXEC);
    if (heap_fd < 0) {
        perror("打开中央 DMA-HEAP 节点失败");
        return -1;
    }

    struct dma_heap_allocation_data data;
    memset(&data, 0, sizeof(data));
    data.len = size;
    data.fd_flags = O_CLOEXEC | O_RDWR;
    data.heap_flags = 0;

    // 向内核分配器申请物理空间
    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &data) < 0) {
        perror("中央内存池物理分配失败");
        close(heap_fd);
        return -1;
    }

    out_buf->fd = data.fd;
    out_buf->size = size;
    
    // 映射出 CPU 虚拟地址，专门给后续的 CPU 写入文件（fwrite）使用
    out_buf->vaddr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, data.fd, 0);
    if (out_buf->vaddr == MAP_FAILED) {
        perror("全局核心内存 mmap 映射失败");
        close(data.fd);
        close(heap_fd);
        return -1;
    }

    close(heap_fd); // 内存句柄 data.fd 已建立，可安全关闭 heap 节点描述符
    return 0;
}

// 1. 初始化 V4L2 摄像头 (彻底蜕变为纯正的 DMABUF 消费者模式)
int init_pure_v4l2_dmabuf() {
    g_v4l2_fd = open(CAMERA_DEV, O_RDWR);
    if (g_v4l2_fd < 0) {
        perror("打开摄像头设备失败");
        return -1;
    }

    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmt.fmt.pix_mp.width = SRC_WIDTH;
    fmt.fmt.pix_mp.height = SRC_HEIGHT;
    fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
    fmt.fmt.pix_mp.field = V4L2_FIELD_NONE;
    fmt.fmt.pix_mp.num_planes = 1;

    if (ioctl(g_v4l2_fd, VIDIOC_S_FMT, &fmt) < 0) {
        perror("设置 V4L2 格式失败");
        return -1;
    }

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 1; // 简化为单个中央全局环形项，验证纯正 DMABUF 逻辑
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    // 💡 核心改变 1：声明 V4L2 为 DMABUF 消费者。驱动不占内存，只坐等外面送券送 fd 进来
    req.memory = V4L2_MEMORY_DMABUF; 

    if (ioctl(g_v4l2_fd, VIDIOC_REQBUFS, &req) < 0) {
        perror("申请 V4L2 DMABUF 队列失败");
        return -1;
    }

    // 💡 核心改变 2：在开启视频流前，首轮 QBUF 就必须把我们从大管家那里分配来的 4K fd 强行借给 Camera
    struct v4l2_buffer buf;
    struct v4l2_plane planes;
    memset(&buf, 0, sizeof(buf));
    memset(&planes, 0, sizeof(planes));

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    buf.memory = V4L2_MEMORY_DMABUF; // 保持一致，坚决抵制任何 MMAP 字眼
    buf.index = 0;
    buf.length = 1;
    buf.m.planes = &planes;
    
    // 把通用的外部生产的输入端 fd 挂载进驱动结构体
    buf.m.planes[0].m.fd = g_src_pool.fd;
    buf.m.planes[0].length = g_src_pool.size;

    if (ioctl(g_v4l2_fd, VIDIOC_QBUF, &buf) < 0) {
        perror("首轮外部 DMABUF 绑定 QBUF 失败");
        return -1;
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (ioctl(g_v4l2_fd, VIDIOC_STREAMON, &type) < 0) {
        perror("开启视频流失败");
        return -1;
    }

    return 0;
}

int main(int argc, char *argv[]) {

    int ret = 0;
  
    g_out_fp = fopen(OUTPUT_FILE, "wb");
    if (!g_out_fp) {
        perror("创建输出文件失败");
        return -1;
    }

    // [STEP-1] 大管家最先登场，在外部物理划分两片绝对连续独立的 DMA-BUF
    size_t src_size = SRC_WIDTH * SRC_HEIGHT * 3 / 2; // 4K NV12
    size_t dst_size = DST_WIDTH * DST_HEIGHT * 3 / 2; // 1080P NV12
    
    if (allocate_from_central_heap(src_size, &g_src_pool) < 0 ||
        allocate_from_central_heap(dst_size, &g_dst_pool) < 0) {
        printf("全局中央统一 DMA-BUF 缓冲区池分配失败!\n");
        return -1;
    }

    // [STEP-2] 初始化 Camera，强制其以纯正消费者身份去读 g_src_pool.fd
    if (init_pure_v4l2_dmabuf() < 0) return -1;

    // [STEP-3] 规范 Stride 参数对齐，彻底封杀硬件级跨度报错
    int src_wstride = (SRC_WIDTH + 15) & ~15;
    int src_hstride = (SRC_HEIGHT + 1) & ~1;
    int dst_wstride = (DST_WIDTH + 15) & ~15;
    int dst_hstride = (DST_HEIGHT + 1) & ~1;

    // [STEP-4] 统一将两端的固定外部 fd 提前导入并持久化包装进 RGA 管理体系中
    // 💡 优化调优：由于 fd 在外部完全固定，整个程序生命周期只需 import 一次，免去循环内高频操作引起的页表振荡
    im_handle_param_t src_param = {SRC_WIDTH, SRC_HEIGHT, RK_FORMAT_YCbCr_420_SP};
    im_handle_param_t dst_param = {DST_WIDTH, DST_HEIGHT, RK_FORMAT_YCbCr_420_SP};
    rga_buffer_handle_t rga_src_handle = importbuffer_fd(g_src_pool.fd, &src_param);
    rga_buffer_handle_t rga_dst_handle = importbuffer_fd(g_dst_pool.fd, &dst_param);
    if (rga_src_handle == 0 || rga_dst_handle == 0) {
        printf("RGA 统一导入全局通道失败!\n");
        return -1;
    }

    rga_buffer_t rga_src_img = wrapbuffer_handle(rga_src_handle, SRC_WIDTH, SRC_HEIGHT, RK_FORMAT_YCbCr_420_SP);
    rga_src_img.wstride = src_wstride; rga_src_img.hstride = src_hstride;

    rga_buffer_t rga_dst_img = wrapbuffer_handle(rga_dst_handle, DST_WIDTH, DST_HEIGHT, RK_FORMAT_YCbCr_420_SP);
    rga_dst_img.wstride = dst_wstride; rga_dst_img.hstride = dst_hstride;

    printf("💡【绝对零拷贝·全局中央统一 DMA-BUF 闭环管线】启动...\n");

    for (int i = 0; i < TOTAL_FRAMES; i++) {
        struct v4l2_buffer buf;
        struct v4l2_plane planes;
        memset(&buf, 0, sizeof(buf));
        memset(&planes, 0, sizeof(planes));

        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory = V4L2_MEMORY_DMABUF; // 💡 从头到尾，保持纯正的 DMABUF 交互特征
        buf.length = 1;
        buf.m.planes = &planes;

        // [BLOCK-1] DQBUF：等待摄像头硬件把最新图像强行以 DMA 方式写到我们的外部 g_src_pool.fd 上
        if (ioctl(g_v4l2_fd, VIDIOC_DQBUF, &buf) < 0) {
            perror("DQBUF 错误");
            break;
        }

        // [BLOCK-2] 硬件极速越过：驱动 RGA 硬件执行【外部源 fd】到【外部目的 fd】的硬件缩放转换
        // 此时由于没有循环内部的 import/release 损耗，RGA 响应速度达到极限
        ret = imresize(rga_src_img, rga_dst_img);
        if (ret != IM_STATUS_SUCCESS) {
            printf("RGA 纯物理级运算失败: %s\n", imStrError((IM_STATUS)ret));
            break;
        }

        // [BLOCK-3] 越微核心调优落地：由于 CPU 马上要在下一步通过指针读取 g_dst_pool.vaddr 并写入文件，
        // 必须触发针对目的端全局 fd 的内核 Cache 作废锁，阻断一切缓存不一致导致的花屏和数据空洞。
        struct dma_buf_sync sync_start = { .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ };
        ioctl(g_dst_pool.fd, DMA_BUF_IOCTL_SYNC, &sync_start);

        // [BLOCK-4] CPU 接入：将全硬件闭环流转出来的 1080P 显存画面快速写入本地文件
        #if 0
        if (g_out_fp && g_dst_pool.vaddr) {
            fwrite(g_dst_pool.vaddr, 1, dst_size, g_out_fp);
        }
        #endif

        // CPU 读取动作结束，利索释放内核 Cache 锁
        struct dma_buf_sync sync_end = { .flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ };
        ioctl(g_dst_pool.fd, DMA_BUF_IOCTL_SYNC, &sync_end);

        // [BLOCK-5] QBUF：把当前这一帧外部 fd 重新塞给 Camera 驱动结构体，告知驱动你可以去覆盖下一帧画面了
        buf.m.planes[0].m.fd = g_src_pool.fd;
        buf.m.planes[0].length = g_src_pool.size;

        if (ioctl(g_v4l2_fd, VIDIOC_QBUF, &buf) < 0) {
            perror("循环 QBUF 失败");
            break;
        }

        if ((i + 1) % 10 == 0) {
            printf("[中央全闭环体系] 已成功流转、计算并落盘 %d 帧...\n", i + 1);
        }
    }

    printf("全链条运行完结，优雅释放全局中央池资源...\n");

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    ioctl(g_v4l2_fd, VIDIOC_STREAMOFF, &type);

    // 回收 RGA 通道持有的持久化引用
    if (rga_src_handle) releasebuffer_handle(rga_src_handle);
    if (rga_dst_handle) releasebuffer_handle(rga_dst_handle);
    
    // 回收大管家申请的源端、目的端全局物理缓冲区
    if (g_src_pool.vaddr && g_src_pool.vaddr != MAP_FAILED) munmap(g_src_pool.vaddr, g_src_pool.size);
    if (g_src_pool.fd >= 0) close(g_src_pool.fd);
    
    if (g_dst_pool.vaddr && g_dst_pool.vaddr != MAP_FAILED) munmap(g_dst_pool.vaddr, g_dst_pool.size);
    if (g_dst_pool.fd >= 0) close(g_dst_pool.fd);
    
    close(g_v4l2_fd);
    if (g_out_fp) fclose(g_out_fp);

    return 0;
}
