#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <poll.h>
#include <linux/videodev2.h>
#include <linux/dma-buf.h>   // 用于内核 Cache 一致性同步锁
#include <linux/dma-heap.h>  // 用于统一的物理连续内存大管家分配






// 💡 引入标准 DRM/KMS 头文件
#include <drm_fourcc.h>
#include <drm_mode.h>
#include <drm.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <errno.h>

// 引入瑞芯微 RGA 核心头文件
#include "im2d.h"
#include "RgaUtils.h"

#define CAMERA_DEV      "/dev/video-camera0"             // 根据板子实际挂载的节点修改
#define DMA_HEAP_DEV    "/dev/dma_heap/cma" // RK3588 统一连续内存堆节点
#define OUTPUT_FILE     "pure_dma_1080p.yuv"
#define TOTAL_FRAMES    100                      // 测试抓取 100 帧

// 输入 4K 规格
#define SRC_WIDTH       3840
#define SRC_HEIGHT      2160

// DSI 面板为竖屏 1080x1920
#define DST_WIDTH       1080
#define DST_HEIGHT      1920


// V4L2 采集端缓冲区数量
#define CAM_BUF_COUNT   2
struct CamBuffer {
    int fd;         // 导出的 Camera DMA-BUF 文件描述符
    size_t size;    // 内存长度
    void *vaddr;    // 供 CPU 访问、落盘映射的用户态虚拟地址
};

struct DRMContext {
    int drm_fd;
    uint32_t crtc_id;
    uint32_t conn_id;
    drmModeModeInfo mode;
    
    // 双缓冲显存管理
    uint32_t fb_id[2];
    int gem_handle[2];
    int dma_buf_fd[2];   // 💡 核心调优：供 RGA 写入的显存 FD
    void *vaddr[2];     //存放映射后的虚拟地址，供 CPU 访问
    int size;
    uint32_t pitch[2];
};


// 💡 调优规范：封装一个标准的内核 Cache 同步控制函数
int sync_cpu_cache(int buf_fd, uint64_t sync_flags) {
    struct dma_buf_sync sync = { .flags = sync_flags };
    return ioctl(buf_fd, DMA_BUF_IOCTL_SYNC, &sync);
}

static void page_flip_handler(int fd, unsigned int sequence, unsigned int tv_sec,
                              unsigned int tv_usec, void *user_data) {
    int *flip_done = user_data;
    (void)fd;
    (void)sequence;
    (void)tv_sec;
    (void)tv_usec;
    *flip_done = 1;
}

static int wait_for_page_flip(int drm_fd, int *flip_done) {
    struct pollfd pfd = { .fd = drm_fd, .events = POLLIN };
    drmEventContext event_ctx;
    memset(&event_ctx, 0, sizeof(event_ctx));
    event_ctx.version = DRM_EVENT_CONTEXT_VERSION;
    event_ctx.page_flip_handler = page_flip_handler;

    while (!*flip_done) {
        int ret = poll(&pfd, 1, 2000);
        if (ret < 0 && errno == EINTR)
            continue;
        if (ret <= 0) {
            if (ret == 0)
                errno = ETIMEDOUT;
            return -1;
        }
        if (drmHandleEvent(drm_fd, &event_ctx) != 0)
            return -1;
    }
    return 0;
}

// 全局中央统一控制块
struct GlobalDMABuffer {
    int fd;         // 内核标准的 DMA-BUF 文件描述符
    size_t size;    // 物理分配尺寸
    void *vaddr;    // 供 CPU 访问、落盘映射的用户态虚拟地址
};

// 分别为源端（Camera输入）和目的端（RGA缩放输出）在最开头通过 DMA-HEAP 圈地
struct GlobalDMABuffer g_src_pool; 
//struct GlobalDMABuffer g_dst_pool; 

// 全局上下文
struct CamBuffer g_cam_bufs[CAM_BUF_COUNT];
int g_v4l2_fd = -1;
struct DRMContext g_drm;

// 🛠 辅助函数：跨越所有硬件屏障的统一 DMA-HEAP 物理连续内存开辟逻辑
int allocate_from_central_heap(size_t size, struct CamBuffer *out_buf) {
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

// 1. 初始化 V4L2 摄像头并导出 DMA-BUF FDs
int init_v4l2() {
    g_v4l2_fd = open(CAMERA_DEV, O_RDWR);
    if (g_v4l2_fd < 0) {
        perror("❌ 错误：打开摄像头设备失败");
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
        perror("❌ 错误：设置 V4L2 格式失败");
        return -1;
    }

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = CAM_BUF_COUNT;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    req.memory = V4L2_MEMORY_DMABUF; 

    if (ioctl(g_v4l2_fd, VIDIOC_REQBUFS, &req) < 0) {
        perror("❌ 错误：申请 V4L2 队列失败");
        return -1;
    }

    for (int i = 0; i < CAM_BUF_COUNT; i++) {
        struct v4l2_buffer buf;
        struct v4l2_plane planes;
        memset(&buf, 0, sizeof(buf));
        memset(&planes, 0, sizeof(planes));

        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory = V4L2_MEMORY_DMABUF; // 保持一致，坚决抵制任何 MMAP 字眼
        buf.index = i;
        buf.length = 1;
        buf.m.planes = &planes;

        // 把通用的外部生产的输入端 fd 挂载进驱动结构体
        buf.m.planes[0].m.fd = g_cam_bufs[i].fd;
        buf.m.planes[0].length = g_cam_bufs[i].size;

        if (ioctl(g_v4l2_fd, VIDIOC_QBUF, &buf) < 0) {
            perror("首轮外部 DMABUF 绑定 QBUF 失败");
            return -1;
        }
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (ioctl(g_v4l2_fd, VIDIOC_STREAMON, &type) < 0) {
        perror("❌ 错误：开启视频流失败");
        return -1;
    }
    return 0;
}
// 2. 通过 drmOpen 初始化并自动检索 RK3588 显示硬件资源
int init_drm_display(struct DRMContext *ctx) {
    // 💡 通过驱动名称 "rockchip" 自动检索并打开 VOP2 显示引擎
    ctx->drm_fd = drmOpen("rockchip", NULL); 
    if (ctx->drm_fd < 0) {
        printf("❌ 错误：drmOpen 寻找 rockchip 引擎失败！请确认是否加了 sudo。\n");
        return -1;
    }

    drmModeResPtr res = drmModeGetResources(ctx->drm_fd);
    if (!res) {
        printf("❌ 错误：无法获取 DRM 资源（请确认屏幕已物理连接通电）。\n");
        return -1;
    }

    int found_monitor = 0;
    for (int i = 0; i < res->count_connectors; i++) {
        drmModeConnectorPtr conn = drmModeGetConnector(ctx->drm_fd, res->connectors[i]);
        if (!conn) continue;

        if (conn->connection == DRM_MODE_CONNECTED && conn->count_modes > 0) {
            ctx->conn_id = conn->connector_id;
            ctx->mode = conn->modes[0]; // 获取 native 最优分辨率
            
            if (conn->encoder_id) {
                drmModeEncoderPtr enc = drmModeGetEncoder(ctx->drm_fd, conn->encoder_id);
                if (enc) {
                    ctx->crtc_id = enc->crtc_id;
                    drmModeFreeEncoder(enc);
                }
            }
            drmModeFreeConnector(conn);
            found_monitor = 1;
            break;
        }
        drmModeFreeConnector(conn);
    }
    drmModeFreeResources(res);

    if (!found_monitor) {
        printf("❌ 错误：未检测到任何处于激活连接状态的显示器！\n");
        return -1;
    }
    printf("📺 DRM 显示引擎初始化成功。CRTC_ID: %d, 目标屏幕分辨率: %dx%d\n", 
            ctx->crtc_id, ctx->mode.hdisplay, ctx->mode.vdisplay);
    return 0;
}
// 3. DRM 作为生产者分配 Dumb 显存并导出标准 DMA-BUF FD
int create_drm_framebuffer(struct DRMContext *ctx, int idx) {
    struct drm_mode_create_dumb create_req;
    memset(&create_req, 0, sizeof(create_req));
    create_req.width = DST_WIDTH;
    create_req.height = DST_HEIGHT * 3 / 2;
    create_req.bpp = 8;

    if (ioctl(ctx->drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &create_req) < 0) {
        perror("❌ 错误：DRM 物理显存 Dumb 分配失败");
        return -1;
    }
    ctx->gem_handle[idx] = create_req.handle;
    ctx->pitch[idx] = create_req.pitch;

    // ⚡【核心解错行】：将显存转换为可以被 RGA 顺利识别的通用标准的 dma-buf fd
    if (drmPrimeHandleToFD(ctx->drm_fd, create_req.handle, DRM_CLOEXEC | DRM_RDWR, &ctx->dma_buf_fd[idx]) < 0) {
        perror("❌ 错误：显存转化为 DMA-BUF FD 失败");
        return -1;
    }
    ctx->vaddr[idx] = mmap(NULL, create_req.size, PROT_READ | PROT_WRITE, MAP_SHARED, ctx->dma_buf_fd[idx], 0);
    if (!ctx->vaddr[idx]) {
        perror("❌ 错误：显存映射失败");
        return -1;
    }

    ctx->size = create_req.size;
    printf("分配的显存大小: %u 字节\n", create_req.size);

    // 在显卡内核中将该显存注册绑定为标准的 NV12 格式图层
    uint32_t pitches[4] = { create_req.pitch, 0, 0, 0 };
    uint32_t offsets[4] = { 0, 0, 0, 0 };
    uint32_t handles[4] = { create_req.handle, 0, 0, 0 };
    
    pitches[1] = create_req.pitch;
    offsets[1] = create_req.pitch * DST_HEIGHT;
    handles[1] = create_req.handle;
   
    int ret = drmModeAddFB2(ctx->drm_fd, DST_WIDTH, DST_HEIGHT, DRM_FORMAT_NV12, 
                         handles, pitches, offsets, &ctx->fb_id[idx], 0);
    if (ret < 0) {
         // 💡 调优关键：保存当前的 errno，防止接下来的 printf 改变了它
        int err_code = errno; 
        
        fprintf(stderr, "❌ drmModeAddFB2 失败！\n");
        fprintf(stderr, "   - 原始返回值 (ret): %d\n", ret);
        fprintf(stderr, "   - 底层错误码 (errno): %d\n", err_code);
        fprintf(stderr, "   - 具体的错误描述: %s\n", strerror(err_code)); // 💡 转换为文本
        
        // 针对 RK3588 的特殊错误码给出硬核提示
        if (err_code == EINVAL) {
            fprintf(stderr, "   💡 调优提示: 参数不合规。请检查 pitches 跨度对齐、offsets 偏移计算，或格式(NV12)是否填错。\n");
        } else if (err_code == EACCES || err_code == EPERM) {
            fprintf(stderr, "   💡 调优提示: 权限不足。打开 DRM 节点或注册 FB2 必须使用 root 权限(sudo)。\n");
        }
        
        return -err_code; // 返回负的错误码
    }
    return 0;
}


int main(int argc, char *argv[]) {

    FILE *fp = NULL;
    fp = fopen(OUTPUT_FILE, "ab");
     if (fp == NULL) {
        perror("fopen failed"); // 打印错误原因（如权限不足、路径不存在）
        return -1;

    }
    int ret = 0;
    // 1. 点亮 DRM 显示架构并初始化双缓冲显存
    if (init_drm_display(&g_drm) < 0) return -1;
    if (create_drm_framebuffer(&g_drm, 0) < 0 || create_drm_framebuffer(&g_drm, 1) < 0) return -1;
    if (drmModeSetCrtc(g_drm.drm_fd, g_drm.crtc_id, g_drm.fb_id[0], 0, 0,
                       &g_drm.conn_id, 1, &g_drm.mode) < 0) {
        perror("DRM 初始 modeset 失败");
        return -1;
    }
    // [STEP-1] 大管家最先登场，在外部物理划分两片绝对连续独立的 DMA-BUF
    size_t src_size = SRC_WIDTH * SRC_HEIGHT * 3 / 2; // 4K NV12
    //size_t dst_size = DST_WIDTH * DST_HEIGHT * 3 / 2; // 1080P NV12
    
    if (allocate_from_central_heap(src_size, &g_cam_bufs[0]) < 0 ||
        allocate_from_central_heap(src_size, &g_cam_bufs[1]) < 0) {
        printf("全局中央统一 DMA-BUF 缓冲区池分配失败!\n");
        return -1;
    }

    // [STEP-2] 初始化 Camera，强制其以纯正消费者身份去读 g_src_pool.fd
    if (init_v4l2() < 0) return -1;

    // [STEP-3] 规范 Stride 参数对齐，彻底封杀硬件级跨度报错
    int src_wstride = (SRC_WIDTH + 15) & ~15;
    int src_hstride = (SRC_HEIGHT + 1) & ~1;
    int dst_hstride = DST_HEIGHT;

    // [STEP-4] 统一将两端的固定外部 fd 提前导入并持久化包装进 RGA 管理体系中
    // 💡 优化调优：由于 fd 在外部完全固定，整个程序生命周期只需 import 一次，免去循环内高频操作引起的页表振荡
    im_handle_param_t src_param = {SRC_WIDTH, SRC_HEIGHT, RK_FORMAT_YCbCr_420_SP};
    im_handle_param_t dst_param = {DST_WIDTH, DST_HEIGHT, RK_FORMAT_YCbCr_420_SP};

    // 4. 将 DRM 分配持有的 2 个目的端显存永久导入 RGA 缓存池体系
    // 💡 调优：由于目的端固定，在循环外只需 import 一次，彻底抹平循环内映射损耗
    rga_buffer_handle_t rga_dst_handle[2];
    rga_buffer_t rga_dst_img[2];
    for (int i = 0; i < 2; i++) {
        rga_dst_handle[i] = importbuffer_fd(g_drm.dma_buf_fd[i], &dst_param);
        if (rga_dst_handle[i] == 0) {
            printf("❌ 致命错误：RGA 导入 DRM 显存 FD 失败！段错误安全拦截。\n");
            return -1;
        }
        rga_dst_img[i] = wrapbuffer_handle(rga_dst_handle[i], DST_WIDTH, DST_HEIGHT, RK_FORMAT_YCbCr_420_SP);
        rga_dst_img[i].wstride = g_drm.pitch[i];
        rga_dst_img[i].hstride = dst_hstride;
    }

    printf("⚡【全链路零拷贝·Camera ➡ RGA ➡ DRM 双缓冲送显】已平稳启航...\n");
    int drm_idx = 0; // 双缓冲显示交替索引（0 和 1）
    int flip_done = 1;

    for (int i = 0; i < TOTAL_FRAMES; i++) {
        if (!flip_done && wait_for_page_flip(g_drm.drm_fd, &flip_done) < 0) {
            perror("等待 DRM PageFlip 完成失败");
            break;
        }

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
        int active_src_fd = g_cam_bufs[buf.index].fd;

        // 💡 翻转指针：RGA 永远只向目前处于隐藏状态的“后台显存”进行硬件缩放写入
        drm_idx = !drm_idx; 
         // [BLOCK-2] 动态将当前这一帧的 4K 相机输入源导入 RGA 管理器
        rga_buffer_handle_t rga_src_handle = importbuffer_fd(active_src_fd, &src_param);
        rga_buffer_t rga_src_img = wrapbuffer_handle(rga_src_handle, SRC_WIDTH, SRC_HEIGHT, RK_FORMAT_YCbCr_420_SP);
        rga_src_img.wstride = src_wstride;
        rga_src_img.hstride = src_hstride;
         if (i == 0) {
            IM_STATUS aligned_status = imcheck(rga_src_img, rga_dst_img[drm_idx],
                                               (im_rect){0}, (im_rect){0});
             rga_buffer_t misaligned_dst = rga_dst_img[drm_idx];
             misaligned_dst.wstride += 2;
            IM_STATUS misaligned_status = imcheck(rga_src_img, misaligned_dst,
                                                  (im_rect){0}, (im_rect){0});

             printf("RGA stride requirement: %s\n", querystring(RGA_BYTE_STRIDE));
             printf("Aligned buffers: src wstride=%d hstride=%d, dst wstride=%d hstride=%d, imcheck=%d (%s)\n",
                 rga_src_img.wstride, rga_src_img.hstride,
                 rga_dst_img[drm_idx].wstride, rga_dst_img[drm_idx].hstride,
                 aligned_status, imStrError(aligned_status));
             printf("Misaligned test only: dst wstride=%d, imcheck=%d (%s)\n",
                 misaligned_dst.wstride, misaligned_status,
                 imStrError(misaligned_status));
         }
        // [BLOCK-2] 硬件极速越过：驱动 RGA 硬件执行【外部源 fd】到【外部目的 fd】的硬件缩放转换
        // 此时由于没有循环内部的 import/release 损耗，RGA 响应速度达到极限
       // [BLOCK-3] 驱动 RGA 硬件执行【4K 相机 FD】到【1080P 显存 FD】的纯物理零拷贝硬件缩放
        // 此时由于没有目的端的重复建立页表开销，硬件效率极大提升
        ret = imrotate(rga_src_img, rga_dst_img[drm_idx], IM_HAL_TRANSFORM_ROT_90);
        if (ret != IM_STATUS_SUCCESS) {
            printf("RGA 纯硬件像素缩放失败: %s\n", imStrError((IM_STATUS)ret));
            releasebuffer_handle(rga_src_handle);
            break;
        }
         
        // ================= 🧪 注入：纯正的 DRM 目的端显存 Cache 脏数据破坏实验 =================
        // 💡 核心对齐：故意删掉或注释掉任何 sync_cpu_cache 锁！
        
         // 1. 获取当前正在处理的 DRM 显示缓冲区的虚拟地址与硬件真实跨度
        uint8_t *drm_vaddr = (uint8_t *)g_drm.vaddr[drm_idx];
        uint32_t real_stride = g_drm.pitch[drm_idx]; // ⚡ 必须用显卡回填的真实步长
        
        // 定位到符合显卡内核拓扑的 UV 平面起始地址 (Y 占满了 real_stride * DST_HEIGHT)
        uint8_t *uv_plane = drm_vaddr + (real_stride * DST_HEIGHT);
        
        // 2. ⚡ 强行高频污染 CPU Cache 缓存行 (保持不变，空读硬件刚写入的 DDR)
        volatile uint8_t dummy_read = 0;
        size_t frame_bytes = g_drm.size;
        for (size_t p = 0; p < frame_bytes; p += 64) {
            dummy_read = drm_vaddr[p]; 
        }



        // 3. ⚡ 精准二阶阵列涂写：在画面正中央区域，硬生生涂写 5 个【绝对处于有效显示区内】的纯红点
        // 我们不采用一维指针递增，直接用 Y 行和 X 列，强行写入硬件可见区域
        int red_dot_count = 0;
        int target_y = DST_HEIGHT / 2; // 选在竖屏的正中间行 (约 960 行处)
        int start_x  = DST_WIDTH / 4;   // 从屏幕水平 1/4 处开始排布

        for (int x_offset = 0; x_offset < 500; x_offset += 100) {
            if (red_dot_count < 5) {
                int cur_x = start_x + x_offset;
                int cur_y = target_y;

                // 💡 核心调优对齐：严格按照 real_stride 进行换行寻址映射！
                // 写入 Y 平面 (亮度层)
                drm_vaddr[cur_y * real_stride + cur_x] = 0x4C; // Y = 76

                // 写入 UV 平面 (色彩层，NV12 格式下 U 和 V 交错排列，且宽高减半)
                int uv_x_aligned = (cur_x / 2) * 2;
                int uv_y_aligned = cur_y / 2;
                size_t uv_offset = (uv_y_aligned * real_stride) + uv_x_aligned;

                uv_plane[uv_offset]     = 0x54; // U = 84
                uv_plane[uv_offset + 1] = 0xFF; // V = 255

                red_dot_count++;
            }
        }
        
        // 4. ⚡ 强行在缓存极度错乱的状态下，用 CPU 抓取这个目的端显存指针直接 fwrite 导出落盘
        sync_cpu_cache(g_drm.dma_buf_fd[drm_idx], DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
        if (fp) {
            // 💡 调优注意：原代码中第 352 行写入的是相机数据，请把它删掉，改成在这里写入经过 RGA 处理且被污染的 drm_vaddr
            size_t written = fwrite(drm_vaddr, 1, frame_bytes, fp);
            printf("🧪 [Cache破坏成功] 成功在肮脏缓存状态下将 RGA 目的端强行落盘 %zu 字节。\n", written);
        }
        sync_cpu_cache(g_drm.dma_buf_fd[drm_idx], DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
        // ======================================================================
        // [BLOCK-4] 越微性能调优：向 DRM 申请原子级硬件翻页（PageFlip），对齐屏幕 VSync 垂直时序
        // ⚡ 因为此时完全纯硬件流转，CPU 从未读写这块显存，故在此链条中彻底不需要执行耗时的 Cache 刷新锁！
        flip_done = 0;
        if (drmModePageFlip(g_drm.drm_fd, g_drm.crtc_id, g_drm.fb_id[drm_idx],
                            DRM_MODE_PAGE_FLIP_EVENT, &flip_done) < 0) {
            flip_done = 1;
            perror("显卡 PageFlip 翻页失败");
            releasebuffer_handle(rga_src_handle);
            break;
        }

        // 规范清理当前帧相机的临时映射句柄，阻断引用计数溢出
        releasebuffer_handle(rga_src_handle);

        if (ioctl(g_v4l2_fd, VIDIOC_QBUF, &buf) < 0) {
            perror("循环 QBUF 失败");
            break;
        }
    }

    printf("流转结束，开始优雅销毁并归还系统显示主权...\n");
    if (!flip_done && wait_for_page_flip(g_drm.drm_fd, &flip_done) < 0)
        perror("退出前等待 DRM PageFlip 完成失败");
    //关闭视频流
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    ioctl(g_v4l2_fd, VIDIOC_STREAMOFF, &type);
    // 释放 RGA 目的端常驻句柄
    if (rga_dst_handle[0]) releasebuffer_handle(rga_dst_handle[0]);
    if (rga_dst_handle[1]) releasebuffer_handle(rga_dst_handle[1]);
    // 销毁显卡分配的显存
    for (int i = 0; i < 2; i++) {
    close(g_drm.dma_buf_fd[i]);
    struct drm_mode_destroy_dumb destroy_req = { .handle = g_drm.gem_handle[i] };
    ioctl(g_drm.drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy_req);
    }
    // 关闭核心驱动句柄
    for (int i = 0; i < CAM_BUF_COUNT; i++) close(g_cam_bufs[i].fd);
    close(g_v4l2_fd);
    drmClose(g_drm.drm_fd);
     if (fclose(fp) != 0) {
        perror("fclose failed");
    }
    printf("全链路资源清理完毕，程序安全退出。\n");
    return 0;
}
