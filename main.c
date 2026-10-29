

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE

#include <fcntl.h>
#include <linux/dma-buf.h>  // 用于内核 Cache 一致性同步锁
#include <linux/dma-heap.h> // 用于统一的物理连续内存分配
#include <linux/videodev2.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

// 💡 引入标准 DRM/KMS 头文件
#include <drm.h>
#include <drm_fourcc.h>
#include <drm_mode.h>
#include <errno.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

// 引入瑞芯微 RGA 核心头文件
#include "RgaUtils.h"
#include "im2d.h"

#define CAMERA_DEV "/dev/video-camera0"  // 根据实际情况修改
#define DMA_HEAP_DEV "/dev/dma_heap/cma"
#define OUTPUT_FILE "pure_dma_1080p.yuv"
#define TOTAL_FRAMES 1000 // 测试抓取 100 帧

// 输入 4K 规格
#define SRC_WIDTH 3840
#define SRC_HEIGHT 2160

// DSI 面板为竖屏 1080x1920
#define DST_WIDTH 1080
#define DST_HEIGHT 1920

// V4L2 采集端缓冲区数量
#define CAM_BUF_COUNT 2

struct CamBuffer {
  int fd;      // 导出的 Camera DMA-BUF 文件描述符
  size_t size; // 内存长度
  void *vaddr; // 用户态虚拟地址
};

struct DRMContext {
  int drm_fd;
  uint32_t crtc_id;
  uint32_t conn_id;
  drmModeModeInfo mode;

  // 双缓冲显存管理
  uint32_t fb_id[2];
  int gem_handle[2];
  int dma_buf_fd[2]; // 供 RGA 写入的显存 FD
  void *vaddr[2];    // 存放映射后的虚拟地址，供 CPU 访问
  int size;
  uint32_t pitch[2];
};

struct GlobalDMABuffer {
  int fd;
  size_t size;
  void *vaddr;
};

struct CamBuffer g_cam_bufs[CAM_BUF_COUNT];
int g_v4l2_fd = -1;
struct DRMContext g_drm;

// 💡 调优规范：封装一个标准的内核 Cache 同步控制函数
int sync_cpu_cache(int buf_fd, uint64_t sync_flags) {
  struct dma_buf_sync sync = {.flags = sync_flags};
  return ioctl(buf_fd, DMA_BUF_IOCTL_SYNC, &sync);
}

static void page_flip_handler(int fd, unsigned int sequence,
                              unsigned int tv_sec, unsigned int tv_usec,
                              void *user_data) {
  int *flip_done = user_data;
  (void)fd;
  (void)sequence;
  (void)tv_sec;
  (void)tv_usec;
  *flip_done = 1;
}

static int wait_for_page_flip(int drm_fd, int *flip_done) {
  struct pollfd pfd = {.fd = drm_fd, .events = POLLIN};
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

// 通过 DMA-HEAP 分配连续内存
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

  if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &data) < 0) {
    perror("中央内存池物理分配失败");
    close(heap_fd);
    return -1;
  }

  out_buf->fd = data.fd;
  out_buf->size = size;

  out_buf->vaddr =
      mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, data.fd, 0);
  if (out_buf->vaddr == MAP_FAILED) {
    perror("全局核心内存 mmap 映射失败");
    close(data.fd);
    close(heap_fd);
    return -1;
  }

  close(heap_fd);
  return 0;
}

// 初始化 V4L2 摄像头
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
    buf.memory = V4L2_MEMORY_DMABUF;
    buf.index = i;
    buf.length = 1;
    buf.m.planes = &planes;

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

// 初始化 DRM 显示器
int init_drm_display(struct DRMContext *ctx) {
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
    drmModeConnectorPtr conn =
        drmModeGetConnector(ctx->drm_fd, res->connectors[i]);
    if (!conn)
      continue;

    if (conn->connection == DRM_MODE_CONNECTED && conn->count_modes > 0) {
      ctx->conn_id = conn->connector_id;
      ctx->mode = conn->modes[0];

      if (conn->encoder_id) {
        drmModeEncoderPtr enc =
            drmModeGetEncoder(ctx->drm_fd, conn->encoder_id);
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

// DRM 作为生产者分配 Dumb 显存并注册 FB2
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

  if (drmPrimeHandleToFD(ctx->drm_fd, create_req.handle, DRM_CLOEXEC | DRM_RDWR,
                         &ctx->dma_buf_fd[idx]) < 0) {
    perror("❌ 错误：显存转化为 DMA-BUF FD 失败");
    return -1;
  }
  ctx->vaddr[idx] = mmap(NULL, create_req.size, PROT_READ | PROT_WRITE,
                         MAP_SHARED, ctx->dma_buf_fd[idx], 0);
  if (!ctx->vaddr[idx]) {
    perror("❌ 错误：显存映射失败");
    return -1;
  }

  ctx->size = create_req.size;
  printf("分配的显存大小: %u 字节\n", create_req.size);

  uint32_t pitches[4] = {create_req.pitch, create_req.pitch, 0, 0};
  uint32_t offsets[4] = {0, create_req.pitch * DST_HEIGHT, 0, 0};
  uint32_t handles[4] = {create_req.handle, create_req.handle, 0, 0};

  int ret = drmModeAddFB2(ctx->drm_fd, DST_WIDTH, DST_HEIGHT, DRM_FORMAT_NV12,
                          handles, pitches, offsets, &ctx->fb_id[idx], 0);
  if (ret < 0) {
    int err_code = errno;
    fprintf(stderr, "❌ drmModeAddFB2 失败: %s\n", strerror(err_code));
    return -err_code;
  }
  return 0;
}

int main(int argc, char *argv[]) {
  FILE *fp = fopen(OUTPUT_FILE, "ab");
  if (fp == NULL) {
    perror("fopen failed");
    return -1;
  }

  // 1. 点亮 DRM 显示架构并初始化双缓冲显存
  if (init_drm_display(&g_drm) < 0)
    return -1;
  if (create_drm_framebuffer(&g_drm, 0) < 0 ||
      create_drm_framebuffer(&g_drm, 1) < 0)
    return -1;

  if (drmModeSetCrtc(g_drm.drm_fd, g_drm.crtc_id, g_drm.fb_id[0], 0, 0,
                     &g_drm.conn_id, 1, &g_drm.mode) < 0) {
    perror("DRM 初始 modeset 失败");
    return -1;
  }

  // [STEP-1] 大管家最先登场，在外部物理划分输入端的 DMA-BUF
  size_t src_size = SRC_WIDTH * SRC_HEIGHT * 3 / 2; // 4K NV12
  if (allocate_from_central_heap(src_size, &g_cam_bufs[0]) < 0 ||
      allocate_from_central_heap(src_size, &g_cam_bufs[1]) < 0) {
    printf("全局中央统一 DMA-BUF 缓冲区池分配失败!\n");
    return -1;
  }

  // [STEP-2] 初始化 Camera 消费者身份
  if (init_v4l2() < 0)
    return -1;

  // [STEP-3] 规范 Stride 参数对齐
  int src_wstride = (SRC_WIDTH + 15) & ~15;
  int src_hstride = (SRC_HEIGHT + 1) & ~1;
  int dst_hstride = DST_HEIGHT;

  // [STEP-4] 统一将两端的固定外部 fd 提前导入并持久化包装进 RGA 管理体系中
  im_handle_param_t src_param = {SRC_WIDTH, SRC_HEIGHT, RK_FORMAT_YCbCr_420_SP};
  im_handle_param_t dst_param = {DST_WIDTH, DST_HEIGHT, RK_FORMAT_YCbCr_420_SP};

  rga_buffer_handle_t rga_dst_handle[2];
  rga_buffer_t rga_dst_img[2];
  for (int i = 0; i < 2; i++) {
    rga_dst_handle[i] = importbuffer_fd(g_drm.dma_buf_fd[i], &dst_param);
    if (rga_dst_handle[i] == 0) {
      printf("❌ 致命错误：RGA 导入 DRM 显存 FD 失败！\n");
      return -1;
    }

    rga_dst_img[i] = wrapbuffer_handle(rga_dst_handle[i], DST_WIDTH, DST_HEIGHT,
                                       RK_FORMAT_YCbCr_420_SP);
    rga_dst_img[i].wstride = g_drm.pitch[i];
    rga_dst_img[i].hstride = dst_hstride;
  }
  // =========================================================================
  // 💡 核心调优：在进入循环前（视频流还未启动时），一次性在 2
  // 个显存中央提前抹红 彻底解耦多硬件异步时间差，完美对齐 Stride
  // 寻址，解决前几十帧红点丢失的隐患
  // =========================================================================
  for (int idx = 0; idx < 2; idx++) {
    uint8_t *static_vaddr = (uint8_t *)g_drm.vaddr[idx];
    uint32_t static_stride = g_drm.pitch[idx];
    uint8_t *static_uv = static_vaddr + (static_stride * DST_HEIGHT);
    int red_dot_count = 0;
    int target_y = DST_HEIGHT / 2; // 选在竖屏的正中间行
    int start_x = DST_WIDTH / 4;   // 从屏幕水平 1/4 处开始排布
    for (int x_offset = 0; x_offset < 500; x_offset += 100) {
      if (red_dot_count < 5) {
        int cx = start_x + x_offset;
        int cy = target_y;
        // 强制将红点 YUV 数值写入物理显存的有效显示区
        static_vaddr[cy * static_stride + cx] = 0x4C;                  // Y=76
        static_uv[(cy / 2) * static_stride + (cx / 2) * 2] = 0x54;     // U=84
        static_uv[(cy / 2) * static_stride + (cx / 2) * 2 + 1] = 0xFF; // V=255
        red_dot_count++;
      }
    }
    // 冲刷初始 Cache，确保静态数据完全同步到 DDR
    sync_cpu_cache(g_drm.dma_buf_fd[idx],
                   DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE);
    sync_cpu_cache(g_drm.dma_buf_fd[idx],
                   DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
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
    buf.memory = V4L2_MEMORY_DMABUF;
    buf.length = 1;
    buf.m.planes = &planes;
    if (ioctl(g_v4l2_fd, VIDIOC_DQBUF, &buf) < 0) {
      perror("DQBUF 错误");
      break;
    }
    int active_src_fd = g_cam_bufs[buf.index].fd;
                 // ================= 🧪 注入：【4K 相机源端·逆向几何对齐】Cache 脏数据与屏幕红点实验场 =================
        // 💡 调优核心 1：在 CPU 动笔改写 4K 连续堆内存前，必须拉起内核【写开始锁】，确保硬件感知

        //这个是刷新缓存的操作，确保CPU对内存的修改能够被其他硬件看到
        sync_cpu_cache(active_src_fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE);

        uint8_t *cam_vaddr = (uint8_t *)g_cam_bufs[buf.index].vaddr;
        size_t cam_bytes = g_cam_bufs[buf.index].size; 
        
        // 定位符合 4K 相机内核拓扑的 UV 平面起始地址 (Y 占满了 src_wstride * src_hstride)
        uint8_t *cam_uv_plane = cam_vaddr + (src_wstride * src_hstride);
        
        // ⚡ 强行污染 CPU Cache 缓存行 (保持空读，强行把这块内存挂在 Cache 上)
        volatile uint8_t dummy_read = 0;
        for (size_t p = 0; p < cam_bytes; p += 64) {
            dummy_read = cam_vaddr[p]; 
        }

        // ⚡ 精准逆向二阶阵列涂写：在 4K（3840x2160）原始底图的【顶部中央区域】画 5 个纯红点！
        // 💡 深度推导：只有将红点配置在 4K 的顶部（如第 200 行），经过 RGA 旋转 90 度后，才会变成竖屏的左右正中行！
        int red_dot_count = 0;
        int cam_target_y = SRC_HEIGHT / 4;  // 设在 4K 图像的顶部 1/4 处 (第 540 行左右)
        int cam_start_x  = SRC_WIDTH / 2;   // 设在 4K 图像的水平正中央 (第 1920 列左右)

        for (int y = 200; y < SRC_HEIGHT - 200; y += 40) {
            for (int x = 200; x < SRC_WIDTH - 200; x++) {
                
                // 💡 调优核心：只要满足网格节拍，就无条件强制染色，封杀一切缩放裁剪死区！
                if ((y % 40 == 0) || (x % 40 == 0)) {
                    // 写入 4K 源端 Y 平面
                    cam_vaddr[y * src_wstride + x] = 0x4C; // Y=76
                    
                    // 写入 4K 源端 UV 平面
                    int uv_x = (x / 2) * 2;
                    int uv_y = y / 2;
                    size_t uv_offset = (uv_y * src_wstride) + uv_x;
                    
                    cam_uv_plane[uv_offset]     = 0x54; // U=84
                    cam_uv_plane[uv_offset + 1] = 0xFF; // V=255
                }
            }
        }

        // 💡 调优核心 2：CPU 涂写红点结束，必须立刻拉起内核【写结束锁】！！！
        // ⚡ 内核收到该指令后，会强制让八核 ARM 打开缓存闸门，将憋在 Cache 里的 5 个红点像素
        // 瞬间同步、冲刷（Flush）进物理 DDR 内存条中。如果注释掉这行，屏幕上红点绝对会闪烁或消失。
        sync_cpu_cache(active_src_fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);

        // --------------------------------------------------------------------------------
        // 💡 读锁包裹：把这帧被 CPU 抹了红点的 4K 原始数据直接落盘抓包
        //sync_cpu_cache(active_src_fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
        if (fp) {
            size_t written = fwrite(cam_vaddr, 1, cam_bytes, fp);
        }
        //sync_cpu_cache(active_src_fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
        // ==============================================================
        // ==============================================================
    // 翻转指针，定位隐藏的后台显存
    drm_idx = !drm_idx;
    rga_buffer_handle_t rga_src_handle =
        importbuffer_fd(active_src_fd, &src_param);
    rga_buffer_t rga_src_img = wrapbuffer_handle(
        rga_src_handle, SRC_WIDTH, SRC_HEIGHT, RK_FORMAT_YCbCr_420_SP);
    rga_src_img.wstride = src_wstride;
    rga_src_img.hstride = src_hstride;
    // 💡 调试打印：首帧通过 imcheck 验证格式
    if (i == 0) {
      IM_STATUS aligned_status = imcheck(rga_src_img, rga_dst_img[drm_idx],
                                         (im_rect){0}, (im_rect){0});
      printf("Aligned buffers: src wstride=%d hstride=%d, dst wstride=%d "
             "hstride=%d, imcheck=%d (%s)\n",
             rga_src_img.wstride, rga_src_img.hstride,
             rga_dst_img[drm_idx].wstride, rga_dst_img[drm_idx].hstride,
             aligned_status, imStrError(aligned_status));
    }
    // [BLOCK-3] 驱动 RGA 硬件执行【4K 相机 FD】到【1080P 显存
    // FD】的硬件级旋转与缩放
    int ret = imrotate(rga_src_img, rga_dst_img[drm_idx], IM_HAL_TRANSFORM_ROT_90);
    if (ret != IM_STATUS_SUCCESS) {
      printf("RGA 纯硬件像素缩放失败: %s\n", imStrError((IM_STATUS)ret));
      releasebuffer_handle(rga_src_handle);
      break;
    }
    // 原子级翻页送显
    flip_done = 0;
    if (drmModePageFlip(g_drm.drm_fd, g_drm.crtc_id, g_drm.fb_id[drm_idx],
                        DRM_MODE_PAGE_FLIP_EVENT, &flip_done) < 0) {
      flip_done = 1;
      perror("显卡 PageFlip 翻页失败");
      releasebuffer_handle(rga_src_handle);
      break;
    }
    releasebuffer_handle(rga_src_handle);
    if (ioctl(g_v4l2_fd, VIDIOC_QBUF, &buf) < 0) {
      perror("循环 QBUF 失败");
      break;
    }
  }
  printf("流转结束，开始优雅销毁并归还系统显示主权...\n");
  if (!flip_done && wait_for_page_flip(g_drm.drm_fd, &flip_done) < 0)
    perror("退出前等待 DRM PageFlip 完成失败");
  enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  ioctl(g_v4l2_fd, VIDIOC_STREAMOFF, &type);
  if (rga_dst_handle[0])
    releasebuffer_handle(rga_dst_handle[0]);
  if (rga_dst_handle[1])
    releasebuffer_handle(rga_dst_handle[1]);
  for (int i = 0; i < 2; i++) {
    close(g_drm.dma_buf_fd[i]);
    struct drm_mode_destroy_dumb destroy_req = {.handle = g_drm.gem_handle[i]};
    ioctl(g_drm.drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy_req);
  }
  for (int i = 0; i < CAM_BUF_COUNT; i++)
    close(g_cam_bufs[i].fd);
  close(g_v4l2_fd);
  drmClose(g_drm.drm_fd);
  fclose(fp);
  printf("全链路资源清理完毕，程序安全退出。\n");
  return 0;
}