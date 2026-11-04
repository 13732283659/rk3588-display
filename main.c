

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE

#include <fcntl.h>
#include <linux/dma-buf.h>  // 用于内核 Cache 一致性同步锁
#include <linux/dma-heap.h> // 用于统一的物理连续内存分配
#include <linux/videodev2.h>
#include <poll.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

// 💡 引入标准 DRM/KMS 头文件
#include <drm.h>
#include <drm/rockchip_drm.h>
#include <drm_fourcc.h>
#include <drm_mode.h>
#include <errno.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

// 引入瑞芯微 RGA 核心头文件
#include "RgaUtils.h"
#include "im2d.h"
#include "rknn_infer.h"

#define CAMERA_DEV "/dev/video-camera0"  // 根据实际情况修改
#define DMA_HEAP_DEV "/dev/dma_heap/cma"
#define OUTPUT_FILE "pure_dma_1080p.yuv"
#define TOTAL_FRAMES 10000 // 测试抓取 100 帧

// 输入 4K 规格
#define SRC_WIDTH 3840
#define SRC_HEIGHT 2160

// DSI 面板为竖屏 1080x1920
#define DST_WIDTH 1080
#define DST_HEIGHT 1920
#define DEFAULT_RKNN_MODEL "./model/RK3588/yolov5s-640-640.rknn"

// V4L2 采集端缓冲区数量
#define CAM_BUF_COUNT 2
#define DRM_BUF_COUNT 3
#define TIMING_REPORT_INTERVAL 100

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
  uint32_t fb_id[DRM_BUF_COUNT];
  int gem_handle[DRM_BUF_COUNT];
  int dma_buf_fd[DRM_BUF_COUNT]; // 供 RGA 写入的显存 FD
  void *vaddr[DRM_BUF_COUNT];    // 存放映射后的虚拟地址，供 CPU 访问
  int size;
  uint32_t pitch[DRM_BUF_COUNT];
};

struct GlobalDMABuffer {
  int fd;
  size_t size;
  void *vaddr;
};

struct CamBuffer g_cam_bufs[CAM_BUF_COUNT];
int g_v4l2_fd = -1;
struct DRMContext g_drm;

static uint64_t monotonic_time_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static double ns_to_ms(uint64_t ns) { return (double)ns / 1000000.0; }

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
#if 0
int create_drm_framebuffer(struct DRMContext *ctx, int idx) {
  struct drm_mode_create_dumb create_req;
  memset(&create_req, 0, sizeof(create_req));
  create_req.width = DST_WIDTH;
  create_req.height = DST_HEIGHT * 3 / 2;
  create_req.bpp = 8;
  // create_req.flags = ROCKCHIP_BO_DMA32; 目前不需要强制使用 32 位 DMA-BUF，除非遇到特定硬件限制

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
#endif

// 💡【核心重构】摒弃 DUMB 命令，使用瑞芯微原生私有扩展 GEM 分配器实现强制 DMA32 寻址锁区
int create_drm_framebuffer(struct DRMContext *ctx, int idx) {
  uint32_t align_pitch = (DST_WIDTH + 15) & ~15;
  size_t alloc_size = align_pitch * DST_HEIGHT * 3 / 2;

  struct drm_rockchip_gem_create gc_req = {
    .size = alloc_size,
    .flags = ROCKCHIP_BO_DMA32 // 完美注入私有标志，物理地址铁定锁在 4G 以内
  };

  if (drmIoctl(ctx->drm_fd, DRM_IOCTL_ROCKCHIP_GEM_CREATE, &gc_req) < 0) {
    perror("❌ 错误：调用私有 DRM_IOCTL_ROCKCHIP_GEM_CREATE 失败");
    return -1;
  }

  ctx->gem_handle[idx] = gc_req.handle;
  ctx->pitch[idx] = align_pitch;
  ctx->size = alloc_size;

  if (drmPrimeHandleToFD(ctx->drm_fd, gc_req.handle, DRM_CLOEXEC | DRM_RDWR, &ctx->dma_buf_fd[idx]) < 0) return -1;
  ctx->vaddr[idx] = mmap(NULL, alloc_size, PROT_READ | PROT_WRITE, MAP_SHARED, ctx->dma_buf_fd[idx], 0);
  if (ctx->vaddr[idx] == MAP_FAILED) {
    perror("映射 DRM framebuffer 失败");
    close(ctx->dma_buf_fd[idx]);
    return -1;
  }

  uint32_t pitches[4] = {align_pitch, align_pitch, 0, 0};
  uint32_t offsets[4] = {0, align_pitch * DST_HEIGHT, 0, 0};
  uint32_t handles[4] = {gc_req.handle, gc_req.handle, 0, 0};

  return drmModeAddFB2(ctx->drm_fd, DST_WIDTH, DST_HEIGHT, DRM_FORMAT_NV12, handles, pitches, offsets, &ctx->fb_id[idx], 0);
}



static void draw_nv12_pixel(uint8_t *frame, int pitch, int x, int y) {
  if (x < 0 || x >= DST_WIDTH || y < 0 || y >= DST_HEIGHT)
    return;
  frame[(size_t)y * pitch + x] = 149;
  uint8_t *uv = frame + (size_t)pitch * DST_HEIGHT;
  size_t uv_index = (size_t)(y / 2) * pitch + (x & ~1);
  uv[uv_index] = 43;
  uv[uv_index + 1] = 21;
}

static void draw_detection_boxes_cpu(uint8_t *frame, int pitch,
                                     const RknnDetection *detections,
                                     int count) {
  for (int i = 0; i < count; ++i) {
    int left = detections[i].left;
    int top = detections[i].top;
    int right = detections[i].right;
    int bottom = detections[i].bottom;
    if (left < 0)
      left = 0;
    if (top < 0)
      top = 0;
    if (right > DST_WIDTH)
      right = DST_WIDTH;
    if (bottom > DST_HEIGHT)
      bottom = DST_HEIGHT;
    left &= ~1;
    top &= ~1;
    right &= ~1;
    bottom &= ~1;
    if (right - left < 10 || bottom - top < 10)
      continue;

    for (int thickness = 0; thickness < 4; thickness += 2) {
      int y_top = top + thickness;
      int y_bottom = bottom - 2 - thickness;
      int x_left = left + thickness;
      int x_right = right - 2 - thickness;
      for (int x = left; x < right; x += 2) {
        draw_nv12_pixel(frame, pitch, x, y_top);
        draw_nv12_pixel(frame, pitch, x, y_bottom);
      }
      for (int y = top; y < bottom; y += 2) {
        draw_nv12_pixel(frame, pitch, x_left, y);
        draw_nv12_pixel(frame, pitch, x_right, y);
      }
    }
  }
}

static uint8_t score_glyph_row(char character, int row) {
  static const uint8_t letters[26][7] = {
      {14, 17, 17, 31, 17, 17, 17}, {30, 17, 17, 30, 17, 17, 30},
      {14, 17, 16, 16, 16, 17, 14}, {30, 17, 17, 17, 17, 17, 30},
      {31, 16, 16, 30, 16, 16, 31}, {31, 16, 16, 30, 16, 16, 16},
      {14, 17, 16, 23, 17, 17, 15}, {17, 17, 17, 31, 17, 17, 17},
      {14, 4, 4, 4, 4, 4, 14},       {7, 2, 2, 2, 18, 18, 12},
      {17, 18, 20, 24, 20, 18, 17}, {16, 16, 16, 16, 16, 16, 31},
      {17, 27, 21, 21, 17, 17, 17}, {17, 25, 21, 19, 17, 17, 17},
      {14, 17, 17, 17, 17, 17, 14}, {30, 17, 17, 30, 16, 16, 16},
      {14, 17, 17, 17, 21, 18, 13}, {30, 17, 17, 30, 20, 18, 17},
      {15, 16, 16, 14, 1, 1, 30},    {31, 4, 4, 4, 4, 4, 4},
      {17, 17, 17, 17, 17, 17, 14}, {17, 17, 17, 17, 17, 10, 4},
      {17, 17, 17, 21, 21, 21, 10}, {17, 17, 10, 4, 10, 17, 17},
      {17, 17, 10, 4, 4, 4, 4},     {31, 1, 2, 4, 8, 16, 31}};
  static const uint8_t digits[10][7] = {
      {14, 17, 19, 21, 25, 17, 14}, {4, 12, 4, 4, 4, 4, 14},
      {14, 17, 1, 2, 4, 8, 31},    {30, 1, 1, 14, 1, 1, 30},
      {2, 6, 10, 18, 31, 2, 2},    {31, 16, 16, 30, 1, 1, 30},
      {14, 16, 16, 30, 17, 17, 14}, {31, 1, 2, 4, 8, 8, 8},
      {14, 17, 17, 14, 17, 17, 14}, {14, 17, 17, 15, 1, 1, 14}};
  static const uint8_t percent[7] = {17, 2, 4, 8, 16, 17, 0};
  if (row < 0 || row >= 7)
    return 0;
  if (character >= 'a' && character <= 'z')
    character -= 'a' - 'A';
  if (character >= 'A' && character <= 'Z')
    return letters[character - 'A'][row];
  if (character >= '0' && character <= '9')
    return digits[character - '0'][row];
  if (character == '%')
    return percent[row];
  return 0;
}

static void draw_score_label(uint8_t *frame, int pitch, int left, int top,
                             const char *name, float score) {
  char text[32];
  int percent = (int)(score * 100.0f + 0.5f);
  if (percent < 0)
    percent = 0;
  if (percent > 100)
    percent = 100;
  snprintf(text, sizeof(text), "%s %d%%", name, percent);

  const int scale = 3;
  const int glyph_width = 5 * scale;
  const int glyph_height = 7 * scale;
  const int spacing = scale;
  int text_width = (int)strlen(text) * (glyph_width + spacing) - spacing;
  int label_x = left;
  if (label_x + text_width > DST_WIDTH)
    label_x = DST_WIDTH - text_width;
  int label_y = top - glyph_height - 2 * scale;
  if (label_y < 0)
    label_y = top;
  if (label_y + glyph_height > DST_HEIGHT)
    label_y = DST_HEIGHT - glyph_height;

  uint8_t *uv = frame + (size_t)pitch * DST_HEIGHT;
  for (int y = label_y - scale; y < label_y + glyph_height + scale; ++y) {
    if (y < 0 || y >= DST_HEIGHT)
      continue;
    memset(frame + (size_t)y * pitch + label_x, 16,
           (size_t)text_width);
    int uv_y = y / 2;
    for (int x = label_x & ~1; x < label_x + text_width; x += 2) {
      if (x + 1 < pitch) {
        uv[uv_y * pitch + x] = 128;
        uv[uv_y * pitch + x + 1] = 128;
      }
    }
  }

  for (int character_index = 0; text[character_index] != '\0';
       ++character_index) {
    int character_x = label_x + character_index * (glyph_width + spacing);
    for (int row = 0; row < 7; ++row) {
      uint8_t bits = score_glyph_row(text[character_index], row);
      for (int column = 0; column < 5; ++column) {
        if (!(bits & (1u << (4 - column))))
          continue;
        for (int dy = 0; dy < scale; ++dy) {
          int y = label_y + row * scale + dy;
          for (int dx = 0; dx < scale; ++dx) {
            int x = character_x + column * scale + dx;
            if (x < 0 || x >= DST_WIDTH || y < 0 || y >= DST_HEIGHT)
              continue;
            frame[(size_t)y * pitch + x] = 235;
          }
        }
      }
    }
  }
}

static void draw_detection_labels(uint8_t *frame, int pitch,
                                  const RknnDetection *detections, int count) {
  for (int i = 0; i < count; ++i) {
    int left = detections[i].left;
    int top = detections[i].top;
    if (detections[i].right <= left || detections[i].bottom <= top)
      continue;
    draw_score_label(frame, pitch, left, top, detections[i].name,
                     detections[i].score);
  }
}

#define FRAME_RING_SIZE 3

typedef enum {
  STAGE_FREE = 0,
  STAGE_CAPTURED,
  STAGE_RGA_BUSY,
  STAGE_ROTATED,
  STAGE_INFER_BUSY,
  STAGE_INFERRED,
  STAGE_DISPLAY_BUSY,
  STAGE_DISPLAYED
} FrameStage;

typedef struct {
  int valid;
  int frame_id;
  int cam_index;
  int cam_fd;
  int drm_index;
  int display_fd;
  int detection_count;
  uint64_t capture_time_ns;
  uint64_t capture_interval_ns;
  uint64_t camera_buffer_hold_ns;
  uint64_t rga_queue_ns;
  uint64_t rga_rotate_ns;
  uint64_t overlay_ns;
  uint64_t page_flip_wait_ns;
  uint64_t end_to_end_ns;
  RknnStageTiming infer_timing;
  FrameStage stage;
  RknnDetection detections[64];
} FrameSlot;

static FrameSlot g_frame_ring[FRAME_RING_SIZE];
static pthread_mutex_t g_ring_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_ring_cond = PTHREAD_COND_INITIALIZER;
static atomic_int g_stop;
static atomic_int g_capture_stop;
static atomic_int g_worker_error;
static atomic_int g_completed_frames;
static int g_front_drm_index;
static int g_drm_reserved[DRM_BUF_COUNT];

static void init_ring_slots(void) {
  pthread_mutex_lock(&g_ring_lock);
  for (int i = 0; i < FRAME_RING_SIZE; ++i) {
    memset(&g_frame_ring[i], 0, sizeof(g_frame_ring[i]));
    g_frame_ring[i].valid = 0;
    g_frame_ring[i].frame_id = i;
    g_frame_ring[i].stage = STAGE_FREE;
  }
  pthread_mutex_unlock(&g_ring_lock);
}

static FrameSlot *store_captured_frame(int cam_index, uint64_t capture_time_ns,
                                       uint64_t capture_interval_ns) {
  pthread_mutex_lock(&g_ring_lock);
  for (int i = 0; i < FRAME_RING_SIZE; ++i) {
    if (!g_frame_ring[i].valid) {
      memset(&g_frame_ring[i], 0, sizeof(g_frame_ring[i]));
      g_frame_ring[i].valid = 1;
      g_frame_ring[i].frame_id = i;
      g_frame_ring[i].cam_index = cam_index;
      g_frame_ring[i].cam_fd = g_cam_bufs[cam_index].fd;
      g_frame_ring[i].capture_time_ns = capture_time_ns;
      g_frame_ring[i].capture_interval_ns = capture_interval_ns;
      g_frame_ring[i].stage = STAGE_CAPTURED;
      pthread_cond_broadcast(&g_ring_cond);
      pthread_mutex_unlock(&g_ring_lock);
      return &g_frame_ring[i];
    }
  }
  pthread_mutex_unlock(&g_ring_lock);
  return NULL;
}

static FrameSlot *claim_slot(FrameStage ready_stage, FrameStage busy_stage) {
  pthread_mutex_lock(&g_ring_lock);
  for (;;) {
    for (int i = 0; i < FRAME_RING_SIZE; ++i) {
      if (g_frame_ring[i].valid && g_frame_ring[i].stage == ready_stage) {
        g_frame_ring[i].stage = busy_stage;
        pthread_mutex_unlock(&g_ring_lock);
        return &g_frame_ring[i];
      }
    }
    if (atomic_load(&g_stop))
      break;
    pthread_cond_wait(&g_ring_cond, &g_ring_lock);
  }
  pthread_mutex_unlock(&g_ring_lock);
  return NULL;
}

static void mark_slot_stage(FrameSlot *slot, FrameStage stage) {
  if (!slot)
    return;
  pthread_mutex_lock(&g_ring_lock);
  slot->stage = stage;
  pthread_cond_broadcast(&g_ring_cond);
  pthread_mutex_unlock(&g_ring_lock);
}

static int acquire_backbuffer(void) {
  pthread_mutex_lock(&g_ring_lock);
  for (;;) {
    for (int i = 0; i < DRM_BUF_COUNT; ++i) {
      if (i != g_front_drm_index && !g_drm_reserved[i]) {
        g_drm_reserved[i] = 1;
        pthread_mutex_unlock(&g_ring_lock);
        return i;
      }
    }
    if (atomic_load(&g_stop))
      break;
    pthread_cond_wait(&g_ring_cond, &g_ring_lock);
  }
  pthread_mutex_unlock(&g_ring_lock);
  return -1;
}

static void release_backbuffer(int index, int became_front) {
  if (index < 0 || index >= DRM_BUF_COUNT)
    return;
  pthread_mutex_lock(&g_ring_lock);
  if (became_front)
    g_front_drm_index = index;
  g_drm_reserved[index] = 0;
  pthread_cond_broadcast(&g_ring_cond);
  pthread_mutex_unlock(&g_ring_lock);
}

static int requeue_camera_buffer(int index) {
  if (index < 0 || index >= CAM_BUF_COUNT)
    return -1;

  struct v4l2_buffer buf;
  struct v4l2_plane plane;
  memset(&buf, 0, sizeof(buf));
  memset(&plane, 0, sizeof(plane));
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  buf.memory = V4L2_MEMORY_DMABUF;
  buf.index = index;
  buf.length = 1;
  buf.m.planes = &plane;
  plane.m.fd = g_cam_bufs[index].fd;
  plane.length = g_cam_bufs[index].size;
  return ioctl(g_v4l2_fd, VIDIOC_QBUF, &buf);
}

static int release_slot_camera_buffer(FrameSlot *slot) {
  if (!slot)
    return -1;
  slot->camera_buffer_hold_ns = monotonic_time_ns() - slot->capture_time_ns;
  return requeue_camera_buffer(slot->cam_index);
}

static int ring_is_empty(void) {
  int empty = 1;
  pthread_mutex_lock(&g_ring_lock);
  for (int i = 0; i < FRAME_RING_SIZE; ++i) {
    if (g_frame_ring[i].valid) {
      empty = 0;
      break;
    }
  }
  pthread_mutex_unlock(&g_ring_lock);
  return empty;
}

static void release_slot(FrameSlot *slot) {
  if (!slot)
    return;
  pthread_mutex_lock(&g_ring_lock);
  memset(slot, 0, sizeof(*slot));
  slot->valid = 0;
  slot->stage = STAGE_FREE;
  pthread_cond_broadcast(&g_ring_cond);
  pthread_mutex_unlock(&g_ring_lock);
}

static void *capture_worker(void *arg) {
  (void)arg;
  uint64_t previous_capture_time_ns = 0;
  while (!atomic_load(&g_capture_stop)) {
    struct pollfd pfd = {.fd = g_v4l2_fd, .events = POLLIN};
    int poll_ret = poll(&pfd, 1, 100);
    if (atomic_load(&g_capture_stop))
      break;
    if (poll_ret < 0) {
      if (errno == EINTR)
        continue;
      perror("capture_worker: poll 错误");
      atomic_store(&g_worker_error, 1);
      break;
    }
    if (poll_ret == 0)
      continue;

    struct v4l2_buffer buf;
    struct v4l2_plane planes;
    memset(&buf, 0, sizeof(buf));
    memset(&planes, 0, sizeof(planes));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    buf.memory = V4L2_MEMORY_DMABUF;
    buf.length = 1;
    buf.m.planes = &planes;

    if (ioctl(g_v4l2_fd, VIDIOC_DQBUF, &buf) < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK)
        continue;
      if (atomic_load(&g_capture_stop))
        break;
      perror("capture_worker: DQBUF 错误");
      atomic_store(&g_worker_error, 1);
      break;
    }

    if (buf.index >= CAM_BUF_COUNT) {
      fprintf(stderr, "capture_worker: invalid V4L2 buffer index %u\n", buf.index);
      atomic_store(&g_worker_error, 1);
      break;
    }

    uint64_t capture_time_ns = monotonic_time_ns();
    uint64_t capture_interval_ns = previous_capture_time_ns == 0
                       ? 0
                       : capture_time_ns - previous_capture_time_ns;
    previous_capture_time_ns = capture_time_ns;
    FrameSlot *slot = store_captured_frame((int)buf.index, capture_time_ns,
                         capture_interval_ns);
    if (!slot) {
      fprintf(stderr, "capture_worker: no free slot\n");
      if (requeue_camera_buffer((int)buf.index) < 0)
        perror("capture_worker: QBUF 失败");
      continue;
    }
  }
  return NULL;
}

static void *rga_worker(void *arg) {
  (void)arg;
  while (!atomic_load(&g_stop)) {
    FrameSlot *slot = claim_slot(STAGE_CAPTURED, STAGE_RGA_BUSY);
    if (!slot)
      break;

    int drm_idx = acquire_backbuffer();
    if (drm_idx < 0) {
      if (requeue_camera_buffer(slot->cam_index) < 0)
        perror("rga_worker: QBUF 失败");
      release_slot(slot);
      continue;
    }
    slot->drm_index = drm_idx;
    slot->rga_queue_ns = monotonic_time_ns() - slot->capture_time_ns;

    im_handle_param_t src_param = {SRC_WIDTH, SRC_HEIGHT, RK_FORMAT_YCbCr_420_SP};
    im_handle_param_t dst_param = {DST_WIDTH, DST_HEIGHT, RK_FORMAT_YCbCr_420_SP};

    rga_buffer_handle_t src_handle = importbuffer_fd(slot->cam_fd, &src_param);
    if (src_handle == 0) {
      fprintf(stderr, "RGA failed to import camera buffer\n");
      release_backbuffer(drm_idx, 0);
      requeue_camera_buffer(slot->cam_index);
      release_slot(slot);
      continue;
    }
    rga_buffer_t src = wrapbuffer_handle(src_handle, SRC_WIDTH, SRC_HEIGHT,
                                         RK_FORMAT_YCbCr_420_SP);
    src.wstride = (SRC_WIDTH + 15) & ~15;
    src.hstride = (SRC_HEIGHT + 1) & ~1;

    rga_buffer_handle_t dst_handle = importbuffer_fd(g_drm.dma_buf_fd[drm_idx], &dst_param);
    if (dst_handle == 0) {
      fprintf(stderr, "RGA failed to import DRM buffer\n");
      releasebuffer_handle(src_handle);
      release_backbuffer(drm_idx, 0);
      requeue_camera_buffer(slot->cam_index);
      release_slot(slot);
      continue;
    }
    rga_buffer_t dst = wrapbuffer_handle(dst_handle, DST_WIDTH, DST_HEIGHT,
                                         RK_FORMAT_YCbCr_420_SP);
    dst.wstride = g_drm.pitch[drm_idx];
    dst.hstride = DST_HEIGHT;

    uint64_t rotate_start_ns = monotonic_time_ns();
    int ret = imrotate(src, dst, IM_HAL_TRANSFORM_ROT_90);
    slot->rga_rotate_ns = monotonic_time_ns() - rotate_start_ns;
    releasebuffer_handle(src_handle);
    releasebuffer_handle(dst_handle);

    if (release_slot_camera_buffer(slot) < 0) {
      perror("rga_worker: RGA 完成后 QBUF 失败");
      atomic_store(&g_worker_error, 1);
    }

    if (ret != IM_STATUS_SUCCESS) {
      fprintf(stderr, "RGA rotate failed: %s\n", imStrError((IM_STATUS)ret));
      release_backbuffer(drm_idx, 0);
      release_slot(slot);
      continue;
    }

    slot->display_fd = g_drm.dma_buf_fd[drm_idx];
    mark_slot_stage(slot, STAGE_ROTATED);
  }
  return NULL;
}

static void *infer_worker(void *arg) {
  RknnDetector *detector = (RknnDetector *)arg;
  while (!atomic_load(&g_stop)) {
    FrameSlot *slot = claim_slot(STAGE_ROTATED, STAGE_INFER_BUSY);
    if (!slot)
      break;

    slot->detection_count = rknn_detector_run(
        detector, slot->display_fd, DST_WIDTH, DST_HEIGHT,
        g_drm.pitch[slot->drm_index], slot->detections, 64,
        &slot->infer_timing);
    if (slot->detection_count < 0) {
      fprintf(stderr, "RKNN inference failed on rotated frame\n");
      slot->detection_count = 0;
    }
    mark_slot_stage(slot, STAGE_INFERRED);
  }
  return NULL;
}

static void *display_worker(void *arg) {
  (void)arg;
  uint64_t window_start_ns = monotonic_time_ns();
  uint64_t sum_capture_interval_ns = 0;
  uint64_t sum_camera_buffer_hold_ns = 0;
  uint64_t sum_rga_queue_ns = 0;
  uint64_t sum_rga_rotate_ns = 0;
  uint64_t sum_rga_fill_ns = 0;
  uint64_t sum_rga_resize_ns = 0;
  uint64_t sum_npu_run_ns = 0;
  uint64_t sum_output_sync_ns = 0;
  uint64_t sum_postprocess_ns = 0;
  uint64_t sum_overlay_ns = 0;
  uint64_t sum_page_flip_wait_ns = 0;
  uint64_t sum_end_to_end_ns = 0;
  int window_frames = 0;

  while (!atomic_load(&g_stop)) {
    FrameSlot *slot = claim_slot(STAGE_INFERRED, STAGE_DISPLAY_BUSY);
    if (!slot)
      break;

    uint64_t overlay_start_ns = monotonic_time_ns();
    int box_status = rga_draw_detection_boxes(
        g_drm.dma_buf_fd[slot->drm_index], DST_WIDTH, DST_HEIGHT,
        g_drm.pitch[slot->drm_index], slot->detections, slot->detection_count);
    if (box_status != IM_STATUS_SUCCESS) {
      fprintf(stderr, "RGA box draw failed: %s\n",
              imStrError((IM_STATUS)box_status));
    }

    if (sync_cpu_cache(g_drm.dma_buf_fd[slot->drm_index],
                       DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE) == 0) {
      draw_detection_labels((uint8_t *)g_drm.vaddr[slot->drm_index],
                            g_drm.pitch[slot->drm_index],
                            slot->detections, slot->detection_count);
      sync_cpu_cache(g_drm.dma_buf_fd[slot->drm_index],
                     DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
    }
    slot->overlay_ns = monotonic_time_ns() - overlay_start_ns;

    int flip_done = 0;
    if (drmModePageFlip(g_drm.drm_fd, g_drm.crtc_id,
                        g_drm.fb_id[slot->drm_index],
                        DRM_MODE_PAGE_FLIP_EVENT, &flip_done) < 0) {
      perror("display_worker: page flip 失败");
      release_backbuffer(slot->drm_index, 0);
      release_slot(slot);
      atomic_store(&g_worker_error, 1);
      continue;
    }
    uint64_t flip_wait_start_ns = monotonic_time_ns();
    if (wait_for_page_flip(g_drm.drm_fd, &flip_done) < 0) {
      perror("display_worker: 等待 page flip 完成失败");
      release_slot(slot);
      atomic_store(&g_worker_error, 1);
      continue;
    }
    slot->page_flip_wait_ns = monotonic_time_ns() - flip_wait_start_ns;
    slot->end_to_end_ns = monotonic_time_ns() - slot->capture_time_ns;
    release_backbuffer(slot->drm_index, 1);

    mark_slot_stage(slot, STAGE_DISPLAYED);
    int completed = atomic_fetch_add(&g_completed_frames, 1) + 1;
    sum_capture_interval_ns += slot->capture_interval_ns;
    sum_camera_buffer_hold_ns += slot->camera_buffer_hold_ns;
    sum_rga_queue_ns += slot->rga_queue_ns;
    sum_rga_rotate_ns += slot->rga_rotate_ns;
    sum_rga_fill_ns += slot->infer_timing.rga_fill_ns;
    sum_rga_resize_ns += slot->infer_timing.rga_resize_ns;
    sum_npu_run_ns += slot->infer_timing.npu_run_ns;
    sum_output_sync_ns += slot->infer_timing.output_sync_ns;
    sum_postprocess_ns += slot->infer_timing.postprocess_ns;
    sum_overlay_ns += slot->overlay_ns;
    sum_page_flip_wait_ns += slot->page_flip_wait_ns;
    sum_end_to_end_ns += slot->end_to_end_ns;
    window_frames++;

    if (window_frames >= TIMING_REPORT_INTERVAL) {
      uint64_t window_elapsed_ns = monotonic_time_ns() - window_start_ns;
      double divisor = (double)window_frames;
      double fps = window_elapsed_ns == 0
                       ? 0.0
                       : divisor * 1000000000.0 / (double)window_elapsed_ns;
      printf("TIMING frames=%d..%d fps=%.2f avg_ms "
             "capture_interval=%.3f camera_buffer_hold=%.3f "
             "rga_queue=%.3f rotate=%.3f "
             "rga_fill=%.3f rga_resize=%.3f npu_run=%.3f "
             "output_sync=%.3f postprocess=%.3f overlay=%.3f "
             "page_flip_wait=%.3f end_to_end=%.3f\n",
             completed - window_frames + 1, completed, fps,
             ns_to_ms(sum_capture_interval_ns) / divisor,
             ns_to_ms(sum_camera_buffer_hold_ns) / divisor,
             ns_to_ms(sum_rga_queue_ns) / divisor,
             ns_to_ms(sum_rga_rotate_ns) / divisor,
             ns_to_ms(sum_rga_fill_ns) / divisor,
             ns_to_ms(sum_rga_resize_ns) / divisor,
             ns_to_ms(sum_npu_run_ns) / divisor,
             ns_to_ms(sum_output_sync_ns) / divisor,
             ns_to_ms(sum_postprocess_ns) / divisor,
             ns_to_ms(sum_overlay_ns) / divisor,
             ns_to_ms(sum_page_flip_wait_ns) / divisor,
             ns_to_ms(sum_end_to_end_ns) / divisor);
      fflush(stdout);
      sum_capture_interval_ns = 0;
      sum_camera_buffer_hold_ns = 0;
      sum_rga_queue_ns = 0;
      sum_rga_rotate_ns = 0;
      sum_rga_fill_ns = 0;
      sum_rga_resize_ns = 0;
      sum_npu_run_ns = 0;
      sum_output_sync_ns = 0;
      sum_postprocess_ns = 0;
      sum_overlay_ns = 0;
      sum_page_flip_wait_ns = 0;
      sum_end_to_end_ns = 0;
      window_frames = 0;
      window_start_ns = monotonic_time_ns();
    }
    release_slot(slot);
  }
  return NULL;
}

int main(int argc, char *argv[]) {
  const char *model_path = argc > 1 ? argv[1] : DEFAULT_RKNN_MODEL;
  RknnDetector detector = {0};
  if (rknn_detector_init(&detector, model_path) < 0) {
    fprintf(stderr, "RKNN model initialization failed: %s\n", model_path);
    return -1;
  }
  printf("RKNN detector ready: %s, input=%dx%d\n", model_path,
         detector.model_width, detector.model_height);

  if (init_drm_display(&g_drm) < 0)
    return -1;
  for (int i = 0; i < DRM_BUF_COUNT; ++i) {
    if (create_drm_framebuffer(&g_drm, i) < 0)
      return -1;
  }

  if (drmModeSetCrtc(g_drm.drm_fd, g_drm.crtc_id, g_drm.fb_id[0], 0, 0,
                     &g_drm.conn_id, 1, &g_drm.mode) < 0) {
    perror("DRM 初始 modeset 失败");
    return -1;
  }

  size_t src_size = SRC_WIDTH * SRC_HEIGHT * 3 / 2;
  if (allocate_from_central_heap(src_size, &g_cam_bufs[0]) < 0 ||
      allocate_from_central_heap(src_size, &g_cam_bufs[1]) < 0) {
    printf("全局中央统一 DMA-BUF 缓冲区池分配失败!\n");
    return -1;
  }

  if (init_v4l2() < 0)
    return -1;

  init_ring_slots();
  g_front_drm_index = 0;
  printf("⚡【线程化·Camera ➡ RGA ➡ NPU -> DRM 双三缓冲流水线】已启动...\n");

  pthread_t capture_tid, rga_tid, infer_tid, display_tid;
  pthread_create(&capture_tid, NULL, capture_worker, NULL);
  pthread_create(&rga_tid, NULL, rga_worker, NULL);
  pthread_create(&infer_tid, NULL, infer_worker, &detector);
  pthread_create(&display_tid, NULL, display_worker, NULL);

  while (atomic_load(&g_completed_frames) < TOTAL_FRAMES &&
         !atomic_load(&g_worker_error))
    usleep(10000);

  atomic_store(&g_capture_stop, 1);
  pthread_join(capture_tid, NULL);
  if (!atomic_load(&g_worker_error)) {
    while (!ring_is_empty())
      usleep(1000);
  } else {
    atomic_store(&g_stop, 1);
    pthread_mutex_lock(&g_ring_lock);
    pthread_cond_broadcast(&g_ring_cond);
    pthread_mutex_unlock(&g_ring_lock);
  }

  enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  atomic_store(&g_stop, 1);
  pthread_mutex_lock(&g_ring_lock);
  pthread_cond_broadcast(&g_ring_cond);
  pthread_mutex_unlock(&g_ring_lock);

  pthread_join(rga_tid, NULL);
  pthread_join(infer_tid, NULL);
  pthread_join(display_tid, NULL);
  ioctl(g_v4l2_fd, VIDIOC_STREAMOFF, &type);

  printf("流转结束，开始优雅销毁并归还系统显示主权...\n");
  for (int i = 0; i < DRM_BUF_COUNT; i++) {
    drmModeRmFB(g_drm.drm_fd, g_drm.fb_id[i]);
    if (g_drm.vaddr[i] && g_drm.vaddr[i] != MAP_FAILED)
      munmap(g_drm.vaddr[i], g_drm.size);
    close(g_drm.dma_buf_fd[i]);
    struct drm_gem_close close_req = {.handle = (uint32_t)g_drm.gem_handle[i]};
    drmIoctl(g_drm.drm_fd, DRM_IOCTL_GEM_CLOSE, &close_req);
  }
  for (int i = 0; i < CAM_BUF_COUNT; i++) {
    if (g_cam_bufs[i].vaddr && g_cam_bufs[i].vaddr != MAP_FAILED)
      munmap(g_cam_bufs[i].vaddr, g_cam_bufs[i].size);
    close(g_cam_bufs[i].fd);
  }
  close(g_v4l2_fd);
  drmClose(g_drm.drm_fd);
  pthread_cond_destroy(&g_ring_cond);
  pthread_mutex_destroy(&g_ring_lock);
  rknn_detector_deinit(&detector);
  printf("全链路资源清理完毕，程序安全退出。\n");
  return atomic_load(&g_worker_error) ? -1 : 0;
}