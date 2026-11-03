#include "rknn_infer.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "RgaUtils.h"
#include "im2d.h"
#include "postprocess.h"
#include "rknn_api.h"

struct DetectorState {
  rknn_context context;
  rknn_input_output_num io_count;
  rknn_tensor_attr input_attr;
  rknn_tensor_attr output_attrs[3];
  rknn_tensor_mem *input_mem;
  rknn_tensor_mem *output_mems[3];
  int input_fd;
  void *input_mapping;
  size_t input_mapping_size;
};

int rga_draw_detection_boxes(uint32_t drm_handle, int width, int height, int pitch,
                             const RknnDetection *detections, int count) {
  if (drm_handle == 0 || width <= 0 || height <= 0 || pitch < width ||
      (count > 0 && !detections))
    return -1;
  if (count <= 0)
    return IM_STATUS_SUCCESS;

  std::vector<im_rect> rectangles;
  rectangles.reserve(count);
  for (int i = 0; i < count; ++i) {
    int left = std::max(0, std::min(width, detections[i].left)) & ~1;
    int top = std::max(0, std::min(height, detections[i].top)) & ~1;
    int right = std::max(0, std::min(width, detections[i].right)) & ~1;
    int bottom = std::max(0, std::min(height, detections[i].bottom)) & ~1;
    if (right - left < 10 || bottom - top < 10)
      continue;
    rectangles.push_back({left, top, right - left, bottom - top});
  }
  if (rectangles.empty())
    return IM_STATUS_SUCCESS;

  rga_buffer_t target = wrapbuffer_handle(
      drm_handle, width, height, RK_FORMAT_YCbCr_420_SP, pitch, height);
  return imrectangleArray(target, rectangles.data(),
                          static_cast<int>(rectangles.size()), 0xff00ff00, 4,
                          1);
}

static double monotonic_ms() {
  struct timespec timestamp;
  clock_gettime(CLOCK_MONOTONIC, &timestamp);
  return timestamp.tv_sec * 1000.0 + timestamp.tv_nsec / 1000000.0;
}

int rknn_detector_init(RknnDetector *detector, const char *model_path) {
  if (!detector || !model_path)
    return -1;
  std::memset(detector, 0, sizeof(*detector));

  FILE *model_file = std::fopen(model_path, "rb");
  if (!model_file) {
    std::perror("open RKNN model");
    return -1;
  }
  if (std::fseek(model_file, 0, SEEK_END) != 0) {
    std::fclose(model_file);
    return -1;
  }
  long model_size = std::ftell(model_file);
  if (model_size <= 0 || std::fseek(model_file, 0, SEEK_SET) != 0) {
    std::fclose(model_file);
    return -1;
  }
  void *model_data = std::malloc(static_cast<size_t>(model_size));
  if (!model_data ||
      std::fread(model_data, 1, static_cast<size_t>(model_size), model_file) !=
          static_cast<size_t>(model_size)) {
    std::free(model_data);
    std::fclose(model_file);
    return -1;
  }
  std::fclose(model_file);

  DetectorState *state = new (std::nothrow) DetectorState{};
  if (!state) {
    std::free(model_data);
    return -1;
  }
  state->input_fd = -1;
  state->input_mapping = MAP_FAILED;
  state->context = 0;
  int ret = rknn_init(&state->context, model_data, model_size, 0, nullptr);
  if (ret < 0) {
    std::fprintf(stderr, "rknn_init failed: %d\n", ret);
    delete state;
    std::free(model_data);
    return -1;
  }
  if (rknn_query(state->context, RKNN_QUERY_IN_OUT_NUM, &state->io_count,
                 sizeof(state->io_count)) < 0 ||
      state->io_count.n_input != 1 || state->io_count.n_output != 3) {
    std::fprintf(stderr, "Expected one input and three YOLOv5 outputs\n");
    rknn_destroy(state->context);
    delete state;
    std::free(model_data);
    return -1;
  }

  state->input_attr.index = 0;
  if (rknn_query(state->context, RKNN_QUERY_INPUT_ATTR, &state->input_attr,
                 sizeof(state->input_attr)) < 0) {
    rknn_destroy(state->context);
    delete state;
    std::free(model_data);
    return -1;
  }
  if (state->input_attr.fmt == RKNN_TENSOR_NHWC) {
    detector->model_height = state->input_attr.dims[1];
    detector->model_width = state->input_attr.dims[2];
  } else if (state->input_attr.fmt == RKNN_TENSOR_NCHW) {
    detector->model_height = state->input_attr.dims[2];
    detector->model_width = state->input_attr.dims[3];
  } else {
    std::fprintf(stderr, "Unsupported RKNN input tensor format\n");
    rknn_destroy(state->context);
    delete state;
    std::free(model_data);
    return -1;
  }
  if (detector->model_width <= 0 || detector->model_height <= 0) {
    rknn_destroy(state->context);
    delete state;
    std::free(model_data);
    return -1;
  }

  for (uint32_t i = 0; i < state->io_count.n_output; ++i) {
    state->output_attrs[i].index = i;
    if (rknn_query(state->context, RKNN_QUERY_OUTPUT_ATTR,
                   &state->output_attrs[i], sizeof(rknn_tensor_attr)) < 0) {
      rknn_destroy(state->context);
      delete state;
      std::free(model_data);
      return -1;
    }
  }

  for (uint32_t i = 0; i < state->io_count.n_output; ++i) {
    state->output_mems[i] =
        rknn_create_mem(state->context, state->output_attrs[i].size);
    if (!state->output_mems[i] ||
        rknn_set_io_mem(state->context, state->output_mems[i],
                        &state->output_attrs[i]) < 0) {
      std::fprintf(stderr, "bind RKNN output tensor %u failed\n", i);
      for (uint32_t j = 0; j <= i; ++j) {
        if (state->output_mems[j])
          rknn_destroy_mem(state->context, state->output_mems[j]);
      }
      rknn_destroy(state->context);
      delete state;
      std::free(model_data);
      return -1;
    }
  }

    state->input_attr.type = RKNN_TENSOR_UINT8;
    if (state->input_attr.fmt != RKNN_TENSOR_NHWC ||
      state->input_attr.w_stride <
        static_cast<uint32_t>(detector->model_width) ||
      state->input_attr.w_stride >
        static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    std::fprintf(stderr,
                 "RKNN zero-copy input requires NHWC UINT8 and valid width stride\n");
    for (uint32_t i = 0; i < state->io_count.n_output; ++i)
      rknn_destroy_mem(state->context, state->output_mems[i]);
    rknn_destroy(state->context);
    delete state;
    std::free(model_data);
    return -1;
  }
  uint64_t required_size =
      static_cast<uint64_t>(state->input_attr.w_stride) *
      detector->model_height * 3;
  if (required_size > std::numeric_limits<uint32_t>::max() ||
      state->input_attr.size_with_stride < required_size) {
    std::fprintf(stderr, "RKNN input tensor stride/size is incompatible with RGA\n");
    for (uint32_t i = 0; i < state->io_count.n_output; ++i)
      rknn_destroy_mem(state->context, state->output_mems[i]);
    rknn_destroy(state->context);
    delete state;
    std::free(model_data);
    return -1;
  }

  int heap_fd = open("/dev/dma_heap/cma", O_RDWR | O_CLOEXEC);
  if (heap_fd < 0) {
    std::perror("open CMA DMA heap for RKNN input");
    for (uint32_t i = 0; i < state->io_count.n_output; ++i)
      rknn_destroy_mem(state->context, state->output_mems[i]);
    rknn_destroy(state->context);
    delete state;
    std::free(model_data);
    return -1;
  }
  struct dma_heap_allocation_data allocation = {};
  allocation.len = state->input_attr.size_with_stride;
  allocation.fd_flags = O_RDWR | O_CLOEXEC;
  if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &allocation) < 0) {
    std::perror("allocate RKNN input DMA-BUF");
    close(heap_fd);
    for (uint32_t i = 0; i < state->io_count.n_output; ++i)
      rknn_destroy_mem(state->context, state->output_mems[i]);
    rknn_destroy(state->context);
    delete state;
    std::free(model_data);
    return -1;
  }
  close(heap_fd);
  state->input_fd = static_cast<int>(allocation.fd);
  state->input_mapping_size = state->input_attr.size_with_stride;
  state->input_mapping = mmap(nullptr, state->input_mapping_size,
                              PROT_READ | PROT_WRITE, MAP_SHARED,
                              state->input_fd, 0);
  if (state->input_mapping == MAP_FAILED) {
    std::perror("mmap RKNN input DMA-BUF");
    close(state->input_fd);
    for (uint32_t i = 0; i < state->io_count.n_output; ++i)
      rknn_destroy_mem(state->context, state->output_mems[i]);
    rknn_destroy(state->context);
    delete state;
    std::free(model_data);
    return -1;
  }

  state->input_mem = rknn_create_mem_from_fd(
      state->context, state->input_fd, state->input_mapping,
      state->input_attr.size_with_stride, 0);
  if (!state->input_mem ||
      rknn_set_io_mem(state->context, state->input_mem,
                      &state->input_attr) < 0) {
    std::fprintf(stderr, "bind RKNN input DMA-BUF failed\n");
    if (state->input_mem)
      rknn_destroy_mem(state->context, state->input_mem);
    munmap(state->input_mapping, state->input_mapping_size);
    close(state->input_fd);
    for (uint32_t i = 0; i < state->io_count.n_output; ++i)
      rknn_destroy_mem(state->context, state->output_mems[i]);
    rknn_destroy(state->context);
    delete state;
    std::free(model_data);
    return -1;
  }
  detector->context = state;
  detector->model_data = model_data;
  detector->output_count = state->io_count.n_output;
  detector->output_attrs = state->output_attrs;
  return 0;
}

int rknn_detector_run(RknnDetector *detector, int nv12_fd, int width,
                      int height, int pitch, RknnDetection *detections,
                      int max_detections, RknnStageTimes *stage_times) {
  if (stage_times)
    std::memset(stage_times, 0, sizeof(*stage_times));
  if (!detector || !detector->context || nv12_fd < 0 || !detections ||
      max_detections <= 0 || width <= 0 || height <= 0 || pitch < width)
    return -1;
  DetectorState *state = static_cast<DetectorState *>(detector->context);
  int model_width = detector->model_width;
  int model_height = detector->model_height;
  float scale = std::min(static_cast<float>(model_width) / width,
                         static_cast<float>(model_height) / height);
  int resized_width = static_cast<int>(width * scale + 0.5f);
  int resized_height = static_cast<int>(height * scale + 0.5f);
  int pad_left = (model_width - resized_width) / 2;
  int pad_top = (model_height - resized_height) / 2;
  rga_buffer_t src = wrapbuffer_fd(nv12_fd, width, height,
                                   RK_FORMAT_YCbCr_420_SP, pitch, height);
  rga_buffer_t dst = wrapbuffer_fd(
      state->input_fd, model_width, model_height, RK_FORMAT_RGB_888,
      static_cast<int>(state->input_attr.w_stride), model_height);
  im_rect full_rect = {0, 0, model_width, model_height};
  double stage_start = monotonic_ms();
  IM_STATUS rga_status = imfill(dst, full_rect, 0xff727272);
  if (stage_times)
    stage_times->rga_fill_ms = monotonic_ms() - stage_start;
  if (rga_status != IM_STATUS_SUCCESS) {
    std::fprintf(stderr, "RGA letterbox fill failed: %s\n",
                 imStrError(rga_status));
    return -1;
  }
  im_rect src_rect = {0, 0, width, height};
  im_rect dst_rect = {pad_left, pad_top, resized_width, resized_height};
  stage_start = monotonic_ms();
  rga_status = improcess(src, dst, {}, src_rect, dst_rect, {}, IM_SYNC);
  if (stage_times)
    stage_times->rga_preprocess_ms = monotonic_ms() - stage_start;
  if (rga_status != IM_STATUS_SUCCESS) {
    std::fprintf(stderr, "RGA NV12-to-RGB letterbox resize failed: %s\n",
                 imStrError(rga_status));
    return -1;
  }
  stage_start = monotonic_ms();
  int run_status = rknn_run(state->context, nullptr);
  if (stage_times)
    stage_times->npu_run_ms = monotonic_ms() - stage_start;
  if (run_status < 0)
    return -1;

  stage_start = monotonic_ms();
  for (uint32_t i = 0; i < state->io_count.n_output; ++i) {
    if (rknn_mem_sync(state->context, state->output_mems[i],
                      RKNN_MEMORY_SYNC_FROM_DEVICE) < 0) {
      std::fprintf(stderr, "sync RKNN output tensor %u failed\n", i);
      return -1;
    }
  }
  if (stage_times)
    stage_times->output_sync_ms = monotonic_ms() - stage_start;

  std::vector<int32_t> zero_points;
  std::vector<float> scales;
  for (int i = 0; i < 3; ++i) {
    zero_points.push_back(state->output_attrs[i].zp);
    scales.push_back(state->output_attrs[i].scale);
  }
  BOX_RECT pads{pad_left, pad_left + (model_width - resized_width) % 2,
                pad_top, pad_top + (model_height - resized_height) % 2};
  detect_result_group_t results{};
  stage_start = monotonic_ms();
  int ret = post_process(
                         static_cast<int8_t *>(state->output_mems[0]->virt_addr),
                         static_cast<int8_t *>(state->output_mems[1]->virt_addr),
                         static_cast<int8_t *>(state->output_mems[2]->virt_addr),
                         model_height,
                         model_width, 0.25f, 0.45f, pads, scale, scale,
                         zero_points, scales, &results);
  if (stage_times)
    stage_times->postprocess_ms = monotonic_ms() - stage_start;
  if (ret < 0)
    return -1;

  int count = std::min(results.count, max_detections);
  for (int i = 0; i < count; ++i) {
    detections[i].left = std::max(0, std::min(width, results.results[i].box.left));
    detections[i].top = std::max(0, std::min(height, results.results[i].box.top));
    detections[i].right = std::max(0, std::min(width, results.results[i].box.right));
    detections[i].bottom = std::max(0, std::min(height, results.results[i].box.bottom));
    detections[i].score = results.results[i].prop;
    std::strncpy(detections[i].name, results.results[i].name,
                 sizeof(detections[i].name) - 1);
    detections[i].name[sizeof(detections[i].name) - 1] = '\0';
  }
  return count;
}

void rknn_detector_deinit(RknnDetector *detector) {
  if (!detector)
    return;
  DetectorState *state = static_cast<DetectorState *>(detector->context);
  if (state) {
    for (uint32_t i = 0; i < state->io_count.n_output; ++i) {
      if (state->output_mems[i])
        rknn_destroy_mem(state->context, state->output_mems[i]);
    }
    if (state->input_mem)
      rknn_destroy_mem(state->context, state->input_mem);
    if (state->input_mapping != MAP_FAILED)
      munmap(state->input_mapping, state->input_mapping_size);
    if (state->input_fd >= 0)
      close(state->input_fd);
    rknn_destroy(state->context);
    delete state;
  }
  std::free(detector->model_data);
  std::memset(detector, 0, sizeof(*detector));
  deinitPostProcess();
}