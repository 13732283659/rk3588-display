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
  int input_fd;
  void *input_mapping;
  size_t input_mapping_size;
};

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

    state->input_attr.type = RKNN_TENSOR_UINT8;
    if (state->input_attr.fmt != RKNN_TENSOR_NHWC ||
      state->input_attr.w_stride <
        static_cast<uint32_t>(detector->model_width) ||
      state->input_attr.w_stride >
        static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    std::fprintf(stderr,
                 "RKNN zero-copy input requires NHWC UINT8 and valid width stride\n");
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
    rknn_destroy(state->context);
    delete state;
    std::free(model_data);
    return -1;
  }

  int heap_fd = open("/dev/dma_heap/cma", O_RDWR | O_CLOEXEC);
  if (heap_fd < 0) {
    std::perror("open CMA DMA heap for RKNN input");
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
                      int max_detections) {
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
  IM_STATUS rga_status = imfill(dst, full_rect, 0xff727272);
  if (rga_status != IM_STATUS_SUCCESS) {
    std::fprintf(stderr, "RGA letterbox fill failed: %s\n",
                 imStrError(rga_status));
    return -1;
  }
  im_rect src_rect = {0, 0, width, height};
  im_rect dst_rect = {pad_left, pad_top, resized_width, resized_height};
  rga_status = improcess(src, dst, {}, src_rect, dst_rect, {}, IM_SYNC);
  if (rga_status != IM_STATUS_SUCCESS) {
    std::fprintf(stderr, "RGA NV12-to-RGB letterbox resize failed: %s\n",
                 imStrError(rga_status));
    return -1;
  }
  if (rknn_run(state->context, nullptr) < 0)
    return -1;

  rknn_output outputs[3]{};
  for (int i = 0; i < 3; ++i) {
    outputs[i].index = i;
    outputs[i].want_float = 0;
  }
  if (rknn_outputs_get(state->context, 3, outputs, nullptr) < 0)
    return -1;

  std::vector<int32_t> zero_points;
  std::vector<float> scales;
  for (int i = 0; i < 3; ++i) {
    zero_points.push_back(state->output_attrs[i].zp);
    scales.push_back(state->output_attrs[i].scale);
  }
  BOX_RECT pads{pad_left, pad_left + (model_width - resized_width) % 2,
                pad_top, pad_top + (model_height - resized_height) % 2};
  detect_result_group_t results{};
  int ret = post_process(static_cast<int8_t *>(outputs[0].buf),
                         static_cast<int8_t *>(outputs[1].buf),
                         static_cast<int8_t *>(outputs[2].buf), model_height,
                         model_width, 0.25f, 0.45f, pads, scale, scale,
                         zero_points, scales, &results);
  rknn_outputs_release(state->context, 3, outputs);
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