#include "rknn_infer.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "postprocess.h"
#include "rknn_api.h"

struct DetectorState {
  rknn_context context;
  rknn_input_output_num io_count;
  rknn_tensor_attr input_attr;
  rknn_tensor_attr output_attrs[3];
  std::vector<uint8_t> input_rgb;
};

static uint8_t clamp_byte(int value) {
  return static_cast<uint8_t>(std::max(0, std::min(255, value)));
}

static void nv12_to_rgb(const uint8_t *nv12, int height, int pitch,
                        int x, int y, uint8_t *rgb) {
  int y_value = nv12[y * pitch + x];
  const uint8_t *uv = nv12 + static_cast<size_t>(pitch) * height;
  int uv_index = (y / 2) * pitch + (x & ~1);
  int u_value = uv[uv_index] - 128;
  int v_value = uv[uv_index + 1] - 128;
  int c = std::max(0, y_value - 16);
  rgb[0] = clamp_byte((298 * c + 409 * v_value + 128) >> 8);
  rgb[1] = clamp_byte((298 * c - 100 * u_value - 208 * v_value + 128) >> 8);
  rgb[2] = clamp_byte((298 * c + 516 * u_value + 128) >> 8);
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
  state->input_rgb.resize(static_cast<size_t>(detector->model_width) *
                           detector->model_height * 3);
  detector->context = state;
  detector->model_data = model_data;
  detector->output_count = state->io_count.n_output;
  detector->output_attrs = state->output_attrs;
  return 0;
}

int rknn_detector_run(RknnDetector *detector, const uint8_t *nv12, int width,
                      int height, int pitch, RknnDetection *detections,
                      int max_detections) {
  if (!detector || !detector->context || !nv12 || !detections ||
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
  std::fill(state->input_rgb.begin(), state->input_rgb.end(), 114);

  for (int y = 0; y < resized_height; ++y) {
    int source_y = std::min(height - 1, y * height / resized_height);
    for (int x = 0; x < resized_width; ++x) {
      int source_x = std::min(width - 1, x * width / resized_width);
      size_t dst_index =
          (static_cast<size_t>(y + pad_top) * model_width + x + pad_left) * 3;
      nv12_to_rgb(nv12, height, pitch, source_x, source_y,
                  &state->input_rgb[dst_index]);
    }
  }

  rknn_input input{};
  input.index = 0;
  input.type = RKNN_TENSOR_UINT8;
  input.size = state->input_rgb.size();
  input.fmt = RKNN_TENSOR_NHWC;
  input.buf = state->input_rgb.data();
  if (rknn_inputs_set(state->context, 1, &input) < 0 ||
      rknn_run(state->context, nullptr) < 0)
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
    rknn_destroy(state->context);
    delete state;
  }
  std::free(detector->model_data);
  std::memset(detector, 0, sizeof(*detector));
  deinitPostProcess();
}