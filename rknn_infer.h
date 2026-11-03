#ifndef RK3588_DISPLAY_RKNN_INFER_H
#define RK3588_DISPLAY_RKNN_INFER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int left;
  int top;
  int right;
  int bottom;
  float score;
  char name[16];
} RknnDetection;

typedef struct {
  void *context;
  void *model_data;
  int model_width;
  int model_height;
  int output_count;
  void *output_attrs;
} RknnDetector;

typedef struct {
  double rga_fill_ms;
  double rga_preprocess_ms;
  double npu_run_ms;
  double output_sync_ms;
  double postprocess_ms;
} RknnStageTimes;

int rknn_detector_init(RknnDetector *detector, const char *model_path);
int rknn_detector_run(RknnDetector *detector, int nv12_fd,
                      int width, int height, int pitch,
                      RknnDetection *detections, int max_detections,
                      RknnStageTimes *stage_times);
int rga_draw_detection_boxes(uint32_t drm_handle, int width, int height, int pitch,
                             const RknnDetection *detections, int count);
void rknn_detector_deinit(RknnDetector *detector);

#ifdef __cplusplus
}
#endif

#endif