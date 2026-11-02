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

int rknn_detector_init(RknnDetector *detector, const char *model_path);
int rknn_detector_run(RknnDetector *detector, int nv12_fd,
                      int width, int height, int pitch,
                      RknnDetection *detections, int max_detections);
void rknn_detector_deinit(RknnDetector *detector);

#ifdef __cplusplus
}
#endif

#endif