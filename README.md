# rk3588-display

## Camera to RKNN to DRM

The application captures NV12 from V4L2, rotates/scales it into the inactive
DRM NV12 framebuffer with RGA, copies that frame into an RGB letterboxed input
on the CPU, and submits it to RKNN. YOLOv5 detections are returned to the CPU,
drawn into the same DRM framebuffer, and displayed with a page flip.

Build with `make`. The Makefile defaults to the RKNN toolkit and RK3588 sysroot
paths used by this workspace; override `RKNN_ROOT`, `ROOT_SYSROOT`, and
`CROSS_COMPILE` when building elsewhere.

On the board, run the executable from a directory containing
`model/coco_80_labels_list.txt`. Put the RKNN model at the default path
`model/RK3588/yolov5s-640-640.rknn`, or provide another model path as the first
argument. The model must be the three-output RK3588 YOLOv5 model supported by
the bundled postprocessor.
