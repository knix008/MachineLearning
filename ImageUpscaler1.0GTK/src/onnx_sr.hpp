#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct RgbImage {
  int width = 0;
  int height = 0;
  std::vector<float> data;  // CHW, RGB, [0, 1]
};

class OnnxSuperRes {
 public:
  explicit OnnxSuperRes(const std::string& model_path);
  ~OnnxSuperRes();

  OnnxSuperRes(const OnnxSuperRes&) = delete;
  OnnxSuperRes& operator=(const OnnxSuperRes&) = delete;

  // Upscale by network scale (typically 4). Uses LR-space tiling when tile_lr > 0.
  RgbImage upscale(const RgbImage& input, int tile_lr, int overlap_lr,
                   std::string& err_out) const;
  RgbImage upscale(const RgbImage& input, int tile_lr, int overlap_lr,
                   std::function<void(float)> progress_cb,
                   std::string& err_out) const;

  int scale() const { return scale_; }

 private:
  RgbImage run_once(const RgbImage& tile, std::string& err_out) const;

  struct Impl;
  std::unique_ptr<Impl> impl_;
  int scale_ = 4;
};
