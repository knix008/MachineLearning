#include "onnx_sr.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <onnxruntime_cxx_api.h>

namespace {

std::vector<const char*> AllocatedNames(std::vector<Ort::AllocatedStringPtr>& storage) {
  std::vector<const char*> out;
  out.reserve(storage.size());
  for (auto& s : storage) {
    out.push_back(s.get());
  }
  return out;
}

}  // namespace

struct OnnxSuperRes::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "realesrgan"};
  Ort::SessionOptions session_opts;
  std::unique_ptr<Ort::Session> session;
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<Ort::AllocatedStringPtr> input_names_storage;
  std::vector<Ort::AllocatedStringPtr> output_names_storage;
  std::vector<const char*> input_names;
  std::vector<const char*> output_names;
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  int64_t fixed_in_h = -1;
  int64_t fixed_in_w = -1;
};

OnnxSuperRes::OnnxSuperRes(const std::string& model_path) : impl_(std::make_unique<Impl>()) {
  impl_->session_opts.SetIntraOpNumThreads(0);
  impl_->session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
  std::wstring wpath(model_path.begin(), model_path.end());
  impl_->session = std::make_unique<Ort::Session>(impl_->env, wpath.c_str(),
                                                   impl_->session_opts);
#else
  impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(),
                                                   impl_->session_opts);
#endif

  const size_t in_count = impl_->session->GetInputCount();
  const size_t out_count = impl_->session->GetOutputCount();
  impl_->input_names_storage.reserve(in_count);
  impl_->output_names_storage.reserve(out_count);
  for (size_t i = 0; i < in_count; ++i) {
    impl_->input_names_storage.push_back(
        impl_->session->GetInputNameAllocated(i, impl_->allocator));
  }
  for (size_t i = 0; i < out_count; ++i) {
    impl_->output_names_storage.push_back(
        impl_->session->GetOutputNameAllocated(i, impl_->allocator));
  }
  impl_->input_names = AllocatedNames(impl_->input_names_storage);
  impl_->output_names = AllocatedNames(impl_->output_names_storage);

  Ort::TypeInfo in_type = impl_->session->GetInputTypeInfo(0);
  auto tensor_info = in_type.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> in_shape = tensor_info.GetShape();
  if (in_shape.size() == 4) {
    impl_->fixed_in_h = in_shape[2];
    impl_->fixed_in_w = in_shape[3];
  }

  // Try to infer scale from static output/input dimensions.
  Ort::TypeInfo out_type = impl_->session->GetOutputTypeInfo(0);
  auto out_tensor_info = out_type.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> out_shape = out_tensor_info.GetShape();
  if (in_shape.size() == 4 && out_shape.size() == 4 && in_shape[2] > 0 && in_shape[3] > 0 &&
      out_shape[2] > 0 && out_shape[3] > 0) {
    const int64_t sh = out_shape[2] / in_shape[2];
    const int64_t sw = out_shape[3] / in_shape[3];
    if (sh > 0 && sh == sw) {
      scale_ = static_cast<int>(sh);
    }
  }
}

OnnxSuperRes::~OnnxSuperRes() = default;

RgbImage OnnxSuperRes::run_once(const RgbImage& in, std::string& err_out) const {
  RgbImage out_img;
  if (in.width <= 0 || in.height <= 0 || in.data.empty()) {
    err_out = "empty input";
    return out_img;
  }

  const int64_t n = 1;
  const int64_t c = 3;
  const int64_t h = (impl_->fixed_in_h > 0) ? impl_->fixed_in_h : static_cast<int64_t>(in.height);
  const int64_t w = (impl_->fixed_in_w > 0) ? impl_->fixed_in_w : static_cast<int64_t>(in.width);
  if (in.height > h || in.width > w) {
    err_out = "input tile is larger than model fixed input. Reduce Tile size.";
    return out_img;
  }
  std::vector<int64_t> in_dims{n, c, h, w};

  std::vector<float> input_data(static_cast<size_t>(c) * static_cast<size_t>(h) *
                                static_cast<size_t>(w), 0.0f);
  for (int ch = 0; ch < 3; ++ch) {
    for (int y = 0; y < in.height; ++y) {
      for (int x = 0; x < in.width; ++x) {
        const size_t src_i =
            static_cast<size_t>(ch) * static_cast<size_t>(in.height) * static_cast<size_t>(in.width) +
            static_cast<size_t>(y) * static_cast<size_t>(in.width) + static_cast<size_t>(x);
        const size_t dst_i =
            static_cast<size_t>(ch) * static_cast<size_t>(h) * static_cast<size_t>(w) +
            static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x);
        input_data[dst_i] = in.data[src_i];
      }
    }
  }

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      impl_->mem_info, input_data.data(), input_data.size(), in_dims.data(),
      in_dims.size());

  try {
    auto outputs = impl_->session->Run(Ort::RunOptions{nullptr}, impl_->input_names.data(),
                                       &input_tensor, 1, impl_->output_names.data(),
                                       impl_->output_names.size());

    if (outputs.empty()) {
      err_out = "model returned no outputs";
      return out_img;
    }

    float* out_ptr = outputs[0].GetTensorMutableData<float>();
    auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> out_shape = out_info.GetShape();
    if (out_shape.size() != 4) {
      err_out = "expected NCHW output";
      return out_img;
    }
    const int64_t oh = out_shape[2];
    const int64_t ow = out_shape[3];
    if (oh <= 0 || ow <= 0) {
      err_out = "invalid output shape";
      return out_img;
    }
    const int out_w = static_cast<int>(ow);
    const int out_h = static_cast<int>(oh);
    const int scale_w = (w > 0) ? static_cast<int>(ow / w) : scale_;
    const int scale_h = (h > 0) ? static_cast<int>(oh / h) : scale_;
    if (scale_w <= 0 || scale_h <= 0 || scale_w != scale_h) {
      err_out = "invalid output/input scale from model";
      return RgbImage{};
    }
    const int crop_w = in.width * scale_w;
    const int crop_h = in.height * scale_h;
    if (crop_w <= 0 || crop_h <= 0 || crop_w > out_w || crop_h > out_h) {
      err_out = "invalid cropped output size";
      return RgbImage{};
    }
    out_img.width = crop_w;
    out_img.height = crop_h;
    out_img.data.assign(static_cast<size_t>(3) * static_cast<size_t>(crop_h) *
                            static_cast<size_t>(crop_w),
                        0.0f);
    for (int ch = 0; ch < 3; ++ch) {
      for (int y = 0; y < crop_h; ++y) {
        for (int x = 0; x < crop_w; ++x) {
          const size_t src_i =
              static_cast<size_t>(ch) * static_cast<size_t>(out_h) * static_cast<size_t>(out_w) +
              static_cast<size_t>(y) * static_cast<size_t>(out_w) + static_cast<size_t>(x);
          const size_t dst_i =
              static_cast<size_t>(ch) * static_cast<size_t>(crop_h) * static_cast<size_t>(crop_w) +
              static_cast<size_t>(y) * static_cast<size_t>(crop_w) + static_cast<size_t>(x);
          out_img.data[dst_i] = out_ptr[src_i];
        }
      }
    }
    if (static_cast<int>(out_shape[1]) != 3) {
      err_out = "expected 3-channel output";
      out_img = RgbImage{};
      return out_img;
    }
  } catch (const Ort::Exception& e) {
    err_out = std::string("onnxruntime: ") + e.what();
  }
  return out_img;
}

RgbImage OnnxSuperRes::upscale(const RgbImage& input, int tile_lr, int overlap_lr,
                               std::string& err_out) const {
  return upscale(input, tile_lr, overlap_lr, nullptr, err_out);
}

RgbImage OnnxSuperRes::upscale(const RgbImage& input, int tile_lr, int overlap_lr,
                               std::function<void(float)> progress_cb,
                               std::string& err_out) const {
  err_out.clear();
  const int iw = input.width;
  const int ih = input.height;
  if (iw <= 0 || ih <= 0) {
    err_out = "empty input image";
    return RgbImage{};
  }

  // If the model has fixed input size and full-image mode is requested,
  // auto-switch to tiled mode so we avoid invalid dimension errors.
  if (tile_lr <= 0 && impl_->fixed_in_h > 0 && impl_->fixed_in_w > 0 &&
      (iw > impl_->fixed_in_w || ih > impl_->fixed_in_h)) {
    tile_lr = static_cast<int>(std::min(impl_->fixed_in_w, impl_->fixed_in_h));
    overlap_lr = std::min(overlap_lr, std::max(0, tile_lr / 8));
  }

  if (tile_lr <= 0) {
    if (progress_cb) {
      progress_cb(0.1f);
    }
    RgbImage out = run_once(input, err_out);
    if (progress_cb) {
      progress_cb(err_out.empty() ? 1.0f : 0.0f);
    }
    return out;
  }

  // Clamp tile to safe limits based on image/model size.
  tile_lr = std::max(1, tile_lr);
  tile_lr = std::min(tile_lr, std::min(iw, ih));
  if (impl_->fixed_in_h > 0 && impl_->fixed_in_w > 0) {
    tile_lr = std::min(tile_lr, static_cast<int>(std::min(impl_->fixed_in_h, impl_->fixed_in_w)));
  }
  const int s = scale_;
  overlap_lr = std::max(0, std::min(overlap_lr, tile_lr / 2 - 1));
  const int stride = std::max(1, tile_lr - 2 * overlap_lr);

  const int ow_full = iw * s;
  const int oh_full = ih * s;

  RgbImage acc;
  acc.width = ow_full;
  acc.height = oh_full;
  acc.data.assign(static_cast<size_t>(3) * static_cast<size_t>(oh_full) *
                      static_cast<size_t>(ow_full),
                  0.0f);
  std::vector<float> weight(static_cast<size_t>(oh_full) * static_cast<size_t>(ow_full), 0.0f);
  const int tiles_x = (iw + stride - 1) / stride;
  const int tiles_y = (ih + stride - 1) / stride;
  const int total_tiles = std::max(1, tiles_x * tiles_y);
  int done_tiles = 0;

  auto copy_lr_region = [&](int x0, int y0, int tw, int th, RgbImage& dst) {
    dst.width = tw;
    dst.height = th;
    dst.data.resize(static_cast<size_t>(3) * static_cast<size_t>(th) * static_cast<size_t>(tw));
    for (int y = 0; y < th; ++y) {
      for (int x = 0; x < tw; ++x) {
        const int sx = x0 + x;
        const int sy = y0 + y;
        for (int ch = 0; ch < 3; ++ch) {
          const size_t src_i =
              static_cast<size_t>(ch) * static_cast<size_t>(ih) * static_cast<size_t>(iw) +
              static_cast<size_t>(sy) * static_cast<size_t>(iw) + static_cast<size_t>(sx);
          const size_t dst_i =
              static_cast<size_t>(ch) * static_cast<size_t>(th) * static_cast<size_t>(tw) +
              static_cast<size_t>(y) * static_cast<size_t>(tw) + static_cast<size_t>(x);
          dst.data[dst_i] = input.data[src_i];
        }
      }
    }
  };

  for (int y0 = 0; y0 < ih; y0 += stride) {
    for (int x0 = 0; x0 < iw; x0 += stride) {
      const int tw = std::min(tile_lr, iw - x0);
      const int th = std::min(tile_lr, ih - y0);
      RgbImage tile;
      copy_lr_region(x0, y0, tw, th, tile);
      std::string local_err;
      RgbImage hr = run_once(tile, local_err);
      if (!local_err.empty()) {
        err_out = local_err;
        return RgbImage{};
      }
      if (hr.width != tw * s || hr.height != th * s) {
        err_out = "tile output size mismatch (model scale != " + std::to_string(s) + "?)";
        return RgbImage{};
      }

      const int ox = x0 * s;
      const int oy = y0 * s;
      const bool at_left_lr = (x0 == 0);
      const bool at_right_lr = (x0 + tw >= iw);
      const bool at_top_lr = (y0 == 0);
      const bool at_bottom_lr = (y0 + th >= ih);
      for (int y = 0; y < hr.height; ++y) {
        for (int x = 0; x < hr.width; ++x) {
          const int gx = ox + x;
          const int gy = oy + y;
          if (gx < 0 || gy < 0 || gx >= ow_full || gy >= oh_full) {
            continue;
          }
          const size_t wi =
              static_cast<size_t>(gy) * static_cast<size_t>(ow_full) + static_cast<size_t>(gx);
          float wx = 1.0f;
          const int border = overlap_lr * s;
          if (border > 0) {
            const int lx = x;
            const int ly = y;
            if (!at_left_lr && lx < border) {
              wx *= static_cast<float>(lx + 1) / static_cast<float>(border + 1);
            }
            if (!at_right_lr && lx >= hr.width - border) {
              wx *= static_cast<float>(hr.width - lx) / static_cast<float>(border + 1);
            }
            if (!at_top_lr && ly < border) {
              wx *= static_cast<float>(ly + 1) / static_cast<float>(border + 1);
            }
            if (!at_bottom_lr && ly >= hr.height - border) {
              wx *= static_cast<float>(hr.height - ly) / static_cast<float>(border + 1);
            }
          }
          weight[wi] += wx;
          for (int ch = 0; ch < 3; ++ch) {
            const size_t ti =
                static_cast<size_t>(ch) * static_cast<size_t>(hr.height) *
                    static_cast<size_t>(hr.width) +
                static_cast<size_t>(y) * static_cast<size_t>(hr.width) + static_cast<size_t>(x);
            const size_t ai =
                static_cast<size_t>(ch) * static_cast<size_t>(oh_full) *
                    static_cast<size_t>(ow_full) +
                static_cast<size_t>(gy) * static_cast<size_t>(ow_full) + static_cast<size_t>(gx);
            acc.data[ai] += hr.data[ti] * wx;
          }
        }
      }
      ++done_tiles;
      if (progress_cb) {
        progress_cb(static_cast<float>(done_tiles) / static_cast<float>(total_tiles));
      }
    }
  }

  for (size_t i = 0; i < weight.size(); ++i) {
    if (weight[i] <= 1e-6f) {
      err_out = "internal blend error";
      return RgbImage{};
    }
  }
  for (int ch = 0; ch < 3; ++ch) {
    for (int y = 0; y < oh_full; ++y) {
      for (int x = 0; x < ow_full; ++x) {
        const size_t wi =
            static_cast<size_t>(y) * static_cast<size_t>(ow_full) + static_cast<size_t>(x);
        const size_t ai =
            static_cast<size_t>(ch) * static_cast<size_t>(oh_full) *
                static_cast<size_t>(ow_full) +
            wi;
        acc.data[ai] /= weight[wi];
      }
    }
  }

  return acc;
}
