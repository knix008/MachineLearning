#include "onnx_sr.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <curl/curl.h>
#include <gdk-pixbuf/gdk-pixbuf.h>
#include <gio/gio.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <gtk/gtk.h>

struct AppState {
  GtkWindow* window = nullptr;
  GtkEntry* entry_in = nullptr;
  GtkEntry* entry_out = nullptr;
  GtkEntry* entry_model = nullptr;
  GtkSpinButton* spin_tile = nullptr;
  GtkSpinButton* spin_overlap = nullptr;
  GtkSpinButton* spin_prepad = nullptr;
  GtkSpinButton* spin_zoom_in = nullptr;
  GtkSpinButton* spin_zoom_out = nullptr;
  GtkLabel* label_status = nullptr;
  GtkButton* btn_run = nullptr;
  GtkProgressBar* progress = nullptr;
  GtkPicture* picture_in = nullptr;
  GtkPicture* picture_out = nullptr;
  GtkLabel* label_in_size = nullptr;
  GtkLabel* label_out_size = nullptr;
  GtkScrolledWindow* scroll_in = nullptr;
  GtkScrolledWindow* scroll_out = nullptr;
  double drag_base_in_h = 0.0;
  double drag_base_in_v = 0.0;
  double drag_base_out_h = 0.0;
  double drag_base_out_v = 0.0;
  std::string last_input_path;
  std::string last_output_path;
  std::atomic<bool> busy{false};
};

struct JobPayload {
  AppState* app = nullptr;
  std::string path_in;
  std::string path_out;
  std::string path_model;
  int tile_lr = 0;
  int overlap_lr = 16;
  int pre_pad = 0;
};

struct UiProgressPayload {
  AppState* app = nullptr;
  double value = 0.0;
  std::string text;
  std::string path;
};

static void StartAutoModelLoad(AppState* st, const std::string& model_path);
static void PromptDownloadModelIfMissing(AppState* st);

struct DownloadDialogState {
  AppState* app = nullptr;
  GtkWindow* window = nullptr;
  GtkProgressBar* bar = nullptr;
  GtkLabel* label = nullptr;
  GtkButton* btn_ok = nullptr;
  std::atomic<bool> closed{false};
};

struct DownloadUiPayload {
  DownloadDialogState* dlg = nullptr;
  double fraction = -1.0;
  std::string text;
  bool done = false;
  bool success = false;
  std::string out_path;
};

static std::filesystem::path GetStateFilePath() {
  const char* cfg = g_get_user_config_dir();
  std::filesystem::path dir = cfg ? std::filesystem::path(cfg) : std::filesystem::path(".");
  dir /= "image_upscaler_gtk";
  return dir / "state.txt";
}

static void SaveModelPathState(const std::string& model_path) {
  if (model_path.empty()) return;
  try {
    const auto state_file = GetStateFilePath();
    std::filesystem::create_directories(state_file.parent_path());
    std::ofstream ofs(state_file, std::ios::trunc);
    if (ofs) {
      ofs << model_path << "\n";
    }
  } catch (...) {
    // Ignore state persistence failures.
  }
}

static std::string LoadModelPathState() {
  try {
    const auto state_file = GetStateFilePath();
    std::ifstream ifs(state_file);
    std::string line;
    if (ifs && std::getline(ifs, line)) {
      return line;
    }
  } catch (...) {
    // Ignore state load failures.
  }
  return "";
}

static void SetStatus(AppState* st, const char* text) {
  if (!st || !st->label_status) return;
  gtk_label_set_text(GTK_LABEL(st->label_status), text);
}

static std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

static const char* SaveFormatFromPath(const std::string& path) {
  const auto low = ToLower(path);
  if (low.size() >= 4 && low.substr(low.size() - 4) == ".png") {
    return "png";
  }
  if (low.size() >= 5 && (low.substr(low.size() - 5) == ".jpeg" || low.substr(low.size() - 4) == ".jpg")) {
    return "jpeg";
  }
  return "jpeg";
}

static std::string EnsureOutputExtension(std::string path) {
  const auto low = ToLower(path);
  const auto dot = low.find_last_of('.');
  const auto slash = low.find_last_of("/\\");
  const bool has_ext = (dot != std::string::npos) &&
                       (slash == std::string::npos || dot > slash);
  if (!has_ext) {
    path += ".jpg";
  }
  return path;
}

static void UpdateImageSizeLabel(GtkLabel* label, const char* title, const std::string& path) {
  if (!label) return;
  int w = 0;
  int h = 0;
  char buf[128];
  if (gdk_pixbuf_get_file_info(path.c_str(), &w, &h) && w > 0 && h > 0) {
    std::snprintf(buf, sizeof(buf), "%s: %d x %d", title, w, h);
  } else {
    std::snprintf(buf, sizeof(buf), "%s: -", title);
  }
  gtk_label_set_text(label, buf);
}

static double ComputeFitZoom(const std::string& path, GtkScrolledWindow* scroll) {
  int iw = 0;
  int ih = 0;
  if (!gdk_pixbuf_get_file_info(path.c_str(), &iw, &ih) || iw <= 0 || ih <= 0) {
    return 1.0;
  }
  const int vw = std::max(1, gtk_widget_get_width(GTK_WIDGET(scroll)) - 24);
  const int vh = std::max(1, gtk_widget_get_height(GTK_WIDGET(scroll)) - 24);
  double z = std::min(static_cast<double>(vw) / static_cast<double>(iw),
                      static_cast<double>(vh) / static_cast<double>(ih));
  z = std::max(0.1, std::min(8.0, z));
  return z;
}

static void UpdatePreview(GtkPicture* picture, const std::string& path, double zoom) {
  GError* err = nullptr;
  GdkPixbuf* src = gdk_pixbuf_new_from_file(path.c_str(), &err);
  if (!src) {
    if (err) {
      g_error_free(err);
    }
    return;
  }
  const int w = gdk_pixbuf_get_width(src);
  const int h = gdk_pixbuf_get_height(src);
  const int zw = std::max(1, static_cast<int>(std::lround(static_cast<double>(w) * zoom)));
  const int zh = std::max(1, static_cast<int>(std::lround(static_cast<double>(h) * zoom)));
  GdkPixbuf* scaled = gdk_pixbuf_scale_simple(src, zw, zh, GDK_INTERP_BILINEAR);
  g_object_unref(src);
  if (!scaled) {
    return;
  }
  const int rowstride = gdk_pixbuf_get_rowstride(scaled);
  const int n_channels = gdk_pixbuf_get_n_channels(scaled);
  const int height = gdk_pixbuf_get_height(scaled);
  const gsize byte_len = static_cast<gsize>(rowstride) * static_cast<gsize>(height);
  const guchar* pixels = gdk_pixbuf_get_pixels(scaled);
  guchar* pixels_copy = static_cast<guchar*>(g_memdup2(pixels, byte_len));
  GBytes* bytes = g_bytes_new_take(pixels_copy, byte_len);
  GdkMemoryFormat fmt = (n_channels == 4) ? GDK_MEMORY_R8G8B8A8 : GDK_MEMORY_R8G8B8;
  GdkTexture* tex =
      gdk_memory_texture_new(zw, zh, fmt, bytes, static_cast<gsize>(rowstride));
  g_bytes_unref(bytes);
  g_object_unref(scaled);
  gtk_picture_set_paintable(picture, GDK_PAINTABLE(tex));
  gtk_widget_set_size_request(GTK_WIDGET(picture), zw, zh);
  g_object_unref(tex);
}

static void SetBusy(AppState* st, bool busy) {
  st->busy = busy;
  gtk_widget_set_sensitive(GTK_WIDGET(st->btn_run), !busy);
  if (!busy) {
    gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(st->progress), 0.0);
    gtk_progress_bar_set_text(GTK_PROGRESS_BAR(st->progress), "0%");
  }
}

static std::string FormatEtaSeconds(double seconds) {
  if (seconds < 0.0 || !std::isfinite(seconds)) {
    return "--:--";
  }
  int s = static_cast<int>(std::lround(seconds));
  int h = s / 3600;
  s %= 3600;
  int m = s / 60;
  s %= 60;
  char buf[32];
  if (h > 0) {
    std::snprintf(buf, sizeof(buf), "%d:%02d:%02d", h, m, s);
  } else {
    std::snprintf(buf, sizeof(buf), "%02d:%02d", m, s);
  }
  return std::string(buf);
}

static bool PixbufToRgbFloat(GdkPixbuf* pb, RgbImage& out, GError** err) {
  if (!pb) {
    g_set_error(err, G_FILE_ERROR, G_FILE_ERROR_FAILED, "no pixbuf");
    return false;
  }
  if (gdk_pixbuf_get_colorspace(pb) != GDK_COLORSPACE_RGB) {
    g_set_error(err, G_FILE_ERROR, G_FILE_ERROR_FAILED, "unsupported colorspace");
    return false;
  }

  const int w = gdk_pixbuf_get_width(pb);
  const int h = gdk_pixbuf_get_height(pb);
  const int ch = gdk_pixbuf_get_n_channels(pb);
  if (ch != 3 && ch != 4) {
    g_set_error(err, G_FILE_ERROR, G_FILE_ERROR_FAILED, "need RGB or RGBA");
    return false;
  }
  if (gdk_pixbuf_get_bits_per_sample(pb) != 8) {
    g_set_error(err, G_FILE_ERROR, G_FILE_ERROR_FAILED, "need 8-bit channels");
    return false;
  }

  const guchar* pixels = gdk_pixbuf_get_pixels(pb);
  const int rowstride = gdk_pixbuf_get_rowstride(pb);
  out.width = w;
  out.height = h;
  out.data.resize(static_cast<size_t>(3) * static_cast<size_t>(w) * static_cast<size_t>(h));

  for (int y = 0; y < h; ++y) {
    const guchar* row = pixels + static_cast<size_t>(y) * static_cast<size_t>(rowstride);
    for (int x = 0; x < w; ++x) {
      const guchar* p = row + static_cast<size_t>(x) * static_cast<size_t>(ch);
      const float r = static_cast<float>(p[0]) / 255.0f;
      const float g = static_cast<float>(p[1]) / 255.0f;
      const float b = static_cast<float>(p[2]) / 255.0f;
      const size_t base = static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x);
      const size_t plane = static_cast<size_t>(w) * static_cast<size_t>(h);
      out.data[base] = r;
      out.data[plane + base] = g;
      out.data[2 * plane + base] = b;
    }
  }
  return true;
}

static void ChwToRgb888(const std::vector<float>& chw, int h, int w, std::vector<guchar>& rgb) {
  const size_t plane = static_cast<size_t>(h) * static_cast<size_t>(w);
  rgb.resize(plane * 3);
  const float* r = chw.data();
  const float* g = r + plane;
  const float* b = g + plane;
  for (size_t i = 0; i < plane; ++i) {
    auto clamp01 = [](float v) { return std::max(0.0f, std::min(1.0f, v)); };
    rgb[i * 3 + 0] = static_cast<guchar>(std::lround(clamp01(r[i]) * 255.0f));
    rgb[i * 3 + 1] = static_cast<guchar>(std::lround(clamp01(g[i]) * 255.0f));
    rgb[i * 3 + 2] = static_cast<guchar>(std::lround(clamp01(b[i]) * 255.0f));
  }
}

static GdkPixbuf* RgbFloatToPixbuf(const RgbImage& img, GError** err) {
  if (img.width <= 0 || img.height <= 0) {
    g_set_error(err, G_FILE_ERROR, G_FILE_ERROR_FAILED, "bad image");
    return nullptr;
  }

  std::vector<guchar> tmp;
  ChwToRgb888(img.data, img.height, img.width, tmp);
  const gsize n = static_cast<gsize>(tmp.size());
  guchar* copy = static_cast<guchar*>(g_malloc(n));
  memcpy(copy, tmp.data(), n);

  return gdk_pixbuf_new_from_data(
      copy, GDK_COLORSPACE_RGB, FALSE, 8, img.width, img.height, img.width * 3,
      +[](guchar* pixels, gpointer) { g_free(pixels); }, nullptr);
}

static RgbImage PadImageEdge(const RgbImage& in, int pad) {
  if (pad <= 0) {
    return in;
  }
  RgbImage out;
  out.width = in.width + pad * 2;
  out.height = in.height + pad * 2;
  out.data.assign(static_cast<size_t>(3) * static_cast<size_t>(out.width) *
                      static_cast<size_t>(out.height),
                  0.0f);

  for (int ch = 0; ch < 3; ++ch) {
    for (int y = 0; y < out.height; ++y) {
      const int sy = std::min(in.height - 1, std::max(0, y - pad));
      for (int x = 0; x < out.width; ++x) {
        const int sx = std::min(in.width - 1, std::max(0, x - pad));
        const size_t si =
            static_cast<size_t>(ch) * static_cast<size_t>(in.height) * static_cast<size_t>(in.width) +
            static_cast<size_t>(sy) * static_cast<size_t>(in.width) + static_cast<size_t>(sx);
        const size_t di =
            static_cast<size_t>(ch) * static_cast<size_t>(out.height) * static_cast<size_t>(out.width) +
            static_cast<size_t>(y) * static_cast<size_t>(out.width) + static_cast<size_t>(x);
        out.data[di] = in.data[si];
      }
    }
  }
  return out;
}

static RgbImage CropImage(const RgbImage& in, int x0, int y0, int w, int h) {
  RgbImage out;
  if (w <= 0 || h <= 0 || x0 < 0 || y0 < 0 || x0 + w > in.width || y0 + h > in.height) {
    return out;
  }
  out.width = w;
  out.height = h;
  out.data.assign(static_cast<size_t>(3) * static_cast<size_t>(w) * static_cast<size_t>(h), 0.0f);
  for (int ch = 0; ch < 3; ++ch) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const size_t si =
            static_cast<size_t>(ch) * static_cast<size_t>(in.height) * static_cast<size_t>(in.width) +
            static_cast<size_t>(y0 + y) * static_cast<size_t>(in.width) + static_cast<size_t>(x0 + x);
        const size_t di =
            static_cast<size_t>(ch) * static_cast<size_t>(h) * static_cast<size_t>(w) +
            static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x);
        out.data[di] = in.data[si];
      }
    }
  }
  return out;
}

static gboolean ApplyUiProgress(gpointer user_data) {
  std::unique_ptr<UiProgressPayload> p(static_cast<UiProgressPayload*>(user_data));
  if (p->value >= 0.0) {
    const double frac = std::max(0.0, std::min(1.0, p->value));
    gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(p->app->progress), frac);
  }
  if (!p->text.empty()) {
    SetStatus(p->app, p->text.c_str());
    gtk_progress_bar_set_text(GTK_PROGRESS_BAR(p->app->progress), p->text.c_str());
  }
  return G_SOURCE_REMOVE;
}

static void PostUiProgress(AppState* app, double value, const std::string& text) {
  auto* payload = new UiProgressPayload{app, value, text, ""};
  g_idle_add(ApplyUiProgress, payload);
}

static void FinishJob(AppState* app, const std::string& msg, const std::string& out_path_or_empty) {
  auto* payload = new UiProgressPayload{app, 0.0, msg, out_path_or_empty};
  g_idle_add(
      +[](gpointer data) -> gboolean {
        std::unique_ptr<UiProgressPayload> p(static_cast<UiProgressPayload*>(data));
        SetBusy(p->app, false);
        SetStatus(p->app, p->text.c_str());
        if (!p->path.empty()) {
          p->app->last_output_path = p->path;
          UpdateImageSizeLabel(p->app->label_out_size, "Output Size", p->path);
          const double fit = ComputeFitZoom(p->path, p->app->scroll_out);
          gtk_spin_button_set_value(p->app->spin_zoom_out, fit);
        }
        return G_SOURCE_REMOVE;
      },
      payload);
}

static void RunUpscaleThread(std::unique_ptr<JobPayload> job) {
  AppState* app = job->app;
  try {
    auto started = std::chrono::steady_clock::now();
    PostUiProgress(app, 0.02, "Loading model... ETA --:--");
    OnnxSuperRes engine(job->path_model);

    GError* err = nullptr;
    GdkPixbuf* in_pb = gdk_pixbuf_new_from_file(job->path_in.c_str(), &err);
    if (!in_pb) {
      std::string msg = std::string("load failed: ") + (err ? err->message : "?");
      if (err) g_error_free(err);
      FinishJob(app, msg, "");
      return;
    }

    RgbImage in;
    if (!PixbufToRgbFloat(in_pb, in, &err)) {
      g_object_unref(in_pb);
      std::string msg = std::string("convert failed: ") + (err ? err->message : "?");
      if (err) g_error_free(err);
      FinishJob(app, msg, "");
      return;
    }
    g_object_unref(in_pb);

    const int safe_prepad = std::max(0, job->pre_pad);
    RgbImage infer_in = (safe_prepad > 0) ? PadImageEdge(in, safe_prepad) : in;

    std::string infer_err;
    RgbImage out_full = engine.upscale(
        infer_in, job->tile_lr, job->overlap_lr,
        [app, started](float progress) {
          const double p = std::max(0.0, std::min(1.0, static_cast<double>(progress)));
          const auto now = std::chrono::steady_clock::now();
          const double elapsed =
              std::chrono::duration_cast<std::chrono::duration<double>>(now - started).count();
          double eta = -1.0;
          if (p > 1e-5) {
            eta = elapsed * (1.0 - p) / p;
          }
          const int pct = static_cast<int>(std::lround(p * 100.0));
          PostUiProgress(app, p,
                         "Upscaling... " + std::to_string(pct) + "% | ETA " +
                             FormatEtaSeconds(eta));
        },
        infer_err);

    RgbImage out = out_full;
    if (infer_err.empty() && safe_prepad > 0) {
      const int s = std::max(1, engine.scale());
      out = CropImage(out_full, safe_prepad * s, safe_prepad * s, in.width * s, in.height * s);
      if (out.data.empty()) {
        infer_err = "prepadding crop failed";
      }
    }

    if (!infer_err.empty()) {
      FinishJob(app, std::string("inference failed: ") + infer_err, "");
      return;
    }

    PostUiProgress(app, 0.98, "Saving output... ETA 00:00");
    GError* save_err = nullptr;
    GdkPixbuf* out_pb = RgbFloatToPixbuf(out, &save_err);
    if (!out_pb) {
      std::string msg = std::string("pixbuf failed: ") + (save_err ? save_err->message : "?");
      if (save_err) g_error_free(save_err);
      FinishJob(app, msg, "");
      return;
    }

    const char* format = SaveFormatFromPath(job->path_out);
    gboolean ok = FALSE;
    if (strcmp(format, "jpeg") == 0) {
      ok = gdk_pixbuf_save(out_pb, job->path_out.c_str(), "jpeg", &save_err, "quality", "95", nullptr);
    } else {
      ok = gdk_pixbuf_save(out_pb, job->path_out.c_str(), "png", &save_err, nullptr);
    }
    g_object_unref(out_pb);
    if (!ok) {
      std::string msg = std::string("save failed: ") + (save_err ? save_err->message : "?");
      if (save_err) g_error_free(save_err);
      FinishJob(app, msg, "");
      return;
    }

    FinishJob(app, "Saved: " + job->path_out, job->path_out);
  } catch (const std::exception& e) {
    FinishJob(app, std::string("error: ") + e.what(), "");
  }
}

static void OnRunClicked(GtkButton*, gpointer user_data) {
  auto* st = static_cast<AppState*>(user_data);
  if (st->busy.exchange(true)) return;

  const char* pin = gtk_editable_get_text(GTK_EDITABLE(st->entry_in));
  const char* pout = gtk_editable_get_text(GTK_EDITABLE(st->entry_out));
  const char* pmodel = gtk_editable_get_text(GTK_EDITABLE(st->entry_model));
  if (!pin || !*pin || !pout || !*pout || !pmodel || !*pmodel) {
    st->busy = false;
    SetStatus(st, "Fill input/output/model paths.");
    return;
  }

  auto job = std::make_unique<JobPayload>();
  job->app = st;
  job->path_in = pin;
  job->path_out = EnsureOutputExtension(pout);
  job->path_model = pmodel;
  job->tile_lr = gtk_spin_button_get_value_as_int(st->spin_tile);
  job->overlap_lr = gtk_spin_button_get_value_as_int(st->spin_overlap);
  job->pre_pad = gtk_spin_button_get_value_as_int(st->spin_prepad);

  st->last_input_path = pin;
  UpdateImageSizeLabel(st->label_in_size, "Input Size", st->last_input_path);
  gtk_editable_set_text(GTK_EDITABLE(st->entry_out), job->path_out.c_str());
  gtk_spin_button_set_value(st->spin_zoom_in, ComputeFitZoom(st->last_input_path, st->scroll_in));
  SetBusy(st, true);
  SetStatus(st, "Upscaling...");
  std::thread(RunUpscaleThread, std::move(job)).detach();
}

static void OnBrowseInput(GtkButton*, gpointer user_data) {
  auto* st = static_cast<AppState*>(user_data);
  GtkFileDialog* d = gtk_file_dialog_new();
  gtk_file_dialog_open(
      d, st->window, nullptr,
      +[](GObject* src, GAsyncResult* res, gpointer ud) {
        auto* s = static_cast<AppState*>(ud);
        GError* err = nullptr;
        GFile* f = gtk_file_dialog_open_finish(GTK_FILE_DIALOG(src), res, &err);
        if (f) {
          char* p = g_file_get_path(f);
          if (p) {
            gtk_editable_set_text(GTK_EDITABLE(s->entry_in), p);
            s->last_input_path = p;
            UpdateImageSizeLabel(s->label_in_size, "Input Size", s->last_input_path);
            gtk_spin_button_set_value(s->spin_zoom_in,
                                      ComputeFitZoom(s->last_input_path, s->scroll_in));
            g_free(p);
          }
          g_object_unref(f);
        }
        if (err) {
          g_error_free(err);
        }
        g_object_unref(src);
      },
      st);
}

static void OnBrowseOutput(GtkButton*, gpointer user_data) {
  auto* st = static_cast<AppState*>(user_data);
  GtkFileDialog* d = gtk_file_dialog_new();
  gtk_file_dialog_set_initial_name(d, "upscaled.jpg");
  gtk_file_dialog_save(
      d, st->window, nullptr,
      +[](GObject* src, GAsyncResult* res, gpointer ud) {
        auto* s = static_cast<AppState*>(ud);
        GError* err = nullptr;
        GFile* f = gtk_file_dialog_save_finish(GTK_FILE_DIALOG(src), res, &err);
        if (f) {
          char* p = g_file_get_path(f);
          if (p) {
            std::string out = EnsureOutputExtension(p);
            gtk_editable_set_text(GTK_EDITABLE(s->entry_out), out.c_str());
            g_free(p);
          }
          g_object_unref(f);
        }
        if (err) {
          g_error_free(err);
        }
        g_object_unref(src);
      },
      st);
}

static void OnBrowseModel(GtkButton*, gpointer user_data) {
  auto* st = static_cast<AppState*>(user_data);
  GtkFileDialog* d = gtk_file_dialog_new();
  gtk_file_dialog_open(
      d, st->window, nullptr,
      +[](GObject* src, GAsyncResult* res, gpointer ud) {
        auto* s = static_cast<AppState*>(ud);
        GError* err = nullptr;
        GFile* f = gtk_file_dialog_open_finish(GTK_FILE_DIALOG(src), res, &err);
        if (f) {
          char* p = g_file_get_path(f);
          if (p) {
            gtk_editable_set_text(GTK_EDITABLE(s->entry_model), p);
            SaveModelPathState(p);
            StartAutoModelLoad(s, p);
            g_free(p);
          }
          g_object_unref(f);
        }
        if (err) {
          g_error_free(err);
        }
        g_object_unref(src);
      },
      st);
}

static void OnModelPathChanged(GtkEditable* editable, gpointer) {
  const char* p = gtk_editable_get_text(editable);
  if (p && *p) {
    SaveModelPathState(p);
  }
}

static void StartAutoModelLoad(AppState* st, const std::string& model_path) {
  if (model_path.empty()) return;
  if (!std::filesystem::exists(model_path)) {
    SetStatus(st, "Model path set (file not found).");
    return;
  }
  SetStatus(st, "Loading model...");
  std::thread([st, model_path]() {
    std::string msg;
    try {
      OnnxSuperRes test(model_path);
      msg = "Model loaded: " + model_path;
    } catch (const std::exception& e) {
      msg = std::string("Model load failed: ") + e.what();
    }
    auto* payload = new UiProgressPayload{st, -1.0, msg, ""};
    g_idle_add(
        +[](gpointer data) -> gboolean {
          std::unique_ptr<UiProgressPayload> p(static_cast<UiProgressPayload*>(data));
          SetStatus(p->app, p->text.c_str());
          return G_SOURCE_REMOVE;
        },
        payload);
  }).detach();
}

static void StartModelDownload(AppState* st, const std::string& url, const std::string& out_path) {
  SetBusy(st, true);
  SetStatus(st, "Downloading default model...");

  auto* dlg = new DownloadDialogState();
  dlg->app = st;
  GtkWidget* win = gtk_window_new();
  dlg->window = GTK_WINDOW(win);
  gtk_window_set_title(dlg->window, "Downloading Model");
  gtk_window_set_transient_for(dlg->window, st->window);
  gtk_window_set_modal(dlg->window, TRUE);
  gtk_window_set_default_size(dlg->window, 420, 120);
  GtkWidget* box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
  gtk_widget_set_margin_top(box, 12);
  gtk_widget_set_margin_bottom(box, 12);
  gtk_widget_set_margin_start(box, 12);
  gtk_widget_set_margin_end(box, 12);
  GtkWidget* url_label = gtk_label_new(("URL: " + url).c_str());
  gtk_label_set_xalign(GTK_LABEL(url_label), 0.0f);
  gtk_label_set_wrap(GTK_LABEL(url_label), TRUE);
  dlg->label = GTK_LABEL(gtk_label_new("Downloading... 0%"));
  gtk_label_set_xalign(dlg->label, 0.0f);
  dlg->bar = GTK_PROGRESS_BAR(gtk_progress_bar_new());
  gtk_progress_bar_set_show_text(dlg->bar, TRUE);
  gtk_progress_bar_set_fraction(dlg->bar, 0.0);
  gtk_progress_bar_set_text(dlg->bar, "0%");
  dlg->btn_ok = GTK_BUTTON(gtk_button_new_with_label("확인"));
  gtk_widget_set_sensitive(GTK_WIDGET(dlg->btn_ok), FALSE);
  g_signal_connect(
      dlg->btn_ok, "clicked",
      G_CALLBACK(+[](GtkButton*, gpointer data) {
        auto* d = static_cast<DownloadDialogState*>(data);
        d->closed.store(true);
        gtk_window_destroy(d->window);
        delete d;
      }),
      dlg);
  gtk_box_append(GTK_BOX(box), url_label);
  gtk_box_append(GTK_BOX(box), GTK_WIDGET(dlg->label));
  gtk_box_append(GTK_BOX(box), GTK_WIDGET(dlg->bar));
  gtk_box_append(GTK_BOX(box), GTK_WIDGET(dlg->btn_ok));
  gtk_window_set_child(dlg->window, box);
  gtk_window_present(dlg->window);

  auto post_ui = [](DownloadDialogState* d, double frac, const std::string& text, bool done,
                    bool success, const std::string& path) {
    auto* payload = new DownloadUiPayload{d, frac, text, done, success, path};
    g_idle_add(
        +[](gpointer data) -> gboolean {
          std::unique_ptr<DownloadUiPayload> p(static_cast<DownloadUiPayload*>(data));
          if (!p->dlg || p->dlg->closed.load()) {
            return G_SOURCE_REMOVE;
          }
          if (!p->dlg->label || !GTK_IS_LABEL(p->dlg->label) || !p->dlg->bar ||
              !GTK_IS_PROGRESS_BAR(p->dlg->bar)) {
            return G_SOURCE_REMOVE;
          }
          if (p->fraction >= 0.0) {
            const double clamped = std::max(0.0, std::min(1.0, p->fraction));
            gtk_progress_bar_set_fraction(p->dlg->bar, clamped);
            const int pct = static_cast<int>(std::lround(clamped * 100.0));
            gtk_progress_bar_set_text(p->dlg->bar, (std::to_string(pct) + "%").c_str());
          }
          if (!p->text.empty()) {
            gtk_label_set_text(p->dlg->label, p->text.c_str());
          }
          if (p->done) {
            SetBusy(p->dlg->app, false);
            SetStatus(p->dlg->app, p->text.c_str());
            if (p->success) {
              gtk_editable_set_text(GTK_EDITABLE(p->dlg->app->entry_model), p->out_path.c_str());
              SaveModelPathState(p->out_path);
              StartAutoModelLoad(p->dlg->app, p->out_path);
            }
            if (p->dlg->btn_ok && GTK_IS_BUTTON(p->dlg->btn_ok)) {
              gtk_widget_set_sensitive(GTK_WIDGET(p->dlg->btn_ok), TRUE);
            }
          }
          return G_SOURCE_REMOVE;
        },
        payload);
  };

  std::thread([url, out_path, dlg, post_ui]() {
    try {
      std::filesystem::create_directories(std::filesystem::path(out_path).parent_path());
    } catch (const std::exception& e) {
      post_ui(dlg, 0.0, std::string("Model download failed: ") + e.what(), true, false, "");
      return;
    }

    FILE* fp = fopen(out_path.c_str(), "wb");
    if (!fp) {
      post_ui(dlg, 0.0, "Model download failed: cannot open output file", true, false, "");
      return;
    }

    CURL* curl = curl_easy_init();
    if (!curl) {
      fclose(fp);
      post_ui(dlg, 0.0, "Model download failed: curl init failed", true, false, "");
      return;
    }

    struct ProgressCtx {
      DownloadDialogState* dlg;
      decltype(post_ui) post;
    } ctx{dlg, post_ui};

    auto write_cb = +[](void* ptr, size_t size, size_t nmemb, void* userdata) -> size_t {
      FILE* f = static_cast<FILE*>(userdata);
      return fwrite(ptr, size, nmemb, f);
    };
    auto prog_cb = +[](void* user, curl_off_t total, curl_off_t now, curl_off_t, curl_off_t) -> int {
      auto* c = static_cast<ProgressCtx*>(user);
      if (total > 0) {
        const double frac = static_cast<double>(now) / static_cast<double>(total);
        const int pct = static_cast<int>(std::lround(frac * 100.0));
        c->post(c->dlg, frac, "Downloading... " + std::to_string(pct) + "%", false, false, "");
      }
      return 0;
    };

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, prog_cb);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &ctx);
    CURLcode rc = curl_easy_perform(curl);
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    curl_easy_cleanup(curl);
    fclose(fp);

    if (rc == CURLE_OK && status >= 200 && status < 300 && std::filesystem::exists(out_path)) {
      post_ui(dlg, 1.0, "Model downloaded.", true, true, out_path);
    } else {
      g_remove(out_path.c_str());
      std::string why = (rc != CURLE_OK) ? curl_easy_strerror(rc)
                                         : ("HTTP " + std::to_string(status));
      post_ui(dlg, 0.0, "Model download failed: " + why, true, false, "");
    }
  }).detach();
}

static void PromptDownloadModelIfMissing(AppState* st) {
  const std::string target_path = "model/realesrgan-x4.onnx";
  const char* model_path = gtk_editable_get_text(GTK_EDITABLE(st->entry_model));
  if (model_path && *model_path && std::filesystem::exists(model_path)) {
    return;
  }

  GtkWidget* dialog = gtk_window_new();
  gtk_window_set_title(GTK_WINDOW(dialog), "Model Not Found");
  gtk_window_set_transient_for(GTK_WINDOW(dialog), st->window);
  gtk_window_set_modal(GTK_WINDOW(dialog), TRUE);
  gtk_window_set_default_size(GTK_WINDOW(dialog), 440, 140);

  GtkWidget* root = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
  gtk_widget_set_margin_top(root, 14);
  gtk_widget_set_margin_bottom(root, 14);
  gtk_widget_set_margin_start(root, 14);
  gtk_widget_set_margin_end(root, 14);
  std::string popup_text =
      "기본 RealESRGAN ONNX 모델이 없습니다.\n"
      "다운로드 경로: " +
      target_path + "\n지금 다운로드하시겠습니까?";
  GtkWidget* lbl = gtk_label_new(popup_text.c_str());
  gtk_label_set_xalign(GTK_LABEL(lbl), 0.0f);
  gtk_label_set_wrap(GTK_LABEL(lbl), TRUE);
  gtk_box_append(GTK_BOX(root), lbl);

  GtkWidget* row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
  GtkWidget* btn_cancel = gtk_button_new_with_label("Cancel");
  GtkWidget* btn_download = gtk_button_new_with_label("Download");
  gtk_widget_add_css_class(btn_download, "suggested-action");
  gtk_box_append(GTK_BOX(row), btn_cancel);
  gtk_box_append(GTK_BOX(row), btn_download);
  gtk_box_append(GTK_BOX(root), row);
  gtk_window_set_child(GTK_WINDOW(dialog), root);

  struct DownloadDialogCtx {
    AppState* app;
    GtkWindow* dialog;
    std::string target_path;
  };
  auto* ctx = new DownloadDialogCtx{st, GTK_WINDOW(dialog), target_path};

  g_signal_connect(
      btn_cancel, "clicked",
      G_CALLBACK(+[](GtkButton*, gpointer data) {
        auto* c = static_cast<DownloadDialogCtx*>(data);
        gtk_window_destroy(c->dialog);
        delete c;
      }),
      ctx);
  g_signal_connect(
      btn_download, "clicked",
      G_CALLBACK(+[](GtkButton*, gpointer data) {
        auto* c = static_cast<DownloadDialogCtx*>(data);
        gtk_window_destroy(c->dialog);
        StartModelDownload(
            c->app,
            "https://huggingface.co/AXERA-TECH/Real-ESRGAN/resolve/main/onnx/realesrgan-x4.onnx",
            c->target_path);
        delete c;
      }),
      ctx);
  gtk_window_present(GTK_WINDOW(dialog));
}

static GtkWidget* BuildPathRow(const char* label, GtkEntry** out_entry, const char* btn_text,
                               GCallback on_click, AppState* st) {
  GtkWidget* row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
  GtkWidget* l = gtk_label_new(label);
  gtk_widget_set_size_request(l, 90, -1);
  gtk_box_append(GTK_BOX(row), l);

  GtkWidget* entry = gtk_entry_new();
  gtk_widget_set_hexpand(entry, TRUE);
  gtk_box_append(GTK_BOX(row), entry);
  *out_entry = GTK_ENTRY(entry);

  GtkWidget* btn = gtk_button_new_with_label(btn_text);
  g_signal_connect(btn, "clicked", on_click, st);
  gtk_box_append(GTK_BOX(row), btn);
  return row;
}

static void OnInputZoomChanged(GtkSpinButton* spin, gpointer user_data) {
  auto* st = static_cast<AppState*>(user_data);
  if (!st->last_input_path.empty()) {
    UpdatePreview(st->picture_in, st->last_input_path, gtk_spin_button_get_value(spin));
  }
}

static void OnOutputZoomChanged(GtkSpinButton* spin, gpointer user_data) {
  auto* st = static_cast<AppState*>(user_data);
  if (!st->last_output_path.empty()) {
    UpdatePreview(st->picture_out, st->last_output_path, gtk_spin_button_get_value(spin));
  }
}

static void OnFitInputClicked(GtkButton*, gpointer user_data) {
  auto* st = static_cast<AppState*>(user_data);
  if (st->last_input_path.empty()) return;
  const double z = ComputeFitZoom(st->last_input_path, st->scroll_in);
  gtk_spin_button_set_value(st->spin_zoom_in, z);
}

static void OnFitOutputClicked(GtkButton*, gpointer user_data) {
  auto* st = static_cast<AppState*>(user_data);
  if (st->last_output_path.empty()) return;
  const double z = ComputeFitZoom(st->last_output_path, st->scroll_out);
  gtk_spin_button_set_value(st->spin_zoom_out, z);
}

static gboolean OnInputPreviewScroll(GtkEventControllerScroll*, double, double dy, gpointer user_data) {
  auto* st = static_cast<AppState*>(user_data);
  double v = gtk_spin_button_get_value(st->spin_zoom_in);
  v = (dy < 0.0) ? (v * 1.1) : (v / 1.1);
  v = std::max(0.1, std::min(8.0, v));
  gtk_spin_button_set_value(st->spin_zoom_in, v);
  return TRUE;
}

static gboolean OnOutputPreviewScroll(GtkEventControllerScroll*, double, double dy, gpointer user_data) {
  auto* st = static_cast<AppState*>(user_data);
  double v = gtk_spin_button_get_value(st->spin_zoom_out);
  v = (dy < 0.0) ? (v * 1.1) : (v / 1.1);
  v = std::max(0.1, std::min(8.0, v));
  gtk_spin_button_set_value(st->spin_zoom_out, v);
  return TRUE;
}

static void UpdateDragOnScroll(GtkScrolledWindow* scroll, double base_h, double base_v, double dx, double dy) {
  GtkAdjustment* hadj = gtk_scrolled_window_get_hadjustment(scroll);
  GtkAdjustment* vadj = gtk_scrolled_window_get_vadjustment(scroll);
  const double h_lower = gtk_adjustment_get_lower(hadj);
  const double h_upper = gtk_adjustment_get_upper(hadj) - gtk_adjustment_get_page_size(hadj);
  const double v_lower = gtk_adjustment_get_lower(vadj);
  const double v_upper = gtk_adjustment_get_upper(vadj) - gtk_adjustment_get_page_size(vadj);
  const double next_h = std::max(h_lower, std::min(h_upper, base_h - dx));
  const double next_v = std::max(v_lower, std::min(v_upper, base_v - dy));
  gtk_adjustment_set_value(hadj, next_h);
  gtk_adjustment_set_value(vadj, next_v);
}

static void OnInputDragBegin(GtkGestureDrag*, double, double, gpointer user_data) {
  auto* st = static_cast<AppState*>(user_data);
  GtkAdjustment* hadj = gtk_scrolled_window_get_hadjustment(st->scroll_in);
  GtkAdjustment* vadj = gtk_scrolled_window_get_vadjustment(st->scroll_in);
  st->drag_base_in_h = gtk_adjustment_get_value(hadj);
  st->drag_base_in_v = gtk_adjustment_get_value(vadj);
}

static void OnInputDragUpdate(GtkGestureDrag*, double dx, double dy, gpointer user_data) {
  auto* st = static_cast<AppState*>(user_data);
  UpdateDragOnScroll(st->scroll_in, st->drag_base_in_h, st->drag_base_in_v, dx, dy);
}

static void OnOutputDragBegin(GtkGestureDrag*, double, double, gpointer user_data) {
  auto* st = static_cast<AppState*>(user_data);
  GtkAdjustment* hadj = gtk_scrolled_window_get_hadjustment(st->scroll_out);
  GtkAdjustment* vadj = gtk_scrolled_window_get_vadjustment(st->scroll_out);
  st->drag_base_out_h = gtk_adjustment_get_value(hadj);
  st->drag_base_out_v = gtk_adjustment_get_value(vadj);
}

static void OnOutputDragUpdate(GtkGestureDrag*, double dx, double dy, gpointer user_data) {
  auto* st = static_cast<AppState*>(user_data);
  UpdateDragOnScroll(st->scroll_out, st->drag_base_out_h, st->drag_base_out_v, dx, dy);
}

static void OnActivate(GtkApplication* app, gpointer) {
  auto* st = new AppState();

  GtkWidget* win = gtk_application_window_new(app);
  st->window = GTK_WINDOW(win);
  gtk_window_set_title(st->window, "RealESRGAN ONNX Upscaler");
  gtk_window_set_default_size(st->window, 1000, 700);

  GtkWidget* root = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
  gtk_widget_set_margin_top(root, 12);
  gtk_widget_set_margin_bottom(root, 12);
  gtk_widget_set_margin_start(root, 12);
  gtk_widget_set_margin_end(root, 12);

  gtk_box_append(GTK_BOX(root), BuildPathRow("Input", &st->entry_in, "Browse", G_CALLBACK(OnBrowseInput), st));
  gtk_box_append(GTK_BOX(root), BuildPathRow("Output (JPG)", &st->entry_out, "Browse", G_CALLBACK(OnBrowseOutput), st));
  gtk_entry_set_placeholder_text(st->entry_out, "upscaled.jpg");
  gtk_box_append(GTK_BOX(root), BuildPathRow("Model", &st->entry_model, "Browse", G_CALLBACK(OnBrowseModel), st));
  g_signal_connect(st->entry_model, "changed", G_CALLBACK(OnModelPathChanged), nullptr);
  std::string deferred_model_load;
  const std::string saved_model = LoadModelPathState();
  if (!saved_model.empty() && std::filesystem::exists(saved_model)) {
    gtk_editable_set_text(GTK_EDITABLE(st->entry_model), saved_model.c_str());
    deferred_model_load = saved_model;
  } else if (std::filesystem::exists("model/realesrgan-x4.onnx")) {
    gtk_editable_set_text(GTK_EDITABLE(st->entry_model), "model/realesrgan-x4.onnx");
    SaveModelPathState("model/realesrgan-x4.onnx");
    deferred_model_load = "model/realesrgan-x4.onnx";
  } else if (!saved_model.empty()) {
    gtk_editable_set_text(GTK_EDITABLE(st->entry_model), saved_model.c_str());
    SetStatus(st, "Saved model path restored (file not found).");
  } else {
    gtk_editable_set_text(GTK_EDITABLE(st->entry_model), "model/realesrgan-x4.onnx");
  }

  GtkWidget* setting_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
  st->spin_tile = GTK_SPIN_BUTTON(gtk_spin_button_new_with_range(0, 4096, 16));
  gtk_spin_button_set_value(st->spin_tile, 0);
  st->spin_overlap = GTK_SPIN_BUTTON(gtk_spin_button_new_with_range(0, 256, 4));
  gtk_spin_button_set_value(st->spin_overlap, 10);
  st->spin_prepad = GTK_SPIN_BUTTON(gtk_spin_button_new_with_range(0, 256, 1));
  gtk_spin_button_set_value(st->spin_prepad, 0);
  st->spin_zoom_in = GTK_SPIN_BUTTON(gtk_spin_button_new_with_range(0.1, 8.0, 0.1));
  gtk_spin_button_set_value(st->spin_zoom_in, 1.0);
  st->spin_zoom_out = GTK_SPIN_BUTTON(gtk_spin_button_new_with_range(0.1, 8.0, 0.1));
  gtk_spin_button_set_value(st->spin_zoom_out, 1.0);
  g_signal_connect(st->spin_zoom_in, "value-changed", G_CALLBACK(OnInputZoomChanged), st);
  g_signal_connect(st->spin_zoom_out, "value-changed", G_CALLBACK(OnOutputZoomChanged), st);
  gtk_box_append(GTK_BOX(setting_row), gtk_label_new("Tile"));
  gtk_box_append(GTK_BOX(setting_row), GTK_WIDGET(st->spin_tile));
  gtk_box_append(GTK_BOX(setting_row), gtk_label_new("Padding"));
  gtk_box_append(GTK_BOX(setting_row), GTK_WIDGET(st->spin_overlap));
  gtk_box_append(GTK_BOX(setting_row), gtk_label_new("Prepadding"));
  gtk_box_append(GTK_BOX(setting_row), GTK_WIDGET(st->spin_prepad));
  gtk_box_append(GTK_BOX(setting_row), gtk_label_new("Input Zoom"));
  gtk_box_append(GTK_BOX(setting_row), GTK_WIDGET(st->spin_zoom_in));
  GtkWidget* btn_fit_in = gtk_button_new_with_label("Fit In");
  g_signal_connect(btn_fit_in, "clicked", G_CALLBACK(OnFitInputClicked), st);
  gtk_box_append(GTK_BOX(setting_row), btn_fit_in);
  gtk_box_append(GTK_BOX(setting_row), gtk_label_new("Output Zoom"));
  gtk_box_append(GTK_BOX(setting_row), GTK_WIDGET(st->spin_zoom_out));
  GtkWidget* btn_fit_out = gtk_button_new_with_label("Fit Out");
  g_signal_connect(btn_fit_out, "clicked", G_CALLBACK(OnFitOutputClicked), st);
  gtk_box_append(GTK_BOX(setting_row), btn_fit_out);
  st->btn_run = GTK_BUTTON(gtk_button_new_with_label("Upscale"));
  gtk_widget_add_css_class(GTK_WIDGET(st->btn_run), "suggested-action");
  gtk_widget_add_css_class(GTK_WIDGET(st->btn_run), "upscale-green");
  g_signal_connect(st->btn_run, "clicked", G_CALLBACK(OnRunClicked), st);
  gtk_box_append(GTK_BOX(setting_row), GTK_WIDGET(st->btn_run));
  gtk_box_append(GTK_BOX(root), setting_row);

  GtkWidget* image_info_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 16);
  st->label_in_size = GTK_LABEL(gtk_label_new("Input Size: -"));
  st->label_out_size = GTK_LABEL(gtk_label_new("Output Size: -"));
  gtk_widget_set_halign(GTK_WIDGET(st->label_in_size), GTK_ALIGN_START);
  gtk_widget_set_halign(GTK_WIDGET(st->label_out_size), GTK_ALIGN_START);
  gtk_box_append(GTK_BOX(image_info_row), GTK_WIDGET(st->label_in_size));
  gtk_box_append(GTK_BOX(image_info_row), GTK_WIDGET(st->label_out_size));
  gtk_box_append(GTK_BOX(root), image_info_row);

  st->progress = GTK_PROGRESS_BAR(gtk_progress_bar_new());
  gtk_progress_bar_set_show_text(st->progress, TRUE);
  gtk_progress_bar_set_text(st->progress, "0%");
  gtk_box_append(GTK_BOX(root), GTK_WIDGET(st->progress));

  st->label_status = GTK_LABEL(gtk_label_new("Ready."));
  gtk_label_set_wrap(st->label_status, TRUE);
  gtk_box_append(GTK_BOX(root), GTK_WIDGET(st->label_status));

  GtkWidget* preview = gtk_paned_new(GTK_ORIENTATION_HORIZONTAL);
  st->picture_in = GTK_PICTURE(gtk_picture_new());
  st->picture_out = GTK_PICTURE(gtk_picture_new());
  gtk_picture_set_can_shrink(st->picture_in, FALSE);
  gtk_picture_set_can_shrink(st->picture_out, FALSE);
  gtk_widget_set_hexpand(GTK_WIDGET(st->picture_in), FALSE);
  gtk_widget_set_vexpand(GTK_WIDGET(st->picture_in), FALSE);
  gtk_widget_set_hexpand(GTK_WIDGET(st->picture_out), FALSE);
  gtk_widget_set_vexpand(GTK_WIDGET(st->picture_out), FALSE);
  GtkEventController* ctrl_in =
      gtk_event_controller_scroll_new(GTK_EVENT_CONTROLLER_SCROLL_VERTICAL);
  g_signal_connect(ctrl_in, "scroll", G_CALLBACK(OnInputPreviewScroll), st);
  gtk_widget_add_controller(GTK_WIDGET(st->picture_in), ctrl_in);
  GtkGesture* drag_in = gtk_gesture_drag_new();
  g_signal_connect(drag_in, "drag-begin", G_CALLBACK(OnInputDragBegin), st);
  g_signal_connect(drag_in, "drag-update", G_CALLBACK(OnInputDragUpdate), st);
  gtk_widget_add_controller(GTK_WIDGET(st->picture_in), GTK_EVENT_CONTROLLER(drag_in));
  GtkEventController* ctrl_out =
      gtk_event_controller_scroll_new(GTK_EVENT_CONTROLLER_SCROLL_VERTICAL);
  g_signal_connect(ctrl_out, "scroll", G_CALLBACK(OnOutputPreviewScroll), st);
  gtk_widget_add_controller(GTK_WIDGET(st->picture_out), ctrl_out);
  GtkGesture* drag_out = gtk_gesture_drag_new();
  g_signal_connect(drag_out, "drag-begin", G_CALLBACK(OnOutputDragBegin), st);
  g_signal_connect(drag_out, "drag-update", G_CALLBACK(OnOutputDragUpdate), st);
  gtk_widget_add_controller(GTK_WIDGET(st->picture_out), GTK_EVENT_CONTROLLER(drag_out));

  GtkWidget* in_scroll = gtk_scrolled_window_new();
  st->scroll_in = GTK_SCROLLED_WINDOW(in_scroll);
  gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(in_scroll), GTK_POLICY_AUTOMATIC,
                                 GTK_POLICY_AUTOMATIC);
  gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(in_scroll), GTK_WIDGET(st->picture_in));
  GtkWidget* in_frame = gtk_frame_new("Input Preview");
  gtk_frame_set_child(GTK_FRAME(in_frame), in_scroll);

  GtkWidget* out_scroll = gtk_scrolled_window_new();
  st->scroll_out = GTK_SCROLLED_WINDOW(out_scroll);
  gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(out_scroll), GTK_POLICY_AUTOMATIC,
                                 GTK_POLICY_AUTOMATIC);
  gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(out_scroll), GTK_WIDGET(st->picture_out));
  GtkWidget* out_frame = gtk_frame_new("Output Preview");
  gtk_frame_set_child(GTK_FRAME(out_frame), out_scroll);
  gtk_paned_set_start_child(GTK_PANED(preview), in_frame);
  gtk_paned_set_end_child(GTK_PANED(preview), out_frame);
  gtk_widget_set_vexpand(preview, TRUE);
  gtk_box_append(GTK_BOX(root), preview);

  gtk_window_set_child(st->window, root);
  {
    GtkCssProvider* css = gtk_css_provider_new();
    gtk_css_provider_load_from_string(
        css,
        ".upscale-green {"
        "  background: #2f81f7;"
        "  color: #ffffff;"
        "}"
        ".upscale-green:hover {"
        "  background: #58a6ff;"
        "}"
        ".upscale-green:disabled {"
        "  background: #a9c4ea;"
        "  color: #f5f8ff;"
        "}");
    gtk_style_context_add_provider_for_display(
        gdk_display_get_default(), GTK_STYLE_PROVIDER(css),
        GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
    g_object_unref(css);
  }
  g_signal_connect(
      win, "destroy", G_CALLBACK(+[](GtkWidget*, gpointer p) { delete static_cast<AppState*>(p); }), st);
  gtk_window_present(st->window);
  if (!deferred_model_load.empty()) {
    StartAutoModelLoad(st, deferred_model_load);
  }
  PromptDownloadModelIfMissing(st);
}

int main(int argc, char** argv) {
  GtkApplication* app = gtk_application_new("com.example.ImageUpscalerGTK", G_APPLICATION_DEFAULT_FLAGS);
  g_signal_connect(app, "activate", G_CALLBACK(OnActivate), nullptr);
  int status = g_application_run(G_APPLICATION(app), argc, argv);
  g_object_unref(app);
  return status;
}
