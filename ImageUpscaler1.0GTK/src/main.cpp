#include "onnx_sr.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
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
  GtkEntry* entry_model_url = nullptr;
  GtkSpinButton* spin_tile = nullptr;
  GtkSpinButton* spin_overlap = nullptr;
  GtkSpinButton* spin_zoom_in = nullptr;
  GtkSpinButton* spin_zoom_out = nullptr;
  GtkLabel* label_status = nullptr;
  GtkButton* btn_run = nullptr;
  GtkButton* btn_download = nullptr;
  GtkProgressBar* progress = nullptr;
  GtkPicture* picture_in = nullptr;
  GtkPicture* picture_out = nullptr;
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
};

struct DownloadPayload {
  AppState* app = nullptr;
  std::string url;
  std::string out_path;
};

struct UiProgressPayload {
  AppState* app = nullptr;
  double value = 0.0;
  std::string text;
  std::string path;
};

static void AutoTuneParamsForImage(AppState* st, const std::string& image_path);
static void NormalizeTileParams(AppState* st, const std::string& image_path);

static void SetStatus(AppState* st, const char* text) {
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
  gtk_widget_set_sensitive(GTK_WIDGET(st->btn_download), !busy);
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
          const double zoom = gtk_spin_button_get_value(GTK_SPIN_BUTTON(p->app->spin_zoom_out));
          UpdatePreview(GTK_PICTURE(p->app->picture_out), p->path, zoom);
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

    std::string infer_err;
    RgbImage out = engine.upscale(
        in, job->tile_lr, job->overlap_lr,
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

struct CurlProgressCtx {
  AppState* app = nullptr;
  std::chrono::steady_clock::time_point started;
};

static int CurlXferInfo(void* clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t, curl_off_t) {
  auto* ctx = static_cast<CurlProgressCtx*>(clientp);
  if (!ctx || !ctx->app) return 0;
  if (dltotal > 0) {
    const double frac = static_cast<double>(dlnow) / static_cast<double>(dltotal);
    const auto now = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(now - ctx->started).count();
    double eta = -1.0;
    if (frac > 1e-5) {
      eta = elapsed * (1.0 - frac) / frac;
    }
    const int pct = static_cast<int>(std::lround(frac * 100.0));
    PostUiProgress(ctx->app, frac,
                   "Downloading model... " + std::to_string(pct) + "% | ETA " +
                       FormatEtaSeconds(eta));
  }
  return 0;
}

static size_t CurlWrite(void* ptr, size_t size, size_t nmemb, void* stream) {
  FILE* f = static_cast<FILE*>(stream);
  return fwrite(ptr, size, nmemb, f);
}

static void RunDownloadThread(std::unique_ptr<DownloadPayload> job) {
  AppState* app = job->app;
  PostUiProgress(app, 0.0, "Preparing download...");

  try {
    const std::filesystem::path out_path(job->out_path);
    const auto parent = out_path.parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent);
    }
  } catch (const std::exception& e) {
    FinishJob(app, std::string("cannot create model directory: ") + e.what(), "");
    return;
  }

  CURL* curl = curl_easy_init();
  if (!curl) {
    FinishJob(app, "curl init failed", "");
    return;
  }

  FILE* fp = fopen(job->out_path.c_str(), "wb");
  if (!fp) {
    curl_easy_cleanup(curl);
    FinishJob(app, "cannot open output model file", "");
    return;
  }

  CurlProgressCtx ctx{app, std::chrono::steady_clock::now()};
  curl_easy_setopt(curl, CURLOPT_URL, job->url.c_str());
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CurlWrite);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
  curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
  curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, CurlXferInfo);
  curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &ctx);

  const CURLcode rc = curl_easy_perform(curl);
  fclose(fp);

  if (rc != CURLE_OK) {
    g_remove(job->out_path.c_str());
    std::string msg = std::string("download failed: ") + curl_easy_strerror(rc);
    curl_easy_cleanup(curl);
    FinishJob(app, msg, "");
    return;
  }

  long status = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
  curl_easy_cleanup(curl);
  if (status < 200 || status >= 300) {
    g_remove(job->out_path.c_str());
    FinishJob(app, "download failed: HTTP " + std::to_string(status), "");
    return;
  }

  auto* payload = new UiProgressPayload{app, 0.0, "Model downloaded.", job->out_path};
  g_idle_add(
      +[](gpointer data) -> gboolean {
        std::unique_ptr<UiProgressPayload> p(static_cast<UiProgressPayload*>(data));
        gtk_editable_set_text(GTK_EDITABLE(p->app->entry_model), p->path.c_str());
        SetBusy(p->app, false);
        SetStatus(p->app, p->text.c_str());
        return G_SOURCE_REMOVE;
      },
      payload);
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
  AutoTuneParamsForImage(st, pin);
  NormalizeTileParams(st, pin);
  job->tile_lr = gtk_spin_button_get_value_as_int(st->spin_tile);
  job->overlap_lr = gtk_spin_button_get_value_as_int(st->spin_overlap);

  st->last_input_path = pin;
  gtk_editable_set_text(GTK_EDITABLE(st->entry_out), job->path_out.c_str());
  UpdatePreview(st->picture_in, st->last_input_path,
                gtk_spin_button_get_value(GTK_SPIN_BUTTON(st->spin_zoom_in)));
  SetBusy(st, true);
  SetStatus(st, "Upscaling...");
  std::thread(RunUpscaleThread, std::move(job)).detach();
}

static void OnDownloadClicked(GtkButton*, gpointer user_data) {
  auto* st = static_cast<AppState*>(user_data);
  if (st->busy.exchange(true)) return;

  const char* url = gtk_editable_get_text(GTK_EDITABLE(st->entry_model_url));
  const char* out_model = gtk_editable_get_text(GTK_EDITABLE(st->entry_model));
  if (!url || !*url || !out_model || !*out_model) {
    st->busy = false;
    SetStatus(st, "Fill model URL and model path.");
    return;
  }

  auto job = std::make_unique<DownloadPayload>();
  job->app = st;
  job->url = url;
  job->out_path = out_model;

  SetBusy(st, true);
  SetStatus(st, "Downloading model...");
  std::thread(RunDownloadThread, std::move(job)).detach();
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
            AutoTuneParamsForImage(s, s->last_input_path);
            UpdatePreview(s->picture_in, s->last_input_path,
                          gtk_spin_button_get_value(GTK_SPIN_BUTTON(s->spin_zoom_in)));
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

static void AutoTuneParamsForImage(AppState* st, const std::string& image_path) {
  int w = 0;
  int h = 0;
  if (!gdk_pixbuf_get_file_info(image_path.c_str(), &w, &h) || w <= 0 || h <= 0) {
    return;
  }
  // Match RealESRGANExample01.py auto-tile behavior:
  // if max_side > 2048: tile=1024, elif > 1024: tile=512, else tile=0.
  const int max_side = std::max(w, h);
  int tile = 0;
  if (max_side > 2048) {
    tile = 1024;
  } else if (max_side > 1024) {
    tile = 512;
  }
  // Example uses tile_pad=10, pre_pad=0. Our closest equivalent is overlap.
  int overlap = 10;

  gtk_spin_button_set_value(st->spin_tile, tile);
  gtk_spin_button_set_value(st->spin_overlap, overlap);
}

static void NormalizeTileParams(AppState* st, const std::string& image_path) {
  int w = 0;
  int h = 0;
  if (!gdk_pixbuf_get_file_info(image_path.c_str(), &w, &h) || w <= 0 || h <= 0) {
    return;
  }
  int tile = gtk_spin_button_get_value_as_int(st->spin_tile);
  int overlap = gtk_spin_button_get_value_as_int(st->spin_overlap);
  if (tile <= 0) {
    return;
  }
  tile = std::max(1, std::min(tile, std::min(w, h)));
  overlap = std::max(0, std::min(overlap, tile / 2 - 1));
  gtk_spin_button_set_value(st->spin_tile, tile);
  gtk_spin_button_set_value(st->spin_overlap, overlap);
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
  gtk_box_append(GTK_BOX(root), BuildPathRow("Output", &st->entry_out, "Browse", G_CALLBACK(OnBrowseOutput), st));
  gtk_box_append(GTK_BOX(root), BuildPathRow("Model", &st->entry_model, "Browse", G_CALLBACK(OnBrowseModel), st));
  gtk_editable_set_text(GTK_EDITABLE(st->entry_model), "model/realesrgan-x4.onnx");

  st->entry_model_url = GTK_ENTRY(gtk_entry_new());
  gtk_editable_set_text(
      GTK_EDITABLE(st->entry_model_url),
      "https://huggingface.co/AXERA-TECH/Real-ESRGAN/resolve/main/onnx/realesrgan-x4.onnx");
  GtkWidget* model_url_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
  gtk_box_append(GTK_BOX(model_url_row), gtk_label_new("Model URL"));
  gtk_widget_set_hexpand(GTK_WIDGET(st->entry_model_url), TRUE);
  gtk_box_append(GTK_BOX(model_url_row), GTK_WIDGET(st->entry_model_url));
  st->btn_download = GTK_BUTTON(gtk_button_new_with_label("Download Model"));
  g_signal_connect(st->btn_download, "clicked", G_CALLBACK(OnDownloadClicked), st);
  gtk_box_append(GTK_BOX(model_url_row), GTK_WIDGET(st->btn_download));
  gtk_box_append(GTK_BOX(root), model_url_row);

  GtkWidget* setting_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
  st->spin_tile = GTK_SPIN_BUTTON(gtk_spin_button_new_with_range(0, 4096, 16));
  gtk_spin_button_set_value(st->spin_tile, 256);
  st->spin_overlap = GTK_SPIN_BUTTON(gtk_spin_button_new_with_range(0, 256, 4));
  gtk_spin_button_set_value(st->spin_overlap, 16);
  st->spin_zoom_in = GTK_SPIN_BUTTON(gtk_spin_button_new_with_range(0.1, 8.0, 0.1));
  gtk_spin_button_set_value(st->spin_zoom_in, 1.0);
  st->spin_zoom_out = GTK_SPIN_BUTTON(gtk_spin_button_new_with_range(0.1, 8.0, 0.1));
  gtk_spin_button_set_value(st->spin_zoom_out, 1.0);
  g_signal_connect(st->spin_zoom_in, "value-changed", G_CALLBACK(OnInputZoomChanged), st);
  g_signal_connect(st->spin_zoom_out, "value-changed", G_CALLBACK(OnOutputZoomChanged), st);
  gtk_box_append(GTK_BOX(setting_row), gtk_label_new("Tile"));
  gtk_box_append(GTK_BOX(setting_row), GTK_WIDGET(st->spin_tile));
  gtk_box_append(GTK_BOX(setting_row), gtk_label_new("Overlap"));
  gtk_box_append(GTK_BOX(setting_row), GTK_WIDGET(st->spin_overlap));
  gtk_box_append(GTK_BOX(setting_row), gtk_label_new("Input Zoom"));
  gtk_box_append(GTK_BOX(setting_row), GTK_WIDGET(st->spin_zoom_in));
  gtk_box_append(GTK_BOX(setting_row), gtk_label_new("Output Zoom"));
  gtk_box_append(GTK_BOX(setting_row), GTK_WIDGET(st->spin_zoom_out));
  st->btn_run = GTK_BUTTON(gtk_button_new_with_label("Upscale"));
  g_signal_connect(st->btn_run, "clicked", G_CALLBACK(OnRunClicked), st);
  gtk_box_append(GTK_BOX(setting_row), GTK_WIDGET(st->btn_run));
  gtk_box_append(GTK_BOX(root), setting_row);

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
  GtkEventController* ctrl_out =
      gtk_event_controller_scroll_new(GTK_EVENT_CONTROLLER_SCROLL_VERTICAL);
  g_signal_connect(ctrl_out, "scroll", G_CALLBACK(OnOutputPreviewScroll), st);
  gtk_widget_add_controller(GTK_WIDGET(st->picture_out), ctrl_out);

  GtkWidget* in_scroll = gtk_scrolled_window_new();
  gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(in_scroll), GTK_POLICY_AUTOMATIC,
                                 GTK_POLICY_AUTOMATIC);
  gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(in_scroll), GTK_WIDGET(st->picture_in));
  GtkWidget* in_frame = gtk_frame_new("Input Preview");
  gtk_frame_set_child(GTK_FRAME(in_frame), in_scroll);

  GtkWidget* out_scroll = gtk_scrolled_window_new();
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
  g_signal_connect(
      win, "destroy", G_CALLBACK(+[](GtkWidget*, gpointer p) { delete static_cast<AppState*>(p); }), st);
  gtk_window_present(st->window);
}

int main(int argc, char** argv) {
  curl_global_init(CURL_GLOBAL_DEFAULT);
  GtkApplication* app = gtk_application_new("com.example.ImageUpscalerGTK", G_APPLICATION_DEFAULT_FLAGS);
  g_signal_connect(app, "activate", G_CALLBACK(OnActivate), nullptr);
  int status = g_application_run(G_APPLICATION(app), argc, argv);
  g_object_unref(app);
  curl_global_cleanup();
  return status;
}
