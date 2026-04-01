SHELL := /bin/bash

TARGET := image_upscaler_gtk
SRC_DIR := src
BUILD_DIR := build
SRCS := $(SRC_DIR)/main.cpp $(SRC_DIR)/onnx_sr.cpp
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

CXX ?= c++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
CPPFLAGS += $(shell pkg-config --cflags gtk4 2>/dev/null)
LDFLAGS +=
LDLIBS += $(shell pkg-config --libs gtk4 2>/dev/null) -lcurl

# ONNX Runtime location: prefer ONNXRUNTIME_ROOT, fallback to common prefixes.
ifeq ($(strip $(ONNXRUNTIME_ROOT)),)
  ifneq ("$(wildcard /opt/homebrew/opt/onnxruntime/include/onnxruntime_cxx_api.h)","")
    ONNXRUNTIME_ROOT := /opt/homebrew/opt/onnxruntime
  else ifneq ("$(wildcard /usr/local/opt/onnxruntime/include/onnxruntime_cxx_api.h)","")
    ONNXRUNTIME_ROOT := /usr/local/opt/onnxruntime
  else ifneq ("$(wildcard /usr/include/onnxruntime/onnxruntime_cxx_api.h)","")
    ONNXRUNTIME_ROOT := /usr
  endif
endif

ifdef ONNXRUNTIME_ROOT
  ifeq ($(ONNXRUNTIME_ROOT),/usr)
    CPPFLAGS += -I/usr/include/onnxruntime
    LDFLAGS += -L/usr/lib/x86_64-linux-gnu
  else
    CPPFLAGS += -I$(ONNXRUNTIME_ROOT)/include
    LDFLAGS += -L$(ONNXRUNTIME_ROOT)/lib
  endif
endif
LDLIBS += -lonnxruntime

ifeq ($(V),1)
  Q :=
else
  Q := @
endif

.PHONY: all help check-dirs check-deps show build run clean

all: build

help:
	@echo "Targets:"
	@echo "  make build      - build executable ($(TARGET))"
	@echo "  make run        - run executable"
	@echo "  make clean      - remove build outputs"
	@echo "  make show       - show resolved flags and files"
	@echo ""
	@echo "Options:"
	@echo "  V=1             - verbose compile/link commands"
	@echo "  ONNXRUNTIME_ROOT=/path/to/onnxruntime"

check-dirs:
	$(Q)mkdir -p "$(BUILD_DIR)"

check-deps:
	$(Q)pkg-config --exists gtk4 || (echo "Error: gtk4 not found via pkg-config"; exit 1)
	$(Q)command -v $(CXX) >/dev/null || (echo "Error: compiler not found: $(CXX)"; exit 1)

show:
	@echo "CXX=$(CXX)"
	@echo "CXXFLAGS=$(CXXFLAGS)"
	@echo "CPPFLAGS=$(CPPFLAGS)"
	@echo "LDFLAGS=$(LDFLAGS)"
	@echo "LDLIBS=$(LDLIBS)"
	@echo "ONNXRUNTIME_ROOT=$(ONNXRUNTIME_ROOT)"
	@echo "SRCS=$(SRCS)"
	@echo "OBJS=$(OBJS)"

build: check-deps check-dirs $(TARGET)
	@echo "Build finished: ./$(TARGET)"

$(TARGET): $(OBJS)
	@echo "[Link] $@"
	$(Q)$(CXX) $(CXXFLAGS) $(OBJS) -o "$@" $(LDFLAGS) $(LDLIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "[CXX] $<"
	$(Q)$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c "$<" -o "$@"

run: build
	./$(TARGET)

clean:
	@echo "Cleaning build artifacts..."
	$(Q)rm -rf "$(BUILD_DIR)" "$(TARGET)"
