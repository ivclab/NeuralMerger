CC=g++
MKDIR=mkdir -p
RM=rm -rf
QCNNSRC_DIR=quantized-cnn/src
OBJ_DIR=obj
BIN_DIR=bin
SRCS=$(wildcard $(QCNNSRC_DIR)/*.cc) merged-cnn/main.cpp
OBJS=$(SRCS:$(QCNNSRC_DIR)/%.cc=$(OBJ_DIR)/%.o)
CPPFLAGS=-I/usr/include/atlas -I/home/yiming/Codes/OpenVML/include
CFLAGS=-Wall -std=c++11 -O2 -ggdb
LDFLAGS=-L/usr/lib/atlas-base -L/home/yiming/Codes/OpenVML/build/lib
LDLIBS=-lcblas -latlas -lopenvml
DFLAGS=-D ENABLE_ATLAS -D ENABLE_OPENVML
TARGET=$(BIN_DIR)/MergedCNN

.PHONY: all run clean

all: $(BIN_DIR) $(OBJ_DIR) $(TARGET)

$(BIN_DIR):
	$(MKDIR) $(BIN_DIR)
$(OBJ_DIR):
	$(MKDIR) $(OBJ_DIR)
$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@
$(OBJ_DIR)/%.o: $(QCNNSRC_DIR)/%.cc
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DFLAGS) -c $< -o $@

run:
	$(TARGET)

clean:
	$(RM) $(BIN_DIR) $(OBJ_DIR) $(OBJS) $(TARGET)