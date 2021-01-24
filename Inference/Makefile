CC=g++
MKDIR=mkdir -p
RM=rm -rf
QCNNSRC_DIR=quantized-cnn/src
OBJ_DIR=obj
BIN_DIR=bin
OPENVML_DIR=$(HOME)/Codes/OpenVML
SRCS=$(wildcard $(QCNNSRC_DIR)/*.cc)
OBJS=$(SRCS:$(QCNNSRC_DIR)/%.cc=$(OBJ_DIR)/%.o) $(OBJ_DIR)/mergedCaffeEva.o
#CPPFLAGS=-I/usr/include/atlas -I$(OPENVML_DIR)/include
CPPFLAGS=-I/usr/include -I$(OPENVML_DIR)/include
CFLAGS=-Wall -std=c++11 -O2 -ggdb 
#LDFLAGS=-L/usr/lib/atlas -L$(OPENVML_DIR)/build/lib
LDFLAGS=-L/usr/lib -L$(OPENVML_DIR)/build/lib
#LDLIBS=-lcblas -latlas -lopenvml
LDLIBS=-lopenblas -lopenvml
#DFLAGS=-D ENABLE_ATLAS -D ENABLE_OPENVML
DFLAGS=-D ENABLE_OPENBLAS -D ENABLE_OPENVML
TARGET=$(BIN_DIR)/MergedCNN

.PHONY: all run clean

all: $(BIN_DIR) $(OBJ_DIR) $(TARGET)

$(BIN_DIR):
	$(MKDIR) $(BIN_DIR)
$(OBJ_DIR):
	$(MKDIR) $(OBJ_DIR)
$(TARGET): $(OBJS) main.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@
$(OBJ_DIR)/%.o: $(QCNNSRC_DIR)/%.cc 
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DFLAGS) -c $< -o $@
$(OBJ_DIR)/mergedCaffeEva.o: merged-cnn/mergedCaffeEva.cpp 
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DFLAGS) -c $< -o $@

run:
	$(TARGET)

clean:
	$(RM) $(BIN_DIR) $(OBJ_DIR) $(OBJS) $(TARGET)
