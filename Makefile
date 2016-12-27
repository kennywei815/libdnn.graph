CC=gcc
CXX=g++
CFLAGS=
NVCC=nvcc -w #-Xcompiler "-Wall"

BOTON_UTIL_ROOT=tools/utility/
CUMATRIX_ROOT=tools/libcumatrix/

INCLUDE= -I include/ \
	 -I $(BOTON_UTIL_ROOT)/include/ \
	 -I $(CUMATRIX_ROOT)/include \
 	 -I /usr/local/cuda/samples/common/inc/ \
	 -I /usr/local/cuda/include

CPPFLAGS= -std=c++0x $(CFLAGS) $(INCLUDE) -Ofast #-Werror -Wall
NVCCFLAGS= $(CFLAGS) -O3 \
    -gencode=arch=compute_30,code=sm_30 \
    -gencode=arch=compute_32,code=sm_32 \
    -gencode=arch=compute_35,code=sm_35 \
    -gencode=arch=compute_50,code=sm_50 \
    -gencode=arch=compute_50,code=compute_50

    #-gencode=arch=compute_11,code=sm_11 \
    #-gencode=arch=compute_12,code=sm_12 \
    #-gencode=arch=compute_13,code=sm_13 \
    #-gencode=arch=compute_20,code=sm_20 \
    #-gencode=arch=compute_20,code=sm_21 \
    #-gencode=arch=compute_52,code=sm_52 \

SOURCES=cnn-utility.cu\
	cnn.cpp\
	dnn-utility.cu\
	dnn.cpp\
	dnn-graph.cpp\
	utility.cpp\
	rbm.cpp\
	feature-transform.cpp\
	dataset.cpp\
	batch.cpp\
	config.cpp

EXECUTABLES=dnn-train-graph\
	    dnn-predict\
	    dnn-init\
	    cnn-train\
	    dnn-info\
	    dnn-print\
	    data-statistics\
	    dnn-transpose

EXECUTABLES:=$(addprefix bin/, $(EXECUTABLES))

.PHONY: debug all o3 ctags dump_nrv
all: $(EXECUTABLES) ctags

o3: CFLAGS+=-O3
o3: all
debug: CFLAGS+=-g -DDEBUG
debug: all
dump_nrv: NVCC+=-Xcompiler "-fdump-tree-nrv" all
dump_nrv: all

vpath %.h include/
vpath %.cpp src/
vpath %.cu src/

OBJ:=$(addprefix obj/, $(addsuffix .o,$(basename $(SOURCES))))

LIBRARY=-lpbar -lcumatrix
CUDA_LIBRARY=-lcuda -lcudart -lcublas
LIBRARY_PATH=-L$(BOTON_UTIL_ROOT)/lib/ -L$(CUMATRIX_ROOT)/lib -L/usr/local/cuda/lib64

test: test.cpp
	g++ -std=c++0x $(INCLUDE) $(LIBRARY_PATH) -o test test.cpp -lpthread  $(CUDA_LIBRARY)

$(EXECUTABLES): bin/% : obj/%.o $(OBJ)
	$(CXX) -o $@ $(CFLAGS) -std=c++0x $(INCLUDE) $^ $(LIBRARY_PATH) $(LIBRARY) $(CUDA_LIBRARY)

# +==============================+
# +===== Other Phony Target =====+
# +==============================+
obj/%.o: %.cpp
	$(CXX) $(CPPFLAGS) -std=c++0x -o $@ -c $<

obj/%.o: %.cu include/%.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -o $@ -c $<

obj/%.d: %.cpp
	@$(CXX) -MM $(CPPFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,obj/\1.o $@ : ,g' < $@.$$$$ > $@;\
	rm -f $@.$$$$

-include $(addprefix obj/,$(subst .cpp,.d,$(SOURCES)))

.PHONY: ctags clean
ctags:
	@ctags -R --langmap=C:+.cu *
clean:
	rm -rf bin/* obj/*
