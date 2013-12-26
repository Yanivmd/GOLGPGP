CPP_SRCS := \
FieldReader.cpp \
main.cpp 

CU_SRCS := \
kernels.cu 

CU_DEPS := \
kernels.d 

OBJS := \
FieldReader.o \
kernels.o \
main.o 

CPP_DEPS := \
FieldReader.d \
main.d 

RM := rm -rf

EXENAME := hw2

# All Target
all: GoL

# Tool invocations
hw2: GoL

GoL: $(OBJS)
	/usr/local/cuda-5.5/bin/nvcc --cudart static -link -o  $(EXENAME) $(OBJS) $(USER_OBJS) $(LIBS)

# Other Targets
clean:
	-$(RM) $(CU_DEPS)$(OBJS)$(C++_DEPS)$(CPP_DEPS) $(EXENAME)
	-@echo ' '

%.o: %.cpp
	/usr/local/cuda-5.5/bin/nvcc -lineinfo -O3 -gencode arch=compute_30,code=sm_30 -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-5.5/bin/nvcc -lineinfo -O3 --compile  -x c++ -o  "$@" "$<"

%.o: %.cu
	/usr/local/cuda-5.5/bin/nvcc -lineinfo -O3 -gencode arch=compute_30,code=sm_30 -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-5.5/bin/nvcc --compile -O3 -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30 -lineinfo  -x cu -o  "$@" "$<"

