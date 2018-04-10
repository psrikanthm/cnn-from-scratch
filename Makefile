# declare the variable
CC=g++

#flags
CFLAGS=-c -Wall -std=c++14 -stdlib=libc++ -Wc++11-extensions -g
LDFLAGS=`pkg-config --cflags --libs opencv boost`

# source files - divided into base files and w.r.t. each of target
SRC_FILES = \
	Matrix.cpp \
	linAlgebra.cpp \
	Util.cpp \

CNN_SRC_FILES = \
	test_cnn.cpp \
	cnn.cpp \

FNN_SRC_FILES = \
	test_fnn.cpp \
	fnn.cpp \

TESTS_SRC_FILES = \
	test_np.cpp \

# object files w.r.t. each of target
OBJ_LIST = $(patsubst %.cpp,%.o,$(SRC_FILES))
CNN_OBJ_LIST = $(patsubst %.cpp,%.o,$(CNN_SRC_FILES))
FNN_OBJ_LIST = $(patsubst %.cpp,%.o,$(FNN_SRC_FILES))
TESTS_OBJ_LIST = $(patsubst %.cpp,%.o,$(TESTS_SRC_FILES))

all: cnn fnn tests

# make each of the target
cnn: $(CNN_OBJ_LIST) $(OBJ_LIST)
	$(CC) $(LDFLAGS) -o cnn $(CNN_OBJ_LIST) $(OBJ_LIST) 

fnn: $(FNN_OBJ_LIST) $(OBJ_LIST)
	$(CC) $(LDFLAGS) -o fnn $(FNN_OBJ_LIST) $(OBJ_LIST)

tests: $(TESTS_OBJ_LIST) $(OBJ_LIST)
	$(CC) $(LDFLAGS) -o tests $(TESTS_OBJ_LIST) $(OBJ_LIST)

# compile the source files for each of object list
$(OBJ_LIST): %.o: %.cpp
	$(CC) $(CFLAGS) $<

$(CNN_OBJ_LIST): %.o: %.cpp
	$(CC) $(CFLAGS) $< 

$(FNN_OBJ_LIST): %.o: %.cpp
	$(CC) $(CFLAGS) $< 

$(TESTS_OBJ_LIST): %.o: %.cpp
	$(CC) $(CFLAGS) $< 
# remove object files
clean:
	rm -f *.o cnn fnn tests

# for debugging purpose
print-%  : ; @echo $* = $($*)
