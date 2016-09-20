all: test

test: main.o ImageLoader.o MatrixMultiply.o CostFunctions.o GradientDescent.o
	nvcc -lcublas -o mm main.o ImageLoader.o MatrixMultiply.o CostFunctions.o GradientDescent.o -Xcompiler -fopenmp -std=c++11 -O3

main.o: main.cpp
	nvcc -lcublas -c main.cpp -Xcompiler -fopenmp -std=c++11 -O3

ImageLoader.o: ImageLoader.cpp
	nvcc -lcublas -c ImageLoader.cpp -std=c++11 -O3

MatrixMultiply.o: MatrixMultiply.cpp
	nvcc -lcublas -c MatrixMultiply.cpp -std=c++11 -O3

CostFunctions.o: CostFunctions.cpp
	nvcc -lcublas -c CostFunctions.cpp -Xcompiler -fopenmp -std=c++11 -O3

GradientDescent.o: GradientDescent.cpp
	nvcc -lcublas -c GradientDescent.cpp -Xcompiler -fopenmp -std=c++11 -O3

clean:
	rm -f mm *.o
