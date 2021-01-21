# Test-ML

	This repo allows you to benchmark FPS for an EfficientDet-D0 model.

## Installation

	./install.sh

## Test Model (C++)

	g++ run.cpp utils.cpp /tmp/cppflow/src/Model.cpp /tmp/cppflow/src/Tensor.cpp -o output -ltensorflow `pkg-config opencv --cflags --libs` && ./output

## Test Model (Python 3)

	python3 run.py