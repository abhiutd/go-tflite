#define _GLIBCXX_USE_CXX11_ABI 0

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/optional_debug_tools.h"

#include "predictor.hpp"

#if 0
#define DEBUG_STMT std ::cout << __func__ << "  " << __LINE__ << "\n";
#else
#define DEBUG_STMT
#endif

using namespace tflite;
using std::string;

/* Pair (label, confidence) representing a prediction. */
using Prediction = std::pair<int, float>;

/*
  Predictor class takes in model file (converted into .tflite from the original .pb file
  using tflite_convert CLI tool), batch size and device mode for inference
*/
class Predictor {
  public:
    Predictor(const string &model_file, int batch, int mode);
    void Predict(float* inputData);

    std::unique_ptr<tflite::FlatBufferModel> net_;
    int width_, height_, channels_;
    int batch_;
    int pred_len_;
    int mode_ = 0;
    TfLiteTensor* result_;
};

Predictor::Predictor(const string &model_file, int batch, int mode) {
  /* Load the network. */
  // Tflite uses FlatBufferModel format to store/access model instead of 
	// protobuf unlike tensorflow
  char* model_file_char = const_cast<char*>(model_file.c_str());
	net_ = tflite::FlatBufferModel::BuildFromFile(model_file_char);
  assert(net_ != nullptr);
  mode_ = mode;
  batch_ = batch;
}

void Predictor::Predict(float* inputData) {

	// build interpreter
	// Note: one can have multiple interpreters running the same FlatBuffer
	// model, therefore, we create an interpeter for every call of Predict()
	// rather than one for the Predictor
	// Also, one can add customized operators by rebuilding
	// thge resolver with their own operator definitions
	tflite::ops::builtin::BuiltinOpResolver resolver;
	std::unique_ptr<tflite::Interpreter> interpreter;
	tflite::InterpreterBuilder(*net_, resolver)(&interpreter);
	assert(interpreter != nullptr);

	// set number of threads to 1 for now
	interpreter->SetNumThreads(1);

	// allocate tensor buffers
	if(!(interpreter->AllocateTensors() == kTfLiteOk)) {
		fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);
		exit(1);
	}

	// fill input buffers
	const int input = interpreter->inputs()[0];
	TfLiteTensor* input_tensor = interpreter->tensor(input);
	TfLiteIntArray* input_dims = input_tensor->dims;
	width_ = input_dims->data[1];
	height_ = input_dims->data[2];
	channels_ = input_dims->data[3];

	assert(input_dims->size == 4);
	const int size = batch_ * width_ * height_ * channels_;
	memcpy(input_tensor->data.f, &inputData[0], size);

	const int output = interpreter->outputs()[0];
	result_ = interpreter->tensor(output);

	// run inference
	if(!(interpreter->Invoke() == kTfLiteOk)) {
		fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);
		exit(1);
	}

}

PredictorContext NewTflite(char *model_file, int batch,
                          int mode) {
  try {
    int mode_temp = 0;
    if (mode == 1) {
      mode_temp = 1;
    }
    const auto ctx = new Predictor(model_file, batch,
                                   mode_temp);
    return (void *)ctx;
  } catch (const std::invalid_argument &ex) {
    //LOG(ERROR) << "exception: " << ex.what();
    errno = EINVAL;
    return nullptr;
  }

}

void SetModeTflite(int mode) {
  if(mode == 1) {
		// Do nothing as of now
  }
}

void InitTflite() {}

void PredictTflite(PredictorContext pred, float* inputData) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  predictor->Predict(inputData);
  return;
}

const float*GetPredictionsTflite(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return nullptr;
  }

  return predictor->result_->data.f;
}

void DeleteTflite(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  delete predictor;
}

int GetWidthTflite(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  return predictor->width_;
}

int GetHeightTflite(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  return predictor->height_;
}

int GetChannelsTflite(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  return predictor->channels_;
}

int GetPredLenTflite(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  predictor->pred_len_ = predictor->result_->dims->size;
  return predictor->pred_len_;
}

