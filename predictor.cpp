#define _GLIBCXX_USE_CXX11_ABI 0

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "predictor.hpp"
#include "timer.h"
#include "timer.impl.hpp"

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
  TODO enable accelerator device modes
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
    profile *prof_{nullptr};
    bool profile_enabled_{false};
    int result_;
};

Predictor::Predictor(const string &model_file, int batch, int mode) {
  /* Load the network. */
  // Tflite uses FlatBufferModel format to
  // store/access model instead of protobuf
  // unlike tensorflow
  net_ = tflite::FlatBufferModel::BuildFromFile(model_file);
  assert(net_ != nullptr);
  mode_ = mode;

  // TODO should fetch width and height from model

  width_ = 224;
  height_ = 224;
  channels_ = 3;
  batch_ = batch;

  CHECK(channels_ == 3 || channels_ == 1) << "Input layer should have 1 or 3 channels.";

}

void Predictor::Predict(float* inputData) {

	// build interpreter
	// Note: one can have multiple interpreters running the same FlatBuffer
	// model, therefore, we create an interpeter for every call of Predict()
	// rather than one for the Predictor
	// Also, one can add customized operators by rebuilding
	// thge resolver with their own operator definitions
	tflite::ops::builtin::BuiltinOpResolver resolver;
	InterpreterBuilder builder(net_, resolver);
	std::unique_ptr<Interpreter> interpreter;
	builder(&interpreter);
	assert(interpreter != nullptr);

	// allocate tensor buffers
	if(!(interpreter->AllocateTensors() == kTfLiteOk)) {
		fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);
		exit(1);
	}

	// TODO fill input buffers

	// run inference
	if(!(interpreter->Invoke() == kTfLiteOk)) {
     fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);
     exit(1);
   }

	 // TODO read output buffers into result_
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
    LOG(ERROR) << "exception: " << ex.what();
    errno = EINVAL;
    return nullptr;
  }

}

void SetModeTflite(int mode) {
  if(mode == 1) {
		// TODO set accelerator here
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

  return predictor->result_.data<float>();
}

void DeleteTflite(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (predictor->prof_) {
    predictor->prof_->reset();
    delete predictor->prof_;
    predictor->prof_ = nullptr;
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
  predictor->pred_len_ = predictor->result_.size(1);
  return predictor->pred_len_;
}

