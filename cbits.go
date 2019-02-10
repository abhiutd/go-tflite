package tflite

// #include <stdio.h>
// #include <stdlib.h>
// #include "cbits/predictor.hpp"
import "C"
import (
	"fmt" //"github.com/k0kubun/pp"
	"unsafe"
	"image"
	"path/filepath"

	"github.com/anthonynsimon/bild/imgio"
  "github.com/anthonynsimon/bild/transform"
	"github.com/Unknwon/com"
	"github.com/k0kubun/pp"
	"github.com/pkg/errors"
)

// TODO add specific accelerator modes
const (
	CPUMode = 0
	GPUMode = 1
	DSPMode = 2
	VisualCore = 3
)

// struct for keeping hold of predictor
type PredictorData struct {
	ctx	C.PredictorContext
	mode	int
	batch	int
}

// make mode and batch public
func (pd *PredictorData) Inc() {
   pd.mode++
   pd.batch++
}

func NewPredictorData() *PredictorData {
     return &PredictorData{}
}

// Note: for internal use only
// convert go Image to 1-dim array
func cvtImageTo1DArray(src image.Image, mean []float32) ([]float32, error) {

	if src == nil {
    return nil, fmt.Errorf("src image nil")
  }

  b := src.Bounds()
  h := b.Max.Y - b.Min.Y // image height
  w := b.Max.X - b.Min.X // image width

  res := make([]float32, 3*h*w)
  for y := 0; y < h; y++ {
    for x := 0; x < w; x++ {
      r, g, b, _ := src.At(x+b.Min.X, y+b.Min.Y).RGBA()
      res[y*w+x] = float32(b>>8) - mean[0]
      res[w*h+y*w+x] = float32(g>>8) - mean[1]
      res[2*w*h+y*w+x] = float32(r>>8) - mean[2]
    }
  }

  return res, nil
}

// Note: for internal use only
// preprocess (read/convert) image
func preprocessImage(imageFile string, batchSize int) ([]float32, error) {

	// read image file
	imgDir, _ := filepath.Abs("../_fixtures")
	imagePath := filepath.Join(imgDir, imageFile)
	img, err := imgio.Open(imagePath)
	if err != nil {
		panic(err)
	}
	// convert go image to 1D array (float32 by default)
	var input []float32
	for ii := 0; ii < batchSize; ii++ {
    resized := transform.Resize(img, 227, 227, transform.Linear)
    res, err := cvtImageTo1DArray(resized, []float32{123, 117, 104})
    if err != nil {
      panic(err)
    }
    input = append(input, res...)
	}

	return input, nil
}

func New(model string, mode, batch int) (*PredictorData, error) {

	// fetch model file
	modelFile := model
	if !com.IsFile(modelFile) {
		return nil, errors.Errorf("file %s not found", modelFile)
	}

	// set device for acceleration
	switch mode {
	case 0:
		SetUseCPU()
	case 1:
		SetUseGPU()
	case 2:
		SetUseDSP()
	case 3:
		SetUseVisualCore()
	default:
		SetUseCPU()
	}

	return &PredictorData{
		ctx: C.NewTflite(
			C.CString(modelFile),
			C.int(batch),
			C.int(mode),
		),
		mode:	mode,
		batch:	batch,
	}, nil
}

func SetUseCPU() {
	C.SetModeTflite(C.int(CPUMode))
}

func SetUseGPU() {
	C.SetModeTflite(C.int(GPUMode))
}

func SetUseDSP() {
   C.SetModeTflite(C.int(DSPMode))
 }

func SetUseVisualCore() {
   C.SetModeTflite(C.int(VisualCoreMode))
}

func init() {
	C.InitTflite()
}

func Predict(p *PredictorData, image string) error {

	// check for null imagedata
	if len(image) == 0 {
		return fmt.Errorf("input image filepath is empty")
	}

	batchSize := p.batch
	width := C.GetWidthTflite(p.ctx)
	height := C.GetHeightTflite(p.ctx)
	channels := C.GetChannelsTflite(p.ctx)
	shapeLen := int(width * height * channels)

	// preprocess input image
	data, err := preprocessImage(image, batchSize)
	if err != nil {
		panic(err)
	}

	// pad input image if needed
	dataLen := len(data)

	inputCount := dataLen / shapeLen
	if batchSize > inputCount {
		padding := make([]float32, (batchSize-inputCount)*shapeLen)
		data = append(data, padding...)
	}

	ptr := (*C.float)(unsafe.Pointer(&data[0]))

	C.PredictTflite(p.ctx, ptr)

	return nil
}

// TODO return predicted class (as in do the postprocessing here itself)
func ReadPredictionOutput(p *PredictorData) (float32, error) {

	batchSize := p.batch
	predLen := int(C.GetPredLenTflite(p.ctx))
	length := batchSize * predLen

	cPredictions := C.GetPredictionsTflite(p.ctx)
	if cPredictions == nil {
		return 0, errors.New("empty predictions")
	}

	slice := (*[1 << 15]float32)(unsafe.Pointer(cPredictions))[:length:length]
	pp.Println(slice[:2])

	return slice[3], nil
}

func Close(p *PredictorData) {
	C.DeleteTflite(p.ctx)
}
