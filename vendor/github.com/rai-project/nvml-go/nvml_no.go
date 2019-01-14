// +build !cgo nogpu

package nvml

import (
	"C"
	"fmt"
)
import "errors"

type ComputeMode uint
type Feature uint
type ECCBitType uint
type ECCCounterType uint
type ClockType uint
type DriverModel uint
type PState uint
type InformObject uint
type Result uint

func (c Feature) String() string {
	return "[NVML Disabled]"
}

func (c ComputeMode) String() string {
	return fmt.Sprint("[NVML Disabled]")
}

func (e ECCBitType) String() string {
	return "[NVML Disabled]"
}

func (e ECCCounterType) String() string {
	return "[NVML Disabled]"
}

func (c ClockType) String() string {
	return "[NVML Disabled]"
}

func (d DriverModel) String() string {
	return "[NVML Disabled]"
}

func (p PState) String() string {
	return "[NVML Disabled]"
}

func (i InformObject) String() string {
	return "[NVML Disabled]"
}

func (r Result) String() string {
	return "[NVML Disabled]"
}

func (r Result) Error() string {
	return r.String()
}

func (r Result) SuccessQ() bool {
	return false
}

func NewResult() error {
	return Result(0)
}

func Init() error {
	return NewResult()
}

func Shutdown() error {
	return NewResult()
}

func ErrorString(r Result) string {
	return "[NVML Disabled]"
}

func DeviceCount() (int, error) {
	return 0, errors.New("[NVML Disabled]")
}

type DeviceHandle struct {
}

func DeviceGetHandleByIndex(idx int) (DeviceHandle, error) {
	return DeviceHandle{}, errors.New("[NVML Disabled]")
}

//compute mode

func DeviceComputeMode(dh DeviceHandle) (ComputeMode, error) {
	return ComputeMode(0), errors.New("[NVML Disabled]")
}

func DeviceName(dh DeviceHandle) (string, error) {
	return "", errors.New("[NVML Disabled]")
}

type Utilization struct {
	GPU    int
	Memory int
}

func GetUtilization(dh DeviceHandle) (utilization Utilization, err error) {
	err = errors.New("[NVML Disabled]")
	return
}

type MemoryInformation struct {
	Used  uint64 `json:"used"`
	Free  uint64 `json:"free"`
	Total uint64 `json:"total"`
}

func DeviceMemoryInformation(dh DeviceHandle) (MemoryInformation, error) {
	return MemoryInformation{}, errors.New("[NVML Disabled]")
}

type PCIInformation struct {
	BusId       string `json:"bus_id"`
	Domain      uint   `json:"domain"`
	Bus         uint   `json:"bus"`
	Device      uint   `json:"device"`
	DeviceId    uint   `json:"device_id"`
	SubSystemId uint   `json:"subsystem_id"`
}

func DevicePCIInformation(dh DeviceHandle) (PCIInformation, error) {
	return PCIInformation{}, errors.New("[NVML Disabled]")
}

func DeviceTemperature(dh DeviceHandle) (uint, error) {
	return uint(0), errors.New("[NVML Disabled]")
}

func DevicePerformanceState(dh DeviceHandle) (PState, error) {
	return PState(0), errors.New("[NVML Disabled]")
}

func DeviceFanSpeed(dh DeviceHandle) (uint, error) {
	return uint(0), errors.New("[NVML Disabled]")
}
