package util

import (
	"math"
	"strconv"
)

type info struct {
	featureIdx int
	label      int
	value      float64
}

func getGaussianMean(labelClass int, featureIdx int, trainingData [][]string, gaussianChan chan info) {

	numLabelExamples := 0.0
	sum := 0.0
	for _, trainingExample := range trainingData {
		labelIdx := len(trainingExample) - 1
		label, _ := strconv.Atoi(trainingExample[labelIdx])
		if label != labelClass {
			continue
		}
		numLabelExamples = numLabelExamples + 1.0
		featureVal, _ := strconv.ParseFloat(trainingExample[featureIdx], 64)
		sum = sum + featureVal
	}
	gaussianChan <- info{featureIdx, labelClass, (sum / numLabelExamples)}
}

func getGaussianVariance(labelClass int, featureIdx int, trainingData [][]string, mean float64, gaussianChan chan info) {

	numLabelExamples := 0.0
	sum := 0.0
	for _, trainingExample := range trainingData {
		labelIdx := len(trainingExample) - 1
		label, _ := strconv.Atoi(trainingExample[labelIdx])
		if label != labelClass {
			continue
		}
		numLabelExamples = numLabelExamples + 1.0
		featureVal, _ := strconv.ParseFloat(trainingExample[featureIdx], 64)
		sum = sum + math.Pow((featureVal-mean), 2)
	}
	gaussianChan <- info{featureIdx, labelClass, (sum / numLabelExamples)}
}

func GetLabelMLEs(trainingData [][]string) [2]float64 {
	var labelMLEs [2]float64
	for _, trainingExample := range trainingData {
		labelIdx := len(trainingExample) - 1
		label, _ := strconv.Atoi(trainingExample[labelIdx])
		if label == 0 {
			labelMLEs[0] = labelMLEs[0] + 1.0
		} else {
			labelMLEs[1] = labelMLEs[1] + 1.0
		}
	}
	labelMLEs[0] = labelMLEs[0] / float64(len(trainingData))
	labelMLEs[1] = labelMLEs[1] / float64(len(trainingData))
	return labelMLEs
}

func GetGaussianMeans(numFeatures int, trainingData [][]string) [2][]float64 {

	var gaussianMeans [2][]float64
	gaussianMeans[0] = make([]float64, numFeatures)
	gaussianMeans[1] = make([]float64, numFeatures)

	gaussianChan := make(chan info)
	for i := 0; i < numFeatures; i++ {
		go getGaussianMean(0, i, trainingData, gaussianChan)
		go getGaussianMean(1, i, trainingData, gaussianChan)
	}
	for i := 0; i < 2*numFeatures; i++ {
		info := <-gaussianChan
		gaussianMeans[info.label][info.featureIdx] = info.value
	}
	return gaussianMeans
}

func GetGaussianVariances(numFeatures int, trainingData [][]string, gaussianMeans [2][]float64) [2][]float64 {

	var gaussianVariances [2][]float64
	gaussianVariances[0] = make([]float64, numFeatures)
	gaussianVariances[1] = make([]float64, numFeatures)

	gaussianChan := make(chan info)
	for i := 0; i < numFeatures; i++ {
		go getGaussianVariance(0, i, trainingData, gaussianMeans[0][i], gaussianChan)
		go getGaussianVariance(1, i, trainingData, gaussianMeans[1][i], gaussianChan)
	}
	for i := 0; i < 2*numFeatures; i++ {
		info := <-gaussianChan
		gaussianVariances[info.label][info.featureIdx] = info.value
	}
	return gaussianVariances
}

func getGaussianValue(x float64, mean float64, variance float64) float64 {
	expo := -math.Pow(x-mean, 2) / (2.0 * variance)
	denom := math.Sqrt(2.0 * math.Pi * variance)
	return math.Exp(expo) / denom
}

func GetNBLabel(X []string, gaussianMeans [2][]float64, gaussianVariances [2][]float64, labelMLEs [2]float64) int {
	val0 := math.Log(labelMLEs[0])
	val1 := math.Log(labelMLEs[1])
	for featureIdx, sval := range X {
		featureVal, _ := strconv.ParseFloat(sval, 64)
		gaussian0Val := getGaussianValue(featureVal, gaussianMeans[0][featureIdx], gaussianVariances[0][featureIdx])
		val0 = val0 + math.Log(gaussian0Val)
		gaussian1Val := getGaussianValue(featureVal, gaussianMeans[1][featureIdx], gaussianVariances[1][featureIdx])
		val1 = val1 + math.Log(gaussian1Val)
	}
	if val0 > val1 {
		return 0
	}
	return 1
}
