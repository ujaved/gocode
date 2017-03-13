package util

import (
	"log"
	"math"
	"strconv"
        "fmt"
)

var STEP_SIZE float64 = 0.01

func GetLRLabel(weights []float64, X []string) int {
	v := GetLRValue(true, weights, X)
	if v >= 0.5 {
		return 1
	} else {
		return 0
	}
}

func GetLRValue(labelIsOne bool, weights []float64, X []string) float64 {
	if len(weights) != len(X)+1 {
		log.Fatal("the length of the weight vector should be equal to the length of X plus 1")
	}
	weightedSum := weights[0]
	for i, sval := range X {
		val, _ := strconv.Atoi(sval)
		weightedSum += float64(val) * weights[i+1]
	}
	val := 1.0 / (1 + math.Exp(weightedSum))
	if labelIsOne {
		val = 1 - val
	}
	return val
}

func getPredictionErrors(weights []float64, trainingData [][]string) []float64 {

	errors := make([]float64, len(trainingData))
	for i, trainingExample := range trainingData {
		labelIdx := len(trainingExample) - 1
		label, _ := strconv.Atoi(trainingExample[labelIdx])
		LRValue := GetLRValue(true, weights, trainingExample[:labelIdx])
		error := float64(label) - LRValue
		errors[i] = error
	}
	return errors
}

func getNewWeight(idx int, curWeight float64, weightChan chan struct {
	int
	float64
}, predictionErrors []float64, trainingData [][]string) {
	sum := 0.0
	for j, error := range predictionErrors {
		if idx > 0 {
			x, _ := strconv.Atoi(trainingData[j][idx-1])
			error = float64(x) * error
		}
		sum += error
	}
	newWeight := curWeight + STEP_SIZE*sum
	weightChan <- struct {
		int
		float64
	}{idx, newWeight}
}

func GetNewWeights(curWeights []float64, trainingData [][]string) []float64 {

	newWeights := make([]float64, len(curWeights))
	predictionErrors := getPredictionErrors(curWeights, trainingData)
	weightChan := make(chan struct {
		int
		float64
	})
	for i, w := range curWeights {
		go getNewWeight(i, w, weightChan, predictionErrors, trainingData)
	}
	for range newWeights {
		pair := <-weightChan
		newWeights[pair.int] = pair.float64
	}
	return newWeights
}
