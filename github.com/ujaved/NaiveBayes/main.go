package main

import (
	"bufio"
	"flag"
	"fmt"
	"github.com/ujaved/NaiveBayes/util"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

var TRAIN_TEST_SPLIT = 0.9

func check(e error) {
	if e != nil {
		log.Fatal(e)
	}
}

func getData(file *os.File) ([][]string, [][]string) {
	data := [][]string{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Fields(line)
		// remove the first field since that's the ID
		data = append(data, fields[1:])
	}
	// shuffle the data
	randData := make([][]string, len(data))
	perm := rand.Perm(len(data))
	for i, v := range perm {
		randData[i] = data[v]
	}
	testStartIdx := int(math.Floor(float64(len(data)) * TRAIN_TEST_SPLIT))
	return randData[:testStartIdx], randData[testStartIdx:]
}

func main() {
	fileStrPtr := flag.String("file", "", "train file location")
	flag.Parse()
	file, err := os.Open(*fileStrPtr)
	check(err)
	defer file.Close()

	trainData, testData := getData(file)
	numFeatures := len(trainData[0]) - 1

	// gaussian means for the two label classes
	gaussianMeans := util.GetGaussianMeans(numFeatures, trainData)

	// gaussian variances for the two label classes
	gaussianVariances := util.GetGaussianVariances(numFeatures, trainData, gaussianMeans)

	labelMLEs := util.GetLabelMLEs(trainData)

	numErrors := 0
	s := make(chan util.Info)
	for i, t := range testData {
		go util.GetNBLabel(t[:len(t)-1], i, gaussianMeans, gaussianVariances, labelMLEs, s)
	}
	for range testData {
		info := <-s
		trueLabel, _ := strconv.Atoi(testData[info.Idx][numFeatures])
		if info.Label != trueLabel {
			numErrors = numErrors + 1
		}
	}
	fmt.Println(len(testData), numErrors)
}
