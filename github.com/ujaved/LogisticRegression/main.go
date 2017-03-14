package main

import (
	"bufio"
	"flag"
	"fmt"
	"github.com/ujaved/LogisticRegression/util"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

var NUM_ITERATIONS = 100
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
	useRegularization := flag.Bool("reg", false, "use regularization")
	flag.Parse()
	file, err := os.Open(*fileStrPtr)
	check(err)
	defer file.Close()

	trainData, testData := getData(file)
	numFeatures := len(trainData[0]) - 1
	weights := make([]float64, numFeatures+1)

	for i := 0; i < NUM_ITERATIONS; i++ {
		weights = util.GetNewWeights(weights, trainData, *useRegularization)
	}
	numErrors := 0
	for _, t := range testData {
		label := util.GetLRLabel(weights, t[:len(t)-1])
		trueLabel, _ := strconv.Atoi(t[len(t)-1])
		if label != trueLabel {
			numErrors = numErrors + 1
		}
	}
	fmt.Println(len(testData), numErrors)
}
