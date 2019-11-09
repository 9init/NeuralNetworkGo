package main

import(
    "fmt"
    "math/rand"
    "time"
    neuraln "github.com/9init/NeuralNetworkGo"
)

type objects struct {
	inputs  []float64
	outputs []float64
}

func shuffle(l *[]objects) {
	rand.Seed(time.Now().UnixNano())
	time.Sleep(1)
	rand.Shuffle(len(*l), func(i, j int) { (*l)[i], (*l)[j] = (*l)[j], (*l)[i] })
}

func main() {
	var nn = new(neuraln.NeuralN)
	nn.Create(2, 50, 1)
	list := []objects{
		objects{[]float64{1,0}, []float64{1}},
		objects{[]float64{0,1}, []float64{1}},
		objects{[]float64{1,1}, []float64{0}},
		objects{[]float64{0,0}, []float64{0}},
	}
	
	//lets train our code :)
	//you have to shuffle your list
	for i := 0; i < 1000; i++ {
		for _, v := range list {
			shuffle(&list)
			nn.Train(v.inputs, v.outputs)
		}
	}

	test:=[][]float64{{0,1},{0,0},{1,0},{1,1}}
	for _, f := range test{
		m := nn.FeedForword(f)
		for _, v := range m.Matrix {
			for _,v2:=range v{
				fmt.Println(v2)
			}
		}
		fmt.Print("\n")
	}

}
