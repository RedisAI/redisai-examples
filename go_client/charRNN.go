package main

import "github.com/go-redis/redis"

import (
	"fmt"
	"bytes"
	"encoding/binary"
	"io/ioutil"
	"log"
)

const (
	ascii = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!_#$%&_'()*,+.-/:;<=>?@[]^_`{|}~ \t\n\r\x0b\x0c"
)

func main() {
	client := redis.NewClient(&redis.Options{
			Addr:     "localhost:6379",
			Password: "",
			DB:       0,
		})
	hidden_size := 300
	n_layers := 2
	batch_size := 1
	
	model_path := "../models/CharRNN/CharRNN_pipeline.pt"
	model, err := ioutil.ReadFile(model_path)
	if err != nil {
		log.Fatal(err)
	}
	
	arraylen := hidden_size * batch_size * n_layers
	hidden := new(bytes.Buffer)
	binary.Write(hidden, binary.BigEndian, make([]float32, arraylen))  // Array is zero valued by default
	
	client.Do("AI.MODELSET", "charRnn", "TORCH", "CPU", model)
	client.Do("AI.TENSORSET", "hidden", "FLOAT", n_layers, batch_size, hidden_size, "BLOB", hidden.Bytes())
	client.Do("AI.TENSORSET", "prime", "INT64", 1, "VALUES", 6)
	client.Do("AI.MODELRUN", "charRnn", "INPUTS", "prime", "hidden", "OUTPUTS", "out")
	v, err := client.Do("AI.TENSORGET", "out", "VALUES").Result()
	
	val := v.([]interface{})[2].([]interface {})
	var outstrbuffer bytes.Buffer
	for _, elem := range val {
		outstrbuffer.WriteString(string(ascii[elem.(int64)]))
    }
    fmt.Println(outstrbuffer.String())
}