package main

import "github.com/go-redis/redis"

import (
    "fmt"
    "image"
    "image/jpeg"
    "image/draw"
    "os"
    "bytes"
    "encoding/binary"
    "io/ioutil"
    "encoding/json"
    "strconv"
)

func init() {
    image.RegisterFormat("jpeg", "jpeg", jpeg.Decode, jpeg.DecodeConfig)
}


func getRGBImage(imgPath string) (*bytes.Buffer, image.Rectangle) {
    var rgbImg []uint8
    imgfile, _ := os.Open(imgPath)
    defer imgfile.Close()
    decodedImg, _, _ := image.Decode(imgfile)
    rect := decodedImg.Bounds()
    rgba := image.NewRGBA(rect)
    draw.Draw(rgba, rect, decodedImg, rect.Min, draw.Src)
    for x:= 0; x < len(rgba.Pix); x += 4 {
        rgbImg = append(rgbImg, rgba.Pix[x], rgba.Pix[x + 1], rgba.Pix[x + 2])
    }
    imgbuf := new(bytes.Buffer)
    binary.Write(imgbuf, binary.BigEndian, rgbImg)
    return imgbuf, rect
}


func main() {
    var classes map[string]string

    client := redis.NewClient(&redis.Options{
            Addr:     "localhost:6379",
            Password: "",
            DB:       0,
        })

    imgPath := "../models/imagenet/data/cat.jpg"
    modelPath := "../models/imagenet/pytorch/resnet50.pt"
    scriptPath := "../models/imagenet/pytorch/data_processing_script.txt"
    jsonPath := "../models/imagenet/data/imagenet_classes.json"


    imgbuf, rect := getRGBImage(imgPath)
    
    model, _ := ioutil.ReadFile(modelPath)
    script, _ := ioutil.ReadFile(scriptPath)
    client.Do("AI.MODELSET", "imagenet_model", "TORCH", "CPU", model)
    client.Do("AI.SCRIPTSET", "imagenet_script", "CPU", script)
    client.Do("AI.TENSORSET", "image", "UINT8", rect.Max.X, rect.Max.Y, 3, "BLOB", imgbuf.Bytes())
    
    client.Do("AI.SCRIPTRUN", "imagenet_script", "pre_process_3ch", "INPUTS", "image", "OUTPUTS", "temp1")
    client.Do("AI.MODELRUN", "imagenet_model", "INPUTS", "temp1", "OUTPUTS", "temp2")
    client.Do("AI.SCRIPTRUN", "imagenet_script", "post_process", "INPUTS", "temp2", "OUTPUTS", "out")
    
    v, _ := client.Do("AI.TENSORGET", "out", "VALUES").Result()
    val := v.([]interface{})[2].([]interface {})[0].(int64)
    byteValue, _ := ioutil.ReadFile(jsonPath)
    json.Unmarshal([]byte(byteValue), &classes)
    fmt.Println(classes[strconv.FormatInt(val, 10)])
}