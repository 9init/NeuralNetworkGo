package iprocessing

import (
	"image"
	"image/color"
	"image/jpeg"
	"log"
	"os"
	"sort"

	"golang.org/x/image/draw"
)

var convFilter = [11]float64{
	-1, -1, -1,
	-1, 8, -1,
	-1, -1, -1,
	1, 1,
}

var leftEdge = [11]float64{
	-1, 1, 0,
	-1, 1, 0,
	-1, 1, 0,
	1, 1,
}

var rightEdge = [11]float64{
	0, 1, -1,
	0, 1, -1,
	0, 1, -1,
	1, 1,
}

var topEdge = [11]float64{
	-1, -1, -1,
	1, 1, 1,
	0, 0, 0,
	1, 1,
}

var downEdge = [11]float64{
	0, 0, 0,
	1, 1, 1,
	-1, -1, -1,
	1, 1,
}

var gaussianBlur = [11]float64{
	1, 2, 1,
	2, 4, 2,
	1, 2, 1,
	0.0625, 6,
}

func saveImage(picture *image.Image, name string) {
	f, err := os.Create(name)
	defer f.Close()
	errorHandler(err)
	options := jpeg.Options{Quality: 100}
	jpeg.Encode(f, *picture, &options)

}
func errorHandler(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

//RGB (R)ed (G)reen (B)lue
type RGB struct {
	R float64
	G float64
	B float64
}

func colorScale(num float64) float64 {
	switch true {
	case num > 255:
		num = 255
	case num <= 0:
		num = 0
	}
	return num
}

func greyScale(rgb *RGB) {
	lum := 0.299*rgb.R + 0.587*rgb.G + 0.114*rgb.B
	*rgb = RGB{lum, lum, lum}
}

func fixDim(src *[][]RGB, dHeight, dWidth int) {
	width, height := len((*src)[0]), len(*src)
	if width != dWidth || height != dHeight {
		img := array2Imge(src)
		rect := image.Rect(0, 0, dWidth, dHeight)
		dst := image.NewRGBA(rect)
		draw.BiLinear.Scale(dst, rect, img, img.Bounds(), draw.Over, nil)
		imgARR0 := dst.SubImage(dst.Rect)
		imageArr := image2Array(&imgARR0)
		*src = *imageArr
	}
}

func applykernel(src *[][]RGB, kernel [11]float64, dHeight, dWidth int) {

	for i := float64(0); i < kernel[10]; i++ {

		width, height := len((*src)[0]), len(*src)
		finalWidth := width - 2
		finalHeight := height - 2

		finalImage := make([][]RGB, finalHeight)
		for i := range finalImage {
			finalImage[i] = make([]RGB, finalWidth)
		}

		for y := 1; y < height-1; y++ {
			for x := 1; x < width-1; x++ {
				y1x1, y1x2, y1x3 := &(*src)[y-1][x-1], &(*src)[y-1][x], &(*src)[y-1][x+1]
				y2x1, y2x2, y2x3 := &(*src)[y][x-1], &(*src)[y][x], &(*src)[y][x+1]
				y3x1, y3x2, y3x3 := &(*src)[y+1][x-1], &(*src)[y+1][x], &(*src)[y+1][x+1]

				colR := (y1x1.R * kernel[0]) + (y1x2.R * kernel[1]) + (y1x3.R * kernel[2]) +
					(y2x1.R * kernel[3]) + (y2x2.R * kernel[4]) + (y2x3.R * kernel[5]) +
					(y3x1.R * kernel[6]) + (y3x2.R * kernel[7]) + (y3x3.R * kernel[8])
				colR *= kernel[9]

				colG := (y1x1.G * kernel[0]) + (y1x2.G * kernel[1]) + (y1x3.G * kernel[2]) +
					(y2x1.G * kernel[3]) + (y2x2.G * kernel[4]) + (y2x3.G * kernel[5]) +
					(y3x1.G * kernel[6]) + (y3x2.G * kernel[7]) + (y3x3.G * kernel[8])
				colG *= kernel[9]

				colB := (y1x1.B * kernel[0]) + (y1x2.B * kernel[1]) + (y1x3.B * kernel[2]) +
					(y2x1.B * kernel[3]) + (y2x2.B * kernel[4]) + (y2x3.B * kernel[5]) +
					(y3x1.B * kernel[6]) + (y3x2.B * kernel[7]) + (y3x3.B * kernel[8])
				colB *= kernel[9]

				colRGB := RGB{colorScale(colR), colorScale(colG), colorScale(colB)}
				greyScale(&colRGB)
				finalImage[y-1][x-1] = colRGB
			}
		}
		*src = finalImage
	}
}

func image2Array(src *image.Image) *[][]RGB {
	bounds := (*src).Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	iaa := make([][]RGB, height)
	srcrgba := image.NewRGBA((*src).Bounds())
	draw.Copy(srcrgba, image.Point{}, *src, (*src).Bounds(), draw.Src, nil)

	for y := 0; y < height; y++ {
		row := make([]RGB, width)
		for x := 0; x < width; x++ {
			idxS := (y*width + x) * 4
			pix := srcrgba.Pix[idxS : idxS+4]
			row[x] = RGB{float64(pix[0]), float64(pix[1]), float64(pix[2])}
		}
		iaa[y] = row
	}
	return &iaa
}

func array2Imge(src *[][]RGB) *image.RGBA {
	width, height := len((*src)[0]), len((*src))
	rect := image.Rect(0, 0, width, height)
	theImage := image.NewRGBA(rect)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			Col := (*src)[y][x]
			theImage.Set(x, y, color.RGBA{uint8(Col.R), uint8(Col.G), uint8(Col.B), 0xff})
		}
	}
	return theImage
}

func maxBooling(src *[][]RGB) {
	finalImage := make([][]RGB, 0)
	for y, aY := 0, 0; y+2 < len(*src); y, aY = y+2, aY+1 {
		finalImage = append(finalImage, []RGB{})
		for x := 0; x+2 < len((*src)[0]); x = x + 2 {
			//RED
			y1x1, y1x2 := &(*src)[y][x], &(*src)[y][x+1]
			y2x1, y2x2 := &(*src)[y+1][x+1], &(*src)[y+1][x+1]

			Reds := []float64{y1x1.R, y1x2.R, y2x1.R, y2x2.R}
			sort.Float64s(Reds)

			Greens := []float64{y1x1.G, y1x2.G, y2x1.G, y2x2.G}
			sort.Float64s(Greens)

			Blues := []float64{y1x1.B, y1x2.B, y2x1.B, y2x2.B}
			sort.Float64s(Blues)

			Col := RGB{Reds[3], Greens[3], Blues[3]}
			finalImage[aY] = append(finalImage[aY], Col)
		}
	}
	*src = finalImage
}

// func main() {
// 	aa := time.Now()
// 	data := process("image-test.jpg", convFilter)
// 	// fmt.Println(data)
// 	img := array2Imge(data)
// 	i := img.SubImage(img.Rect)
// 	saveImage(&i, "test.jpg")
// 	bb := time.Since(aa)
// 	fmt.Println("At Time: ", bb)

// }

func process(imagePath string, filter [11]float64, yDim, xDim int) *[][]RGB {
	testImage, err := os.Open(imagePath)
	defer testImage.Close()
	errorHandler(err)

	image.RegisterFormat("jpeg", "?", jpeg.Decode, jpeg.DecodeConfig)
	theImage, _, err := image.Decode(testImage)
	errorHandler(err)

	imgArr := image2Array(&theImage)
	fixDim(imgArr, yDim, xDim)

	applykernel(imgArr, gaussianBlur, len(*imgArr), len((*imgArr)[0]))
	applykernel(imgArr, filter, len(*imgArr), len((*imgArr)[0]))
	maxBooling(imgArr)
	applykernel(imgArr, filter, len(*imgArr), len((*imgArr)[0]))
	maxBooling(imgArr)
	applykernel(imgArr, filter, len(*imgArr), len((*imgArr)[0]))
	maxBooling(imgArr)

	return imgArr
}

func SetUpTrainingData(imagePath string, yDim, xDim int) *[]float64 {

	img1 := process(imagePath, convFilter, yDim, xDim)
	data := make([]float64, 0)
	for _, v := range *img1 {
		for _, Col := range v {
			data = append(data, Col.R/0xff)
		}
	}

	img2 := process(imagePath, leftEdge, yDim, xDim)
	for _, v := range *img2 {
		for _, Col := range v {
			data = append(data, Col.R/0xff)
		}
	}

	img3 := process(imagePath, rightEdge, yDim, xDim)
	for _, v := range *img3 {
		for _, Col := range v {
			data = append(data, Col.R/0xff)
		}
	}

	img4 := process(imagePath, topEdge, yDim, xDim)
	for _, v := range *img4 {
		for _, Col := range v {
			data = append(data, Col.R/0xff)
		}
	}

	img5 := process(imagePath, downEdge, yDim, xDim)
	for _, v := range *img5 {
		for _, Col := range v {
			data = append(data, Col.R/0xff)
		}
	}

	return &data
}
