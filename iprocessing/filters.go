package iprocessing

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
	0.0625, 14,
}
