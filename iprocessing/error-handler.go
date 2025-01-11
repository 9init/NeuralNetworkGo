package iprocessing

import "log"

func errorHandler(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
