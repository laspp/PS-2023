// pozdrav
// dve gorutini, argument, vrstni red izpisov se od zagona do zagona spreminja
// glavna gorutina tudi dela

package main

import (
	"fmt"
	"strconv"
)

const printouts = 10

var stringStream chan string

func hello(s string) {
	defer close(stringStream)
	fmt.Println("G", s, "start")
	for i := 0; i < printouts; i++ {
		stringStream <- s + "-" + strconv.Itoa(i)
	}
	fmt.Println("G", s, "end")
}

func main() {

	stringStream = make(chan string)

	go hello("hello-world")

	for msg := range stringStream {
		fmt.Print(msg, " ")
	}

	fmt.Println()
}
