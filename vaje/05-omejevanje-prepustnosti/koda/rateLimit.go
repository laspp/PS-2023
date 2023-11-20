/*
Koda prikazuje primer omejevanja prepustnosti v golang s pomočjo paketa time
*/
package main

import (
	"fmt"
	"time"
)

// 10 zahtevkov na sekundo
const rateLimit = time.Second / 10

// Funkcija, ki simulira delo
func doWork() {
	fmt.Println(time.Now())
}

func main() {
	//Ustvarimo kanal, ki bo diktiral delo
	rateLimiter := make(chan time.Time)

	go func() {
		ticker := time.NewTicker(rateLimit)
		for t := range ticker.C {
			rateLimiter <- t
		}
	}()

	for {
		<-rateLimiter
		doWork()
	}
}
