# Omejevanje prepustnosti

Pri nudenju spletnih servisov želimo velikokrat omejiti število zahtevkov nekega tipa, ki se lahko sprocesirajo v nekem določenem časovnem intervalu, da preprečimo izstradanje nekega drugega dela sistema. **Go** nam nudi različne možnosti, kako to dosežemo.

Primer omejevanja prepustnosti:
```Go
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
```
Zgornji način je primeren za omejevanje prepustnosti do nekaj 10/100 zahtevkov na sekundo. Za doseganje višjih prepustnosti je priporočena uporaba paketa `rate` ali paketa `github.com/uber-go/ratelimit`. 

## Naloga

Navodila za drugo domačo nalogo najdete [tukaj](../naloga-2/naloga-2.md).