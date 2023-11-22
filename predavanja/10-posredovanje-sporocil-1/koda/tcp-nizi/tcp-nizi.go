// Komunikacija po protokolu TCP
// 		niz pred pošiljanjem pretvorimo v []byte, ob prejemu pa []byte v niz
//
// 		odjemalec pošlje strežniku sporočilo, dopolnjeno s časovnim žigom
//		strežnik sporočilo dopolni, zamenja časovni žig in ga pošlje odjmelcu
//		pred pošiljanjem počaka 5 s, da lažje pokažemo hkratno streženje več odjemalcem
//
// zaženemo strežnik
// 		go run *.go
// zaženemo enega ali več odjemalcev
//		go run *.go -s [ime strežnika] -p [vrata] -m [sporočilo]
// za [ime strežnika] in [vrata] vpišemo vrednosti, ki jih izpiše strežnik ob zagonu
//
// pri uporabi SLURMa lahko s stikalom --nodelist=[vozlišče] določimo vozlišče, kjer naj se zažene program

package main

import (
	"flag"
	"fmt"
)

func main() {
	// preberemo argumente iz ukazne vrstice
	sPtr := flag.String("s", "", "server URL")
	pPtr := flag.Int("p", 9876, "port number")
	mStr := flag.String("m", "world", "message")
	flag.Parse()

	// zaženemo strežnik ali odjemalca
	url := fmt.Sprintf("%v:%v", *sPtr, *pPtr)
	if *sPtr == "" {
		Server(url)
	} else {
		Client(url, *mStr)
	}
}
