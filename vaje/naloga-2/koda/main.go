/*
 */
package main

import (
	"fmt"
	"pc/socialNetwork"
	"time"
)

func main() {
	var producer socialNetwork.Q
	producer.New(0.5)

	start := time.Now()

	go func() {
		for {
			<-producer.TaskChan
		}
	}()
	go producer.Run()
	time.Sleep(time.Second * 2)
	producer.Stop()
	elapsed := time.Since(start)
	fmt.Printf("Spam rate: %f MReqs/s\n", float64(producer.N[socialNetwork.LowPriority]+producer.N[socialNetwork.HighPriority])/float64(elapsed.Seconds())/1000000.0)
}
