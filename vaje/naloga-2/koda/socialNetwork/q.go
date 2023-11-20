/*
Paket socialNetwork nudi funkcije za ustvarjanje, zagon in ustavitev generatorja zahtevkov za indeksiranje objav
*/
package socialNetwork

import (
	_ "embed"
	"math/rand"
	"strings"
	"time"
)

// Definiramo dve različni prioriteti
const (
	LowPriority  int = 0
	HighPriority int = 1
)

// Preberemo objave in ključne besede iz datoteke
//
//go:embed modrosti.txt
var content string

//go:embed besede.txt
var words string

// Podatkovna struktura Task, ki predstavlja posamezni zahtevek
type Task struct {
	Id       uint64
	TaskType string
	Data     string
}

// Podatkovna struktura Q, ki vlkjučuje vse potrebne spremenljivke za generiranje zahtevkov
type Q struct {
	Types          [2]string
	N              [2]uint64
	PriorityLowP   float64
	TaskChan       chan Task
	quit           chan bool
	rnd            *rand.Rand
	listOfFortunes []string
	wordList       []string
}

// Ustvari nov generator, inicializira vse podatkovne struture
// S parametrom PriorityLowP nastavimo verjetnost generiranja zahtevka z nižjo prioriteto
func (load *Q) New(PriorityLowP float64) {
	load.listOfFortunes = strings.Split(content, "\n%\n")
	load.wordList = strings.Split(words, "\n")
	load.rnd = rand.New(rand.NewSource(time.Now().UnixNano()))
	load.Types[LowPriority] = "search"
	load.Types[HighPriority] = "index"
	load.PriorityLowP = PriorityLowP
	load.TaskChan = make(chan Task)
	load.quit = make(chan bool)
}

// Zaženemo generator
func (load *Q) Run() {
	var newTask Task
	for {
		select {
		case <-load.quit:
			close(load.TaskChan)
			return
		default:
			if load.rnd.Float64() < load.PriorityLowP {
				i := load.rnd.Intn(len(load.wordList))
				newTask = Task{Id: load.N[LowPriority], TaskType: load.Types[LowPriority], Data: load.wordList[i]}
				load.N[LowPriority]++
			} else {
				i := load.rnd.Intn(len(load.listOfFortunes))
				newTask = Task{Id: uint64(i), TaskType: load.Types[HighPriority], Data: load.listOfFortunes[i]}
				load.N[HighPriority]++
			}
			//
			load.TaskChan <- newTask
			for d := 1; d < load.rnd.Intn(3000); d++ {
				//Naključna zakasnitev do naslednjega zahtevka
			}
		}
	}
}

// Ustavimo generator
func (load *Q) Stop() {
	load.quit <- true
	close(load.quit)
}
