# Posredovanje sporočil

## Programski vmesniki

### Operacije CRUD

- akronim CRUD (*angl.* create, read, update, delete) se nanaša na štiri osnovne operacije na podatkovnih shrambah: ustvarjanje, branje, posodabljanje in brisanje zapisov
- paket [`storage`](koda/storage/storage.go) predstavlja enostaven primer shrambe
    - osnova je slovar `TodoStorage` za shranjevanje nalog `Todo`
    - vključuje funkcijo `NewTodoStorage`, ki shrambo ustvari
    - jezik go ni objekten, pozna pa metode, ki jih lahko dodajamo strukturam 
    - metodo definiramo podobno kot funkcijo, le da za ključno besedo `func` napišemo strukturo ali kazalec nanjo
        ```go
        func (tds *TodoStorage) Create(todo *Todo, ret *struct{}) error {
            ...
        }
        ```
    - paket vključuje štiri metode: `Create`, `Read`, `Update` in `Delete`
- osnovno uporabo shrambe prikazuje program [`uporaba.go`](koda/storage/uporaba/uporaba.go)
    - shrambo najprej ustvarimo
        ```go
        store := storage.NewTodoStorage()
        ```
    - nato pa na spremenljivki `store` kličemo želene metode, na primer
        ```go
        lecturesCreate := storage.Todo{Task: "predavanja", Completed: false}
        err := store.Create(&lecturesCreate, &reply)
        ```
- v nadaljevanju bo shramba `TodoStore` tekla na strežniku, odjemalci pa bodo dostopali do nje preko programskih vmesnikov
- zato, da je gradnja vmesnikov sploh mogoča, je pri načrtovanju shrambe potrebno upoštevati določena pravila
    - jezik go v programski vmesnik za komunikacijo med procesi RPC vključi samo metode, ki imajo dva argumenta (vhodni argument in kazalec na odgovor) in vračajo tip `error`
        ```go
        func (t *T) ImeMetode(argument T1, odgovor *T2) error
        ```
    - za programski vmesnik HTTP (Restful), moramo ob strukturi podati preslikavo v format JSON (*angl.* javascript object notation)
        ```go
        type Todo struct {
    	    Task      string `json:"task"`
    	    Completed bool   `json:"completed"`
        }
        ```

### Namen programskega vmesnika

- želimo sistem, v katerem odjemalec kliče operacije (funkcije, procedure, metode), ki jih ponuja strežnik
- strežnik operacije ponuja preko programskega vmesnika (*angl.* application programming interface)
- vmesnik skrbi za prevajanje sporočil, prejetih iz omrežja, v klice funkcij in metod na strežniku
- komunikacija je lahko 
    - neposredna: odjemalec in strežnik komunicirata neposredno; v tem primeru morata biti aktivna oba
    - posredna: odjemalec in strežnik komunicirata preko posrednika; smiselno, če ne moremo zagotoviti neposredne komunikacije; odjemalec ima več avtonomije, počasnejša komunikacija
- zahteva in odgovor morata biti standardizirana (tekstovna XML in JSON, binarni Protocol Buffers)
- komunikacija je lahko 
    - sinhrona: odjemalec blokira, dokler strežnik ne odgovori; zaradi čakanja na odgovor je neučinkovita
    - asinhrona: odjemalec nadaljuje izvajanje kode, ko prispe odgovor se izvede povratni klic (*angl.* callback); v jeziku go to dosežemo z gorutinami, mnogi drugi jeziki (javascript, C#) poznajo ključne besede kot sta async/await
- danes pogosto uporabljane tehnologije za komunikacijo med procesi
    - RPC (*angl.* remote procedure call): za interno komunikacijo med procesi, napisanimi v istem programskem jeziku
    - gRPC: postaja nov standard, uporablja HTTP/2, zaradi binarnega zapisovanja (Protocol Buffers) je hitrejši od HTTP/1.1
    - HTTP: najbolj uporabljana tehnologija za javne strežnike, podpirajo jo vsi brskalniki preko kode javascript, HTTP/1.1


### RPC
- vzorec RPC (*angl.* remote procedure call) 

- vzorec klicanja oddaljenih metod; programer oddaljene metode kliče na podoben način kot lokalne
    - strežnik nudi storitve v obliki metod, ki jih je mogoče klicati oddaljeno
    - odjemalec lahko pokliče metodo na strežniku, kot bi šlo za lokalni klic
    - metoda vključuje argumente (vhod in izhod)
    - koda posamezne metode se dejansko izvede na strežniku 
    - na odjemalcu kličemo metode na enak način kot na strežniku, le da metoda na odjemalcu (*angl.* stub) vključuje le mehanizme za klic metode na strežniku
        - metoda na odjemalcu vključuje kodiranje (*angl.* marshalling), prenos argumentov na strežnik in zahtevo za izvajanje
        - metoda na strežniku argumente dekodira (*angl.* unmarshalling) in zažene metodo z dekodiranimi argumenti; rezultate izvajanja kodira in pošlje nazaj odjemalcu
        - metoda na odjemalcu dekodira sporočilo, ga zapiše v zahtevane strukture in vrne kot rezultat
    - za razliko od metod, ki se dejansko izvajajo na odjemalcu, pride pri izvajanju na strežniku lahko do napak: pri prenosu argumentov, med izvajanjem, pri prenosu rezultatov
        - kje se je zataknilo, kako se odzvati (ponoven prenos, koliko časa čakati, ...)

- zgodovinski razvoj
    - SunRPC, osnova za porazdeljene podatkovne sisteme, 1980
    - CORBA (*angl.* common object request broker architecture) (1990)
    - Microsoft DCOM (*angl.* distributed component object model), 1996
    - Java RMI, 1997
    - SOAP (*angl.* simple object access protocol), običajno v navezi z XLM, 1998
    - AJAX (*angl.* asynchronous javascript and XML), 1999
    - REST, v navezi z JSON, 2000
    - Google gRPC, 2015

### gRPC
- [gRPC](https://grpc.io/)
- vzorec klicanja oddaljenih metod, tako kot RPC
- Google ga je zasnoval za učinkovit prenos podatkov med mikrostoritvami
- moderni pristopi uporabljajo jezike za opis vmesnika (*angl.* interface definition language, IDL)
- za kodiranje podatkov uporablja binarni protokol [Protocol Buffers](https://protobuf.dev/)
- za prenos podatkov uporablja protokol HTTP/2, možnost overovitve pošiljatelja
- omogoča dvosmerni tok podatkov, majhna latenca, velika pasovna širina
- primeren za kompleksne programske vmesnike
- za razliko od RPC, ki je omejen samo na jezik go, lahko gRPC uporabljamo v množici podprtih programskih jezikov

    <img src="slike/grpc.png" width="60%" />

#### Protocol Buffers
- gRPC privzeto uporablja protokol [Protocol Buffers](https://protobuf.dev/overview) za kodiranje (*angl.* marshalling) strukturiranih podatkov
    - pripravimo opisno datoteko, ki vključuje strukture in metode
    - prevajalnik `protoc` nam iz opisne datoteke pripravi paket, s katerim elegantno izvajamo klice metod na strežniku
        - podrobnosti o namestitvi paketa najdete v komentarju k datoteki [protobufStorage.proto](koda/grpc/protobufStorage/protobufStorage.proto)
    - paket prevedemo skupaj z našo kodo
    - prevedeni paket ali izvršljivo datoteko uporabljamo za izvajanje klicev oddaljenih metod

        <img src="slike/protocolbuffers.png" width="70%" />

- primer: opisna datoteka za shrambo [protobufStorage.proto](koda/grpc/protobufStorage/protobufStorage.proto)
    - podamo ime paketa
    - podamo strukture in podatkovne tipe; vsaki spremenljivki priredimo številko; številke znotraj iste strukture morajo biti različne
    - definiramo storitev (v našem primeru `CRUD˙) ter metode z argumenti in vrednostmi, ki jih vračajo
    - če prevajalnik `protoc` zaženemo s stikali
        ```bash
        srun protoc --go_out=. --go_opt=paths=source_relative --go-grpc_out= --go-grpc_opt=paths=source_relative protobufStorage.proto
        ```
        nam kar v isti mapi pripravi paket za go, v našem primeru `main/grpc/protobufStorage`, ki vključuje dve datoteki [protobufStorage.pb.go](koda/grpc/protobufStorage/protobufStorage.pb.go) in [protobufStorage_grpc.pb.go](koda/grpc/protobufStorage/protobufStorage_grpc.pb.go)
    - paket, ki ga tvorita ti dve datoteki, uporabimo pri pisanju strežnika in odjemalca

#### Strežnik in odjemalec
- [primer gRPC](koda/grpc/grpc.go) ([strežnik](koda/grpc/streznik.go) in [odjemalec](koda/grpc/odjemalec.go))
    - strežnik 
        - pripravimo strukturo in metode, pri tem izhajamo iz predloge (`protobufStorage.UnimplementedCRUDServer`)
        - v strukturo dodamo potrebne elemente
        - pripravimo kodo za vse metode, ki jih strežnik podpira
            - vsaka metoda vključuje izvajalno okolje (*angl.* context), vhodne argumente in vrnjene vrednosti (kazalec na strukturo in napako)
            - primer
                ```go
                func (s *server) Create(ctx context.Context, in *protobufStorage.Todo) (*emptypb.Empty, error) {
                    ...
                }
                ```
            - izvajalno okolje skrbi za nadzorovano izvajanje gorutin (gorutine si lahko izmenjujejo informacije o globalnem stanju aplikacije, na primer, lahko jih prekličemo zaradi časovne omejitve)
        - na strežniku prijavimo metode gRPC in zaženemo strežnik gRPC
    - odjemalec 
        - se povežemo na strežnik s potrebnimi pravicami
        - vzpostavimo izvajalno okolje
            ```go
            contextCRUD, cancel := context.WithTimeout(context.Background(), time.Second)
            ```
        - vzpostavimo vmesnik gRPC, 
            ```go
            grpcClient := protobufStorage.NewCRUDClient(conn)
            ```
        - kličemo oddaljene metode, kot bi bile na voljo lokalno
            ```go
            _, err := grpcClient.Create(contextCRUD, &lecturesCreate)
            ```
