# CUDA - primeri programov

## Primer: računanje razlike vektorjev

- imamo vektorja $\mathbf{a}=(a_0, \ldots, a_{N-1})$ in $\mathbf{b}=(a_0, \ldots, a_{N-1})$
- razlika med njima je enaka $\mathbf{c} = \mathbf{a} - \mathbf{b}$

- [razlika-1.cu](koda/razlika/razlika-1.cu)
  - gostitelj
    - preberemo argumente iz ukazne vrstice
    - rezerviramo pomnilnik za vektorje na gostitelju (`ha`, `hb`, `hc`)
    - nastavimo vrednosti vektorjev na gostitelju
    - rezerviramo pomnilnik za vektorje na napravi (`da`, `db`, `dc`)
    - prenesemo vektorja $\mathbf{a}$ in $\mathbf{b}$ iz gostitelja na napravo
    - na napravi zaženemo ščepec, ki izračuna razliko $\mathbf{c}$
    - prenesemo vektor $\mathbf{c}$ iz naprave na gostitelja
    - preverimo rezultat
    - sprostimo pomnilnik na napravi
    - sprostimo pomnilnik na gostitelju
  - naprava
    - vsaka nit izračuna razliko para istoležnih elementov v vektorjih $\mathbf{a}$ in $\mathbf{b}$
    - indeks niti določa kateri par elementov obdela posamezna nit
    - slaba rešitev: indeks lahko presega velikost tabele
- [razlika-2.cu](koda/razlika/razlika-2.cu)
  - preverimo velikost tabele
  - še vedno podpora za samo en blok niti
- [razlika-3.cu](koda/razlika/razlika-3.cu)
  - podpora za več blokov
  - napačen rezultat, če je elementov več kot niti
- [razlika-4.cu](koda/razlika/razlika-4.cu)
  - pravilna rešitev
  - uporabnik določa število blokov
- [razlika-5.cu](koda/razlika/razlika-5.cu)
  - pravilna rešitev
  - izračun potrebnega števila blokov
- [razlika-6.cu](koda/razlika/razlika-6.cu)
  - rešitev z enotnim pomnilnikom

## Primer: razdalja med vektorjema

- imamo vektorja $\mathbf{a}=(a_0, \ldots, a_{N-1})$ in $\mathbf{b}=(a_0, \ldots, a_{N-1})$
- razdalja med njima je enaka

  $$\mathrm{dist} = \sqrt{\sum_{i=0}^{N-1} (a_i - b_i)^2 }$$

- [razdalja-01.cu](koda/razdalja/razdalja-01.cu)
  - nadgradimo rešitev [razlika-05.cu](../20-cuda/koda/razlika-5.cu)
  - razlike med elementi vektorjev izračunamo na napravi
  - na gostitelju seštejemo kvadrate razlik in izračunamo koren vsote

- [razdalja-02.cu](koda/razdalja/razdalja-02.cu)
  - kvadriranje izvedemo na napravi
  - zaporedno seštevanje vseh kvadratov in koren vsote izvedemo na gostitelju

- [razdalja-03.cu](koda/razdalja/razdalja-03.cu)
  - naprava
    - izračunamo vsote kvadratov za vsak blok niti
    - uporabimo skupni pomnilnik, rezerviramo ga statično
    - niti kvadrate razlik shranijo v skupni pomnilnik
    - ko vse niti izračunajo svoje kvadrate, naredimo pregrado (`__syncThreads()`)
    - na koncu delno vsoto kvadratov za blok izračuna prva nit z lokalnim indeksom `threadIdx.x = 0`, ki zaporedno sešteje vse kvadrate razlik, ki so jih niti prej shranile v skupni pomnilnik
  - gostitelj
    - iz naprave kopiramo vektor delnih vsot, ki ima toliko elementov, kot je blokov niti
    - zaporedno seštejemo delne vsote in izračunamo koren vsote

<img src="slike/razdalja-skupni-pomnilnik.png" width="80%" />
