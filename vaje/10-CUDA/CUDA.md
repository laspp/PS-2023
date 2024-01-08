# Programiranje grafičnih procesnih enot CUDA

Na gruči Arnes je na voljo več vozlišč z računskimi karticami Nvidia [V100](https://www.nvidia.com/en-us/data-center/v100/) in [H100](https://www.nvidia.com/en-us/data-center/h100/)

Primer zagona programa `nvidia-smi` na gruči. Program izpiše podatke o računskih karticah, ki so na voljo na danem računskem vozlišču.
```Bash
$ srun --partition=gpu --reservation=psistemi -G1 nvidia-smi --query
```

Napišimo še lasten [program](./koda/discover-device.cu) v programskem jeziku C, ki izpiše informacije o GPE. Podpora za programiranje grafičnih procesnih enot v programskem jeziku go, je omejena, zato se bomo poslužili jezika C/C++.

Najprej program prevedemos pomočjo prevajalnika nvcc:
```Bash
$ module load CUDA
$ nvcc discover-device.cu -o discover-device
```
in poženemo:
```Bash
$ srun --partition=gpu --reservation=psistemi -G1 discover-device
```

## Naloga

Navodila za peto domačo nalogo najdete [tukaj](../naloga-5/naloga-5.md).
