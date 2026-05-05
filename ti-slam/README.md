# TI-SLAM: Lokalizacja i mapowanie z użyciem kamery termowizyjnej

## Opis projektu

Projekt implementuje system **TI-SLAM**, którego celem jest estymacja trajektorii oraz poprawa dokładności lokalizacji z wykorzystaniem danych z kamery termowizyjnej.

System łączy:

* dane z kamery termowizyjnej,
* dane z IMU

Rozwiązanie pozwala ograniczyć dryf występujący w klasycznej odometrii inercyjnej.

---

## Główne funkcjonalności

* SLAM oparty na danych termicznych (działa w słabych warunkach oświetleniowych)
* Pipeline w Pythonie 

---

## Struktura projektu

```text
ti-slam/
│
├── Python/
│   └── (główna logika przetwarzania i SLAM)
│
├── figures/
│   └── (wizualizacje i wyniki)
│
├── spatial_consistency.py
├── robust_pose_graph_optimization.py
└── README.md
```

Projekt bazuje na: https://github.com/risqiutama/ti-slam  
Autor: Muhamad Risqi Utama Saputra  

Licencja: CC BY 4.0, zmodyfikowany 
