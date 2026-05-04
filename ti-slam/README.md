# TI-SLAM: Lokalizacja i mapowanie z użyciem kamery termowizyjnej

## Opis projektu

Projekt implementuje system **TI-SLAM**, którego celem jest estymacja trajektorii oraz poprawa dokładności lokalizacji z wykorzystaniem danych z kamery termowizyjnej.

System łączy:

* dane z kamery termowizyjnej,
* detekcję domknięć pętli (loop closure),
* optymalizację grafu pozycji (pose graph optimization).

Rozwiązanie pozwala ograniczyć dryf występujący w klasycznej odometrii inercyjnej.

---

## Główne funkcjonalności

* SLAM oparty na danych termicznych (działa w słabych warunkach oświetleniowych)
* Detekcja domknięć pętli na podstawie embeddingów
* Globalna optymalizacja trajektorii (pose graph)
* Pipeline w Pythonie i MATLABie

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
