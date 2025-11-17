# Podsumowanie

## Architekura sieci
Sieć składa się z trzech warstw, w tym jednej ukrytej. Wejściowa ma rozmiar 784, ukryta 32 i funkcję aktywacji ReLU, a wyjściowa jest rozmiaru 10 z 330 parametrami.  

## Opis eksperymentu
Wykorzystałem tuner do dostrojenia parametru uczenia. Nie udało mi się sprawić, by sieć neuronowa uzyskała lepszy wynik od drzewa losowego, w dodatku pogorszyłem dokładność modelu o ok. 0.03. Doprowadziłem do tego poprzez ustawienie zakresu dostrajania parametru uczenia na 0.0001-0.01, ilość prób na 10, z jednym wykonaniem programu w każdej próbie.

## Wnioski
Tuner wydaje się być bardzo efektywnym i przydatnym narzędziem do znajdywania najlepszych parametrów dla modelu.