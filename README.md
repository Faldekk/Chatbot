# Chatbot

Eksperyment w Pythonie i PyTorch z modelem językowym, który przewiduje następny znak na podstawie wcześniejszego tekstu. Model ćwiczy na lokalnym pliku tekstowym, a po treningu generuje próbkę na podstawie pierwszego znaku z danych.

## Trening

1. Zainstaluj Python i PyTorch.
2. Umieść plik `training_data.txt` z tekstem w kodowaniu UTF-8 obok `Training.py`.
3. W katalogu projektu uruchom `python Training.py`.

Skrypt tworzy słownik znaków, dzieli dane na zbiory treningowy i walidacyjny oraz trenuje model z blokami transformera. Co 100 kroków wypisuje stratę dla obu zbiorów. Jeśli PyTorch wykryje CUDA, trening użyje GPU. W przeciwnym razie użyje CPU.

## Tryb rozmowy

`Chat.py` zawiera prototyp rozmowy w konsoli. Aktualny skrypt odwołuje się do klasy i pliku z wagami, których `Training.py` nie udostępnia. Uruchomienie rozmowy wymaga dopasowania skryptu do klasy `TextModel` i dodania zapisu wag po treningu.
