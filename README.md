# Snake Game

Ein klassisches Snake-Spiel implementiert in Python mit Pygame.

## Installation

1. Stelle sicher, dass Python 3.6+ installiert ist
2. Installiere die Abhängigkeiten:
```bash
pip install -r requirements.txt
```

## Spielen

Starte das Spiel mit:
```bash
python snake_game.py
```

## Steuerung

- **Pfeiltasten**: Schlange steuern
- **P oder SPACE**: Spiel pausieren/fortsetzen
- **R**: Neustart (nach Game Over)
- **SPACE**: Spiel starten (vom Menü aus)

## Spielregeln

- Die Schlange bewegt sich kontinuierlich in eine Richtung
- Steuere die Schlange mit den Pfeiltasten
- Sammle das rote Essen, um zu wachsen und Punkte zu sammeln
- Vermeide Kollisionen mit den Wänden und dem eigenen Körper
- Die Geschwindigkeit erhöht sich mit steigendem Score

## Features

- ✅ Objektorientierte Programmierung mit klaren Klassen
- ✅ Start-Bildschirm mit Anweisungen
- ✅ Game-Over-Bildschirm mit Neustart-Option
- ✅ Pausenfunktion
- ✅ Score-System (+10 Punkte pro Essen)
- ✅ Geschwindigkeitserhöhung mit steigendem Score
- ✅ FPS-Anzeige
- ✅ Highscore-Tracking
- ✅ Sauberer, gut kommentierter Code
- ✅ Verhindert direkte Umkehr der Schlange
- ✅ Essen spawnt nie auf der Schlange

## Code-Struktur

- `Snake`: Verwaltet Position, Richtung, Bewegung und Wachstum
- `Food`: Verwaltet Position und Neupositionierung des Essens
- `Game`: Hauptspiellogik, Kollisionserkennung und Rendering

## Technische Details

- Spielfeld: 600x600 Pixel
- Grid-basiert: 20x20 Felder
- FPS: 60 (begrenzt)
- Schlangengeschwindigkeit: 10-20 FPS (abhängig vom Score)