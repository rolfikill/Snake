import pygame
import random
import sys

# Pygame initialisieren
pygame.init()

# Konstanten
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = WINDOW_WIDTH // GRID_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // GRID_SIZE

# Farben
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
DARK_GREEN = (0, 200, 0)
GRAY = (128, 128, 128)

# Richtungen
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class Snake:
    """Klasse für die Schlange mit Position, Richtung, Bewegung und Wachstum"""
    
    def __init__(self):
        # Startposition in der Mitte des Spielfelds
        self.body = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = RIGHT
        self.grow = False
        
    def move(self):
        """Bewegt die Schlange in die aktuelle Richtung"""
        head_x, head_y = self.body[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        
        # Neuen Kopf hinzufügen
        self.body.insert(0, new_head)
        
        # Wenn die Schlange nicht wachsen soll, Schwanz entfernen
        if not self.grow:
            self.body.pop()
        else:
            self.grow = False
    
    def change_direction(self, new_direction):
        """Ändert die Richtung der Schlange (verhindert direkte Umkehr)"""
        # Verhindert direkte Umkehr
        if (self.direction[0] * -1, self.direction[1] * -1) != new_direction:
            self.direction = new_direction
    
    def grow_snake(self):
        """Lässt die Schlange beim nächsten Zug wachsen"""
        self.grow = True
    
    def check_collision(self):
        """Prüft auf Kollisionen mit Wänden oder eigenem Körper"""
        head_x, head_y = self.body[0]
        
        # Kollision mit Wänden
        if (head_x < 0 or head_x >= GRID_WIDTH or 
            head_y < 0 or head_y >= GRID_HEIGHT):
            return True
        
        # Kollision mit eigenem Körper
        if self.body[0] in self.body[1:]:
            return True
        
        return False
    
    def draw(self, screen):
        """Zeichnet die Schlange auf den Bildschirm"""
        for i, segment in enumerate(self.body):
            x = segment[0] * GRID_SIZE
            y = segment[1] * GRID_SIZE
            
            # Kopf in anderer Farbe als Körper
            color = GREEN if i == 0 else DARK_GREEN
            pygame.draw.rect(screen, color, (x, y, GRID_SIZE, GRID_SIZE))
            pygame.draw.rect(screen, BLACK, (x, y, GRID_SIZE, GRID_SIZE), 1)

class Food:
    """Klasse für das Essen mit Position und Neupositionierung"""
    
    def __init__(self):
        self.position = self.generate_position()
    
    def generate_position(self):
        """Generiert eine zufällige Position für das Essen"""
        x = random.randint(0, GRID_WIDTH - 1)
        y = random.randint(0, GRID_HEIGHT - 1)
        return (x, y)
    
    def respawn(self, snake_body):
        """Positioniert das Essen neu, außerhalb der Schlange"""
        while True:
            self.position = self.generate_position()
            if self.position not in snake_body:
                break
    
    def draw(self, screen):
        """Zeichnet das Essen auf den Bildschirm"""
        x = self.position[0] * GRID_SIZE
        y = self.position[1] * GRID_SIZE
        pygame.draw.rect(screen, RED, (x, y, GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(screen, BLACK, (x, y, GRID_SIZE, GRID_SIZE), 1)

class Game:
    """Hauptspiellogik, Kollisionserkennung und Rendering"""
    
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Spielzustand
        self.state = "MENU"  # MENU, PLAYING, PAUSED, GAME_OVER
        self.score = 0
        self.high_score = 0
        self.speed = 10  # FPS für die Schlange
        
        # Spielobjekte
        self.snake = Snake()
        self.food = Food()
        
        # Sicherstellen, dass Essen nicht auf der Schlange spawnen
        self.food.respawn(self.snake.body)
    
    def handle_events(self):
        """Behandelt alle Eingabe-Events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if self.state == "MENU":
                    if event.key == pygame.K_SPACE:
                        self.start_game()
                
                elif self.state == "PLAYING":
                    if event.key == pygame.K_UP:
                        self.snake.change_direction(UP)
                    elif event.key == pygame.K_DOWN:
                        self.snake.change_direction(DOWN)
                    elif event.key == pygame.K_LEFT:
                        self.snake.change_direction(LEFT)
                    elif event.key == pygame.K_RIGHT:
                        self.snake.change_direction(RIGHT)
                    elif event.key == pygame.K_p or event.key == pygame.K_SPACE:
                        self.state = "PAUSED"
                
                elif self.state == "PAUSED":
                    if event.key == pygame.K_p or event.key == pygame.K_SPACE:
                        self.state = "PLAYING"
                
                elif self.state == "GAME_OVER":
                    if event.key == pygame.K_r:
                        self.restart_game()
        
        return True
    
    def start_game(self):
        """Startet ein neues Spiel"""
        self.state = "PLAYING"
        self.score = 0
        self.speed = 10
        self.snake = Snake()
        self.food = Food()
        self.food.respawn(self.snake.body)
    
    def restart_game(self):
        """Startet das Spiel neu"""
        self.start_game()
    
    def update(self):
        """Aktualisiert die Spiellogik"""
        if self.state == "PLAYING":
            self.snake.move()
            
            # Kollisionsprüfung
            if self.snake.check_collision():
                self.state = "GAME_OVER"
                if self.score > self.high_score:
                    self.high_score = self.score
                return
            
            # Essen aufnehmen
            if self.snake.body[0] == self.food.position:
                self.snake.grow_snake()
                self.score += 10
                self.speed = min(20, 10 + self.score // 50)  # Geschwindigkeit erhöhen
                self.food.respawn(self.snake.body)
    
    def draw_menu(self):
        """Zeichnet den Start-Bildschirm"""
        self.screen.fill(BLACK)
        
        # Titel
        title = self.font.render("SNAKE GAME", True, WHITE)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 100))
        self.screen.blit(title, title_rect)
        
        # Anweisungen
        instructions = [
            "Steuerung: Pfeiltasten",
            "Pause: P oder SPACE",
            "Neustart: R (nach Game Over)",
            "",
            "Drücke SPACE zum Starten"
        ]
        
        y_offset = 150
        for instruction in instructions:
            text = self.small_font.render(instruction, True, WHITE)
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + y_offset))
            self.screen.blit(text, text_rect)
            y_offset += 30
    
    def draw_game(self):
        """Zeichnet das Spiel"""
        self.screen.fill(BLACK)
        
        # Score anzeigen
        score_text = self.small_font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        high_score_text = self.small_font.render(f"High Score: {self.high_score}", True, WHITE)
        self.screen.blit(high_score_text, (10, 35))
        
        # FPS anzeigen
        fps_text = self.small_font.render(f"FPS: {int(self.clock.get_fps())}", True, WHITE)
        self.screen.blit(fps_text, (WINDOW_WIDTH - 100, 10))
        
        # Schlange und Essen zeichnen
        self.snake.draw(self.screen)
        self.food.draw(self.screen)
        
        # Pause-Anzeige
        if self.state == "PAUSED":
            pause_text = self.font.render("PAUSIERT", True, WHITE)
            pause_rect = pause_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            self.screen.blit(pause_text, pause_rect)
    
    def draw_game_over(self):
        """Zeichnet den Game-Over-Bildschirm"""
        self.screen.fill(BLACK)
        
        # Game Over Text
        game_over_text = self.font.render("GAME OVER", True, RED)
        game_over_rect = game_over_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 100))
        self.screen.blit(game_over_text, game_over_rect)
        
        # Finale Punktzahl
        final_score_text = self.font.render(f"Final Score: {self.score}", True, WHITE)
        final_score_rect = final_score_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50))
        self.screen.blit(final_score_text, final_score_rect)
        
        # High Score
        if self.score == self.high_score and self.score > 0:
            new_high_text = self.small_font.render("NEUER HIGHSCORE!", True, GREEN)
            new_high_rect = new_high_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 10))
            self.screen.blit(new_high_text, new_high_rect)
        
        # Neustart-Anweisung
        restart_text = self.small_font.render("Drücke R für Neustart", True, WHITE)
        restart_rect = restart_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50))
        self.screen.blit(restart_text, restart_rect)
    
    def draw(self):
        """Hauptzeichnungsfunktion"""
        if self.state == "MENU":
            self.draw_menu()
        elif self.state in ["PLAYING", "PAUSED"]:
            self.draw_game()
        elif self.state == "GAME_OVER":
            self.draw_game_over()
        
        pygame.display.flip()
    
    def run(self):
        """Hauptspielschleife"""
        running = True
        
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            
            # FPS begrenzen
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

def main():
    """Hauptfunktion"""
    game = Game()
    game.run()

if __name__ == "__main__":
    main()
