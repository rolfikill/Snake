#!/bin/bash

# Snake Game Deployment Script für Hetzner Server
# Dieses Script lädt das Spiel auf Ihren Server hoch

echo "🐍 Snake Game Deployment auf Hetzner Server"
echo "=========================================="

# Server-Details (bitte anpassen)
SERVER_USER="root"  # Ihr Server-Username
SERVER_HOST="your-server.hetzner.com"  # Ihre Server-IP oder Domain
SERVER_PATH="/var/www/html/snake"  # Pfad auf dem Server

echo "📁 Repository klonen..."
git clone https://github.com/rolfikill/Snake.git temp_snake

echo "📤 Dateien auf Server hochladen..."
# Erstelle Verzeichnis auf dem Server
ssh $SERVER_USER@$SERVER_HOST "mkdir -p $SERVER_PATH"

# Kopiere Dateien auf den Server
scp -r temp_snake/* $SERVER_USER@$SERVER_HOST:$SERVER_PATH/

echo "🔧 Nginx konfigurieren..."
# Erstelle Nginx-Konfiguration
ssh $SERVER_USER@$SERVER_HOST "cat > /etc/nginx/sites-available/snake << 'EOF'
server {
    listen 80;
    server_name your-domain.com;  # Ihre Domain hier eintragen
    
    root $SERVER_PATH;
    index index.html;
    
    location / {
        try_files \$uri \$uri/ =404;
    }
    
    # CORS für lokale Entwicklung
    add_header Access-Control-Allow-Origin *;
}
EOF"

# Aktiviere die Site
ssh $SERVER_USER@$SERVER_HOST "ln -sf /etc/nginx/sites-available/snake /etc/nginx/sites-enabled/"

# Teste Nginx-Konfiguration
ssh $SERVER_USER@$SERVER_HOST "nginx -t"

# Starte Nginx neu
ssh $SERVER_USER@$SERVER_HOST "systemctl reload nginx"

echo "🧹 Aufräumen..."
rm -rf temp_snake

echo "✅ Deployment abgeschlossen!"
echo "🌐 Ihr Snake-Spiel ist verfügbar unter: http://your-domain.com"
echo "📱 Oder direkt: http://your-domain.com/snake_web.html"
