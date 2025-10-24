FROM nginx:alpine

# Kopiere die Spiel-Dateien
COPY snake_web.html /usr/share/nginx/html/index.html
COPY README.md /usr/share/nginx/html/

# Nginx-Konfiguration
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
