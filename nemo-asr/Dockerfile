FROM node:22-slim AS build

WORKDIR /app

COPY asr-websocket-react/package*.json ./asr-websocket-react/
WORKDIR /app/asr-websocket-react
RUN npm ci

COPY asr-websocket-react/ ./
RUN npm run build

FROM node:22-slim

ENV NODE_ENV=production

WORKDIR /app

COPY asr-websocket-react/package*.json ./asr-websocket-react/
WORKDIR /app/asr-websocket-react
RUN npm ci --omit=dev

COPY asr-websocket-react/server ./server
COPY --from=build /app/asr-websocket-react/dist ./dist

EXPOSE 9090

CMD ["node", "server/asr-proxy.mjs"]
