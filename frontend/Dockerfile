FROM node:gallium-buster
WORKDIR /usr/src/frontend
COPY package.json package-lock.json ./
RUN npm install
COPY . ./
RUN npm run build
RUN npm install -g serve
# CMD ["npm", "run", "start"]
CMD ["serve", "-s", "build", "-l", "3000"]