FROM public.ecr.aws/lambda/nodejs:22

COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build
RUN npm prune --production

CMD ["dist/lambda.handler"]
