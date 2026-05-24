# Use Node.js 24 LTS slim image
FROM node:24-slim

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install packages
RUN npm install

# Copy source files
COPY . .

# Build Next.js project
RUN npm run build

# Expose port 3000
EXPOSE 3000

# Set production environment variables
ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

# Start Next.js server
CMD ["npm", "start"]
