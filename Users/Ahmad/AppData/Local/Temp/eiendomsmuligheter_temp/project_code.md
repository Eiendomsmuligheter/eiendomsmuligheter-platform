# Eiendomsmuligheter Prosjektkode

## Backend

### app.js
```javascript
const express = require('express');
const cors = require('cors');
const connectDB = require('./config/database');
const authRoutes = require('./routes/auth');
const analysisRoutes = require('./routes/analysis');
const paymentRoutes = require('./routes/payment');

require('dotenv').config();

const app = express();

// Connect to MongoDB
connectDB();

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/analysis', analysisRoutes);
app.use('/api/payment', paymentRoutes);

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ message: 'Noe gikk galt!' });
});

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
    console.log(`Server kjører på port ${PORT}`);
});
```

### server.js
```javascript
// Full server.js code here - copied from above
```

## Docker Configuration

### docker-compose.dev.yml
```yaml
version: '3.8'

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    environment:
      - NODE_ENV=development
      - REACT_APP_API_URL=http://localhost:4000
    depends_on:
      - backend

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.dev
    ports:
      - "4000:4000"
    volumes:
      - ./backend:/app
      - /app/node_modules
    environment:
      - NODE_ENV=development
      - MONGODB_URI=mongodb://mongodb:27017/eiendomsmuligheter
      - JWT_SECRET=dev_secret_key
    depends_on:
      - mongodb

  mongodb:
    image: mongo:latest
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

volumes:
  mongodb_data:
```

## Mappestruktur
```
eiendomsmuligheter/
├── backend/
│   ├── app.js
│   ├── server.js
│   ├── config/
│   ├── controllers/
│   ├── models/
│   ├── routes/
│   └── services/
├── frontend/
│   ├── components/
│   ├── pages/
│   ├── styles/
│   └── public/
├── docker-compose.dev.yml
└── docker-compose.prod.yml
```

## Installasjon og Oppsett

1. Opprett mappestrukturen som vist over
2. Installer nødvendige npm-pakker i både backend og frontend:

Backend pakker som trengs:
```bash
npm install express cors mongoose dotenv jsonwebtoken stripe body-parser
```

Frontend pakker som trengs:
```bash
npm install react react-dom next axios @stripe/stripe-js
```

3. Opprett .env fil i backend mappen:
```env
PORT=4000
MONGODB_URI=mongodb://localhost:27017/eiendomsmuligheter
JWT_SECRET=your_secret_key_here
STRIPE_SECRET_KEY=your_stripe_secret_key
```

4. Start prosjektet:
```bash
# Med Docker
docker-compose -f docker-compose.dev.yml up

# Uten Docker
# I backend mappen:
npm run dev

# I frontend mappen:
npm run dev
```