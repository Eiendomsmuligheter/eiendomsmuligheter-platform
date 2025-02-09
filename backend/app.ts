import express from 'express';
import { configureSecurityMiddleware } from './middleware/security';
import propertyRoutes from './routes/property';
import analysisRoutes from './routes/analysis';
import documentRoutes from './routes/documents';
import authRoutes from './routes/auth';
import { errorHandler } from './middleware/errorHandler';
import { setupMonitoring } from './monitoring';

const app = express();

// Configure security middleware
configureSecurityMiddleware(app);

// Setup monitoring
setupMonitoring(app);

// Parse JSON bodies
app.use(express.json({ limit: '50mb' }));

// Parse URL-encoded bodies
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// API routes
app.use('/api/property', propertyRoutes);
app.use('/api/analysis', analysisRoutes);
app.use('/api/documents', documentRoutes);
app.use('/api/auth', authRoutes);

// Error handling
app.use(errorHandler);

const port = process.env.PORT || 8000;

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});