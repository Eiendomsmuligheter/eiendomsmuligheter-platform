import promClient from 'prom-client';
import { Express } from 'express';

// Create a Registry to register metrics
const register = new promClient.Registry();

// Add default Node.js metrics
promClient.collectDefaultMetrics({ register });

// Create custom metrics
const httpRequestDurationMicroseconds = new promClient.Histogram({
    name: 'http_request_duration_seconds',
    help: 'Duration of HTTP requests in seconds',
    labelNames: ['method', 'route', 'status_code'],
    buckets: [0.1, 0.5, 1, 2, 5]
});

const propertyAnalysisCounter = new promClient.Counter({
    name: 'property_analysis_total',
    help: 'Total number of property analyses performed'
});

const propertyAnalysisDuration = new promClient.Histogram({
    name: 'property_analysis_duration_seconds',
    help: 'Duration of property analysis in seconds',
    buckets: [1, 5, 10, 30, 60, 120]
});

const documentGenerationCounter = new promClient.Counter({
    name: 'document_generation_total',
    help: 'Total number of documents generated'
});

const modelInferenceCounter = new promClient.Counter({
    name: 'model_inference_total',
    help: 'Total number of AI model inferences'
});

// Register custom metrics
register.registerMetric(httpRequestDurationMicroseconds);
register.registerMetric(propertyAnalysisCounter);
register.registerMetric(propertyAnalysisDuration);
register.registerMetric(documentGenerationCounter);
register.registerMetric(modelInferenceCounter);

export const setupMonitoring = (app: Express) => {
    // Metrics endpoint for Prometheus
    app.get('/metrics', async (req, res) => {
        res.set('Content-Type', register.contentType);
        res.end(await register.metrics());
    });

    // Middleware to measure request duration
    app.use((req, res, next) => {
        const start = Date.now();
        res.on('finish', () => {
            const duration = Date.now() - start;
            httpRequestDurationMicroseconds
                .labels(req.method, req.path, res.statusCode.toString())
                .observe(duration / 1000); // Convert to seconds
        });
        next();
    });
};

export const monitoring = {
    incrementPropertyAnalysis: () => propertyAnalysisCounter.inc(),
    observePropertyAnalysisDuration: (duration: number) => propertyAnalysisDuration.observe(duration),
    incrementDocumentGeneration: () => documentGenerationCounter.inc(),
    incrementModelInference: () => modelInferenceCounter.inc()
};