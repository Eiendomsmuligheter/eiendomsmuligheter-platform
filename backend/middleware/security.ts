import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { Express } from 'express';

export const configureSecurityMiddleware = (app: Express) => {
    // Basic security headers
    app.use(helmet());

    // Rate limiting
    const limiter = rateLimit({
        windowMs: 15 * 60 * 1000, // 15 minutes
        max: 100, // Limit each IP to 100 requests per windowMs
        message: 'For mange forespørsler fra denne IP-adressen, vennligst prøv igjen senere.'
    });
    app.use('/api/', limiter);

    // CORS configuration
    app.use((req, res, next) => {
        res.setHeader('Access-Control-Allow-Origin', process.env.FRONTEND_URL || 'https://eiendomsmuligheter.no');
        res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
        res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
        res.setHeader('Access-Control-Allow-Credentials', 'true');
        next();
    });

    // XSS Protection
    app.use(helmet.xssFilter());
    
    // Prevent clickjacking
    app.use(helmet.frameguard({ action: 'deny' }));
    
    // Hide X-Powered-By header
    app.use(helmet.hidePoweredBy());
    
    // Content Security Policy
    app.use(helmet.contentSecurityPolicy({
        directives: {
            defaultSrc: ["'self'"],
            scriptSrc: ["'self'", "'unsafe-inline'", "'unsafe-eval'", 'https://cdn.nvidia.com'],
            styleSrc: ["'self'", "'unsafe-inline'", 'https://fonts.googleapis.com'],
            imgSrc: ["'self'", 'data:', 'https:'],
            connectSrc: ["'self'", 'https://api.eiendomsmuligheter.no', 'https://api.stripe.com'],
            fontSrc: ["'self'", 'https://fonts.gstatic.com'],
            objectSrc: ["'none'"],
            mediaSrc: ["'self'"],
            frameSrc: ["'none'"]
        }
    }));

    // HTTP Strict Transport Security
    app.use(helmet.hsts({
        maxAge: 31536000,
        includeSubDomains: true,
        preload: true
    }));
};